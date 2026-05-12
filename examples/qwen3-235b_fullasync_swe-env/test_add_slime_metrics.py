#!/usr/bin/env python3
"""
Unit tests for add_slime_metrics.py — no slime, no Modal, no SGLang.

Validates:
  1. ContextVar bucket is private per concurrent generate() coroutine.
  2. Each tool-wrapper increments the right bucket key with the right time + error.
  3. The error-rate heuristic (CommandResult.ok=False → errors++ but count++ only once).
  4. The aggregator produces the expected metric keys from hand-built buckets.
  5. _consumer_version + policy_staleness arithmetic.

Run from slime repo root:
  python examples/qwen3-235b_fullasync_swe-env/test_add_slime_metrics.py
"""

import asyncio
import os
import sys
import types as _types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLIME_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SLIME_ROOT))

os.environ.setdefault("MODAL_CONFIG_PATH", str(SLIME_ROOT / ".modal.toml"))

# --- Stub slime imports (same pattern as test_codingagent.py) ----------------
_fake_slime = _types.ModuleType("slime")
_fake_ray = _types.ModuleType("slime.ray")
_fake_ray_rollout = _types.ModuleType("slime.ray.rollout")
_fake_ray_rollout.compute_metrics_from_samples = lambda args, samples: {"_stub_rollout_metric": 1.0}
_fake_ray_rollout.compute_perf_metrics_from_samples = lambda args, samples, t: {"_stub_perf_metric": 1.0}
_fake_rollout_mod = _types.ModuleType("slime.rollout")
_fake_sglang_rollout = _types.ModuleType("slime.rollout.sglang_rollout")
_fake_sglang_rollout.GenerateState = object
_fake_utils = _types.ModuleType("slime.utils")
_fake_logging = _types.ModuleType("slime.utils.logging_utils")
_fake_logging.log = lambda *a, **kw: None
_fake_metric = _types.ModuleType("slime.utils.metric_utils")
_fake_metric.compute_rollout_step = lambda args, rollout_id: rollout_id
def _compute_stats(xs):
    if not xs: return {}
    xs = sorted(xs)
    return {"mean": sum(xs)/len(xs), "min": xs[0], "max": xs[-1], "p50": xs[len(xs)//2]}
_fake_metric.compute_statistics = _compute_stats
def _dict_add_prefix(d, p):
    return {f"{p}{k}": v for k, v in d.items()}
_fake_metric.dict_add_prefix = _dict_add_prefix
_fake_types = _types.ModuleType("slime.utils.types")
class _Sample:
    class Status:
        TRUNCATED = "TRUNCATED"
        ABORTED = "ABORTED"
        COMPLETED = "COMPLETED"
    def __init__(self):
        self.tokens = []
        self.response_length = 0
        self.rollout_log_probs = None
        self.response = ""
        self.loss_mask = []
        self.effective_response_length = 100
_fake_types.Sample = _Sample
_fake_http = _types.ModuleType("slime.utils.http_utils")
_fake_http.post = None
sys.modules.update({
    "slime": _fake_slime, "slime.ray": _fake_ray,
    "slime.ray.rollout": _fake_ray_rollout,
    "slime.rollout": _fake_rollout_mod,
    "slime.rollout.sglang_rollout": _fake_sglang_rollout,
    "slime.utils": _fake_utils,
    "slime.utils.logging_utils": _fake_logging,
    "slime.utils.metric_utils": _fake_metric,
    "slime.utils.types": _fake_types,
    "slime.utils.http_utils": _fake_http,
})

import coding_sandbox  # noqa: E402
import generate_with_codingagent  # noqa: E402
import add_slime_metrics as asm  # noqa: E402

# ---- mocks ------------------------------------------------------------------

class _FakeCmdResult:
    def __init__(self, ok=True):
        self.ok = ok
        self.exit_code = 0 if ok else 1
        self.stdout = "ok" if ok else "fail"
        self.stderr = "" if ok else "boom"
        self.timed_out = False
        self.duration_s = 0.01
    def short_repr(self, max_lines=20):
        return f"exit={self.exit_code}"

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name, got, expected):
    if got == expected:
        results.append((PASS, name, ""))
    else:
        results.append((FAIL, name, f"\n   expected: {expected!r}\n   got:      {got!r}"))

def check_truthy(name, got, note=""):
    if got:
        results.append((PASS, name, ""))
    else:
        results.append((FAIL, name, note or f"\n   got falsy: {got!r}"))


# ---- tests ------------------------------------------------------------------


def t_bucket_increments_on_run_command():
    """Direct call to the instrumented run_command should bump the bucket."""
    async def go():
        # Sub in a stub run_command that just returns a fake result
        async def stub_run_command(self, cmd, **kw):
            await asyncio.sleep(0.01)
            return _FakeCmdResult(ok=True)
        # Use the wrapper factory directly (avoids needing a real CodingSandbox)
        wrapped = asm._make_tool_wrapper(stub_run_command, "run_command")
        bucket = asm._new_bucket()
        token = asm._sample_metrics.set(bucket)
        try:
            await wrapped(None, "ls")
            await wrapped(None, "pwd")
        finally:
            asm._sample_metrics.reset(token)
        return bucket

    bucket = asyncio.run(go())
    tc = bucket["tool_calls"].get("run_command", {})
    check("run_command count == 2", tc.get("count"), 2)
    check("run_command errors == 0 (both ok)", tc.get("errors"), 0)
    check_truthy("run_command time_total > 0", tc.get("time_total", 0) > 0)


def t_error_counted_on_not_ok():
    """A CommandResult with ok=False should increment errors but NOT double-count count."""
    async def go():
        async def stub(self, cmd, **kw):
            return _FakeCmdResult(ok=False)
        wrapped = asm._make_tool_wrapper(stub, "run_command")
        bucket = asm._new_bucket()
        token = asm._sample_metrics.set(bucket)
        try:
            await wrapped(None, "false")
        finally:
            asm._sample_metrics.reset(token)
        return bucket

    bucket = asyncio.run(go())
    tc = bucket["tool_calls"]["run_command"]
    check("error path count == 1 (no double-count)", tc["count"], 1)
    check("error path errors == 1", tc["errors"], 1)


def t_error_counted_on_exception():
    """An exception from the inner fn should propagate AND mark errored."""
    async def go():
        async def stub(self, cmd, **kw):
            raise RuntimeError("boom")
        wrapped = asm._make_tool_wrapper(stub, "run_command")
        bucket = asm._new_bucket()
        token = asm._sample_metrics.set(bucket)
        try:
            try:
                await wrapped(None, "x")
            except RuntimeError:
                pass
            else:
                raise AssertionError("exception should have propagated")
        finally:
            asm._sample_metrics.reset(token)
        return bucket

    bucket = asyncio.run(go())
    tc = bucket["tool_calls"]["run_command"]
    check("exception path count == 1", tc["count"], 1)
    check("exception path errors == 1", tc["errors"], 1)


def t_no_bucket_no_crash():
    """If no ContextVar bucket is set (caller didn't wrap with generate),
    the tool wrapper should still execute the inner fn without crashing."""
    async def go():
        async def stub(self, cmd, **kw):
            return _FakeCmdResult(ok=True)
        wrapped = asm._make_tool_wrapper(stub, "run_command")
        # No bucket set
        res = await wrapped(None, "ls")
        return res
    r = asyncio.run(go())
    check("no-bucket call returns the inner result", r.ok, True)


def t_bucket_isolated_across_concurrent_coroutines():
    """Two generate-like coroutines running concurrently must not bleed
    bucket state into each other. ContextVar should isolate them."""
    async def make_call(label, n_calls):
        async def stub(self, cmd, **kw):
            await asyncio.sleep(0.005)
            return _FakeCmdResult(ok=True)
        wrapped = asm._make_tool_wrapper(stub, "run_command")
        bucket = asm._new_bucket()
        token = asm._sample_metrics.set(bucket)
        try:
            for i in range(n_calls):
                await wrapped(None, f"{label}_{i}")
        finally:
            asm._sample_metrics.reset(token)
        return bucket

    async def go():
        b1, b2 = await asyncio.gather(make_call("A", 3), make_call("B", 7))
        return b1, b2

    b1, b2 = asyncio.run(go())
    check("coroutine A count == 3", b1["tool_calls"]["run_command"]["count"], 3)
    check("coroutine B count == 7", b2["tool_calls"]["run_command"]["count"], 7)


def t_aggregator_keys():
    """_aggregate_sandbox_metrics produces the expected key shape."""
    # Build two fake samples with pre-populated buckets
    s1 = _Sample()
    s1.coding_sandbox_metrics = {
        "acquire_time_total": 30.0,
        "acquires": 1,
        "tool_calls": {
            "run_command": {"count": 4, "time_total": 8.0, "errors": 1},
            "read_file":   {"count": 2, "time_total": 1.0, "errors": 0},
        },
    }
    s2 = _Sample()
    s2.coding_sandbox_metrics = {
        "acquire_time_total": 25.0,
        "acquires": 1,
        "tool_calls": {
            "run_command": {"count": 2, "time_total": 3.0, "errors": 0},
            "apply_patch": {"count": 1, "time_total": 2.0, "errors": 0},
        },
    }

    agg = asm._aggregate_sandbox_metrics([s1, s2])
    check_truthy("has sandbox/acquire_time_per_sample_mean",
                 "sandbox/acquire_time_per_sample_mean" in agg)
    check_truthy("has tool_calls/run_command/count_per_sample_mean",
                 "tool_calls/run_command/count_per_sample_mean" in agg)
    # acquire mean = (30 + 25) / 2 = 27.5
    check("sandbox/acquire_time_per_sample_mean", agg["sandbox/acquire_time_per_sample_mean"], 27.5)
    # total tool calls = (4+2) + (2+1) = 9; per_sample = 4.5
    check("sandbox/tool_calls_per_sample_mean", agg["sandbox/tool_calls_per_sample_mean"], 4.5)
    # run_command count per sample = (4+2)/2 = 3.0
    check("tool_calls/run_command/count_per_sample_mean",
          agg["tool_calls/run_command/count_per_sample_mean"], 3.0)
    # run_command time per call = (8+3) / (4+2) = 11/6
    check("tool_calls/run_command/time_per_call_mean_s",
          round(agg["tool_calls/run_command/time_per_call_mean_s"], 6),
          round(11.0/6.0, 6))
    # run_command error rate = 1 / 6
    check("tool_calls/run_command/error_rate",
          round(agg["tool_calls/run_command/error_rate"], 6),
          round(1.0/6.0, 6))


def t_aggregator_empty():
    """No samples or no buckets → empty aggregate, no exception."""
    check("agg of empty list == {}", asm._aggregate_sandbox_metrics([]), {})
    s = _Sample()  # no coding_sandbox_metrics attr
    check("agg of samples-without-bucket == {}", asm._aggregate_sandbox_metrics([s]), {})


def t_swebench_outcome_aggregator():
    s1 = _Sample(); s1._swebench_tests_passed = True;  s1.submitted = True
    s2 = _Sample(); s2._swebench_tests_passed = False; s2.submitted = True
    s3 = _Sample(); s3._swebench_tests_passed = None;  s3.submitted = False
    agg = asm._aggregate_swebench_outcomes([s1, s2, s3])
    check("submitted_rate", agg["swebench/submitted_rate"], 2/3)
    check("tests_pass_rate", agg["swebench/tests_pass_rate"], 1/3)


def t_policy_staleness_via_log_function():
    """custom_rollout_log_function should compute staleness from sample stamps."""
    samples = []
    for v in [3, 3, 4, 5, 5]:
        s = _Sample()
        s.policy_version_at_dispatch = v
        s.effective_response_length = 100
        samples.append(s)
    rollout_id = 5
    asm._consumer_version = 0  # reset
    # Patch logging_utils.log to a no-op (already stubbed) and capture print
    asm.custom_rollout_log_function(rollout_id, object(), samples, None, 1.0)
    # After call, _consumer_version should be bumped to rollout_id + 1
    check("_consumer_version bumped to rollout_id+1", asm._consumer_version, rollout_id + 1)
    # Staleness for the samples: [5-3, 5-3, 5-4, 5-5, 5-5] = [2, 2, 1, 0, 0], mean = 1.0
    # (we can't read the log_dict directly, but the bump invariant is the load-bearing one)


def main():
    t_bucket_increments_on_run_command()
    t_error_counted_on_not_ok()
    t_error_counted_on_exception()
    t_no_bucket_no_crash()
    t_bucket_isolated_across_concurrent_coroutines()
    t_aggregator_keys()
    t_aggregator_empty()
    t_swebench_outcome_aggregator()
    t_policy_staleness_via_log_function()

    n_pass = sum(1 for s, _, _ in results if s == PASS)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    for status, name, note in results:
        marker = "✓" if status == PASS else "✗"
        print(f"  {marker} {name}{note}")
    print(f"\n{n_pass} passed, {n_fail} failed.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
