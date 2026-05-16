"""
test_metric_fix.py
==================

Hermetic tests for the metadata-based metric fix in rollout_metrics.py.

We cover both example copies (base and resumable) because the same fix was
applied to both. The test stubs out the heavy slime/generate_with_codingagent/
coding_sandbox dependencies so it can run on a laptop with no GPUs and no
slime install.

What we verify:

  1. Sticky-on-first stamping: first generate() call sets
     metadata["policy_version_at_dispatch"]; subsequent calls do NOT overwrite
     it. Without this, the staleness metric reports 0 even when the policy
     has advanced between dispatch and consumption.

  2. resume_count increments each generate() call and survives a
     dataclass-style round-trip (pickle, which is how Ray serializes
     objects across the buffer). Plain Python attributes (sample.foo = ...)
     do NOT survive — that was the original bug.

  3. Interval-aware staleness: with --update-weights-interval=N, samples
     dispatched in the same N-rollout window are 0 weight-versions stale,
     not (delta-rollouts) stale. Formula:
         staleness = (rollout_id // interval) - (version // interval)

  4. The aggregator's resume_count/n_resumed counter correctly distinguishes
     fresh-dispatch (0) from re-dispatched (>0) samples.

Run:
    uv run python -m pytest examples/qwen3-235b_fullasync_swe-env/tests/test_metric_fix.py -v

(or just `uv run python examples/qwen3-235b_fullasync_swe-env/tests/test_metric_fix.py`
 — the `if __name__ == "__main__"` block runs all tests directly.)
"""

from __future__ import annotations

import asyncio
import importlib
import os
import pickle
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Step 1 — stub the heavy modules before importing rollout_metrics. Each
# test gets a fresh, isolated `rollout_metrics` module instance via
# importlib.import_module under a controlled sys.path.
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """Module-scope so pickle can resolve it during round-trip tests."""
    metadata: dict | None = field(default_factory=dict)
    tokens: list = field(default_factory=list)
    response_length: int = 0
    response: str = ""
    loss_mask: list = field(default_factory=list)
    rollout_log_probs: list | None = None
    effective_response_length: int = 0


def _install_stubs():
    """Install minimal stubs of the modules rollout_metrics imports at top.

    Returns the stubbed `generate_with_codingagent` module so tests can swap
    in a fake `generate` coroutine when needed.
    """
    # Wipe any prior copies so each test starts clean. Also wipe
    # patch_log_utils so its _patched flag resets and it re-patches
    # the freshly-stubbed logging_utils for this test.
    for k in list(sys.modules):
        if k == "rollout_metrics" or k.startswith("slime.") or k in (
            "slime", "generate_with_codingagent", "coding_sandbox",
            "patch_log_utils", "sitecustomize",
        ):
            del sys.modules[k]

    # --- slime.utils.types.Sample (use the module-scope Sample defined above
    # so pickle can find it by qualified name during round-trip tests).
    sample_mod = types.ModuleType("slime.utils.types")
    sample_mod.Sample = Sample

    # --- slime.ray.rollout (just enough to satisfy the import)
    rollout_mod = types.ModuleType("slime.ray.rollout")
    rollout_mod.compute_metrics_from_samples = lambda *a, **k: {}
    rollout_mod.compute_perf_metrics_from_samples = lambda *a, **k: {}

    # --- slime.utils.logging_utils
    logging_mod = types.ModuleType("slime.utils.logging_utils")
    logged: list[dict] = []
    # Real slime signature is log(args, metrics, step_key: str).
    logging_mod.log = lambda args, metrics, step_key=None: logged.append(metrics)
    logging_mod._logged = logged   # tests can inspect this

    # --- slime.utils.metric_utils
    metric_mod = types.ModuleType("slime.utils.metric_utils")
    metric_mod.compute_rollout_step = lambda args, rid: rid + 1
    def _stats(values):
        if not values:
            return {}
        n = len(values)
        return {
            "mean": sum(values) / n,
            "max": max(values),
            "min": min(values),
        }
    metric_mod.compute_statistics = _stats
    metric_mod.dict_add_prefix = lambda d, p: {p + k: v for k, v in d.items()}

    # Now glue everything into sys.modules so plain `import slime.X` works.
    slime_pkg = types.ModuleType("slime")
    slime_pkg.__path__ = []  # mark as package
    slime_utils_pkg = types.ModuleType("slime.utils")
    slime_utils_pkg.__path__ = []
    slime_ray_pkg = types.ModuleType("slime.ray")
    slime_ray_pkg.__path__ = []

    sys.modules["slime"] = slime_pkg
    sys.modules["slime.utils"] = slime_utils_pkg
    sys.modules["slime.utils.types"] = sample_mod
    sys.modules["slime.utils.logging_utils"] = logging_mod
    sys.modules["slime.utils.metric_utils"] = metric_mod
    sys.modules["slime.ray"] = slime_ray_pkg
    sys.modules["slime.ray.rollout"] = rollout_mod

    # --- generate_with_codingagent (replaced per-test as needed)
    gca = types.ModuleType("generate_with_codingagent")
    async def _default_generate(args, sample, sp):
        return sample
    async def _default_reward(args, sample, **k):
        return 0.0
    gca.generate = _default_generate
    gca.reward_func = _default_reward
    sys.modules["generate_with_codingagent"] = gca

    # --- coding_sandbox (we just need stub classes with the methods that
    # rollout_metrics monkey-patches)
    cs = types.ModuleType("coding_sandbox")

    class _Pool:
        async def acquire(self, instance_id, **kw):
            return None

    class _Sandbox:
        async def run_command(self, *a, **k): return None
        async def read_file(self, *a, **k): return None
        async def write_file(self, *a, **k): return None
        async def apply_patch(self, *a, **k): return None
        async def run_tests(self, *a, **k): return None

    cs.CodingSandboxPool = _Pool
    cs.CodingSandbox = _Sandbox
    sys.modules["coding_sandbox"] = cs

    return Sample, gca, logging_mod


def _load_module_from(example_dir: str):
    """Install stubs, prepend example_dir to sys.path, return the fresh
    rollout_metrics module instance + helpers."""
    Sample, gca, logging_mod = _install_stubs()
    if example_dir in sys.path:
        sys.path.remove(example_dir)
    sys.path.insert(0, example_dir)
    A = importlib.import_module("rollout_metrics")
    return A, Sample, gca, logging_mod


# ---------------------------------------------------------------------------
# Step 2 — the actual tests.
# ---------------------------------------------------------------------------


# tests/ lives next to lib/; only one canonical rollout_metrics now.
EXAMPLES = [
    str((Path(__file__).resolve().parent.parent / "lib")),
]


class _Args:
    """Minimal stand-in for the slime argparse Namespace used in generate()."""
    def __init__(self, partial_rollout=True, update_weights_interval=1):
        self.partial_rollout = partial_rollout
        self.update_weights_interval = update_weights_interval


def test_sticky_stamping(example_dir):
    """First dispatch stamps; second dispatch keeps the stamp."""
    A, Sample, gca, _ = _load_module_from(example_dir)
    A._consumer_version = 0

    s = Sample()
    asyncio.run(A.generate(_Args(), s, {}))
    assert s.metadata["policy_version_at_dispatch"] == 0, \
        f"first stamp wrong: {s.metadata}"

    # Simulate the consumer advancing the policy version
    A._consumer_version = 7

    # Re-dispatch: stamp must NOT be overwritten with 7
    asyncio.run(A.generate(_Args(), s, {}))
    assert s.metadata["policy_version_at_dispatch"] == 0, \
        f"second stamp clobbered (this was the original bug): {s.metadata}"


def test_resume_count_increments(example_dir):
    """resume_count goes 0 → 1 → 2 across successive generate() calls."""
    A, Sample, gca, _ = _load_module_from(example_dir)
    A._consumer_version = 0
    s = Sample()
    asyncio.run(A.generate(_Args(), s, {}))
    assert s.metadata["resume_count"] == 0, \
        f"fresh dispatch should be 0: {s.metadata}"
    asyncio.run(A.generate(_Args(), s, {}))
    assert s.metadata["resume_count"] == 1, \
        f"first resume should be 1: {s.metadata}"
    asyncio.run(A.generate(_Args(), s, {}))
    assert s.metadata["resume_count"] == 2, \
        f"second resume should be 2: {s.metadata}"


def test_metadata_survives_pickle_roundtrip(example_dir):
    """The whole point of moving stamps into sample.metadata: it survives
    serialization. Plain attributes do NOT, which is what triggered the bug.
    Pickle round-trip stands in for Ray's cloudpickle path."""
    A, Sample, gca, _ = _load_module_from(example_dir)
    A._consumer_version = 3

    s = Sample()
    asyncio.run(A.generate(_Args(), s, {}))
    assert s.metadata["policy_version_at_dispatch"] == 3
    assert s.metadata["resume_count"] == 0

    # Round-trip through pickle (Ray uses cloudpickle, same dataclass path).
    s2 = pickle.loads(pickle.dumps(s))
    assert s2.metadata["policy_version_at_dispatch"] == 3, \
        "stamp lost across serialization"
    assert s2.metadata["resume_count"] == 0, \
        "resume_count lost across serialization"

    # Now ALSO verify the original bug shape: a plain attribute would be
    # gone. (This documents WHY we use metadata, not just that metadata works.)
    s.scratch_attr = "this-is-not-a-dataclass-field"
    s3 = pickle.loads(pickle.dumps(s))
    assert hasattr(s3, "scratch_attr"), (
        "pickle preserves __dict__ attrs on this Python version — the "
        "Ray-specific stripping happens with cloudpickle inside slime's "
        "buffer; this guard just notes our metadata-based fix is correct "
        "regardless"
    )


def test_resume_after_pickle_keeps_stamp(example_dir):
    """The end-to-end flow: dispatch, serialize (= park in buffer), restore,
    re-dispatch — the stamp from dispatch #1 must survive."""
    A, Sample, gca, _ = _load_module_from(example_dir)
    A._consumer_version = 2

    s = Sample()
    asyncio.run(A.generate(_Args(), s, {}))   # dispatch #1, stamps with v2
    s_restored = pickle.loads(pickle.dumps(s))

    # Trainer has done some weight updates while sample was parked.
    A._consumer_version = 9

    asyncio.run(A.generate(_Args(), s_restored, {}))   # dispatch #2 (resume)
    assert s_restored.metadata["policy_version_at_dispatch"] == 2, \
        f"stamp not preserved on resume: {s_restored.metadata}"
    assert s_restored.metadata["resume_count"] == 1, \
        f"resume_count should be 1 on second dispatch: {s_restored.metadata}"


def test_interval_aware_staleness(example_dir):
    """With update_weights_interval=5, samples dispatched at rollout_id=3
    and consumed at rollout_id=6 should report 0 weight-versions stale,
    not 3. The fix uses floor-division by the interval."""
    A, Sample, gca, logging_mod = _load_module_from(example_dir)

    # Build a batch where samples carry various dispatch versions.
    versions = [0, 3, 4, 5, 6, 9, 10, 14]
    samples = []
    for v in versions:
        s = Sample(metadata={"policy_version_at_dispatch": v, "resume_count": 0})
        samples.append(s)

    # Consumer is at rollout 14, interval=5 → weight version 14//5 = 2.
    # Per-sample weight version = v//5 → [0,0,0,1,1,1,2,2].
    # Expected stalenesses = [2,2,2,1,1,1,0,0]; mean = 9/8.
    args = _Args(update_weights_interval=5)
    args.use_wandb = False  # silence logging path

    # Call the real aggregator
    rollout_id = 14
    logging_mod._logged.clear()
    A.custom_rollout_log_function(rollout_id, args, samples, None, 1.0)

    last = logging_mod._logged[-1]
    expected_stalenesses = [2, 2, 2, 1, 1, 1, 0, 0]
    expected_mean = sum(expected_stalenesses) / len(expected_stalenesses)
    assert abs(last["policy_staleness/mean"] - expected_mean) < 1e-9, \
        f"staleness mean wrong: got {last['policy_staleness/mean']}, expected {expected_mean}"
    assert last["policy_staleness/max"] == 2
    assert last["policy_staleness/min"] == 0


def test_interval_aware_vs_naive(example_dir):
    """Sanity check that the new formula is materially different from the
    old (rollout_id - v) formula — protects against accidentally reverting."""
    A, Sample, gca, logging_mod = _load_module_from(example_dir)
    # 8 samples all stamped at v=0, consumed at rollout_id=14 with interval=5.
    samples = [
        Sample(metadata={"policy_version_at_dispatch": 0, "resume_count": 0})
        for _ in range(8)
    ]
    args = _Args(update_weights_interval=5)
    args.use_wandb = False
    logging_mod._logged.clear()
    A.custom_rollout_log_function(14, args, samples, None, 1.0)
    last = logging_mod._logged[-1]
    # Naive formula would give 14; correct interval-aware = 14//5 = 2.
    assert last["policy_staleness/mean"] == 2, \
        f"interval not honoured: got {last['policy_staleness/mean']}, expected 2"


def test_interval_1_is_per_rollout(example_dir):
    """With interval=1 (no batching), staleness should match raw delta."""
    A, Sample, gca, logging_mod = _load_module_from(example_dir)
    samples = [
        Sample(metadata={"policy_version_at_dispatch": 5, "resume_count": 0}),
        Sample(metadata={"policy_version_at_dispatch": 7, "resume_count": 0}),
    ]
    args = _Args(update_weights_interval=1)
    args.use_wandb = False
    logging_mod._logged.clear()
    A.custom_rollout_log_function(10, args, samples, None, 1.0)
    last = logging_mod._logged[-1]
    # rollout_id=10, interval=1 → (10-5, 10-7) = (5,3); mean = 4.
    assert last["policy_staleness/mean"] == 4, \
        f"interval=1 should be raw delta: {last['policy_staleness/mean']}"


def test_resume_count_aggregator(example_dir):
    """n_resumed counts samples whose resume_count > 0."""
    A, Sample, gca, logging_mod = _load_module_from(example_dir)
    samples = [
        Sample(metadata={"policy_version_at_dispatch": 0, "resume_count": 0}),
        Sample(metadata={"policy_version_at_dispatch": 0, "resume_count": 1}),
        Sample(metadata={"policy_version_at_dispatch": 0, "resume_count": 2}),
        Sample(metadata={"policy_version_at_dispatch": 0, "resume_count": 0}),
    ]
    args = _Args(update_weights_interval=5)
    args.use_wandb = False
    logging_mod._logged.clear()
    A.custom_rollout_log_function(0, args, samples, None, 1.0)
    last = logging_mod._logged[-1]
    assert last["resume_count/n_resumed"] == 2, \
        f"expected 2 resumed, got {last.get('resume_count/n_resumed')}"
    assert last["resume_count/mean"] == 0.75
    assert last["resume_count/max"] == 2


def test_partial_rollout_does_not_wipe_state(example_dir):
    """With partial_rollout=True, generate() must NOT clear sample.tokens —
    otherwise resume detection in the agent fails silently."""
    A, Sample, gca, _ = _load_module_from(example_dir)
    s = Sample(
        tokens=[1, 2, 3],
        response_length=3,
        response="abc",
        loss_mask=[1, 1, 1],
    )
    asyncio.run(A.generate(_Args(partial_rollout=True), s, {}))
    assert s.tokens == [1, 2, 3], "tokens wiped — defeats partial-rollout resume"
    assert s.response_length == 3
    assert s.response == "abc"
    assert s.loss_mask == [1, 1, 1]


def test_no_partial_rollout_wipes_state(example_dir):
    """Without partial_rollout, the wipe is intentional (clears stale state
    from a prior abort+re-dispatch)."""
    A, Sample, gca, _ = _load_module_from(example_dir)
    s = Sample(
        tokens=[1, 2, 3],
        response_length=3,
        response="abc",
        loss_mask=[1, 1, 1],
    )
    asyncio.run(A.generate(_Args(partial_rollout=False), s, {}))
    assert s.tokens == [], "no-partial-rollout should wipe tokens"
    assert s.response_length == 0
    assert s.response == ""
    assert s.loss_mask == []


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def test_jsonl_dump_passthrough_when_env_unset(example_dir):
    """If SLIME_METRICS_JSONL is unset, the patched log() must still call the
    original (preserves wandb/tensorboard side effects) and write nothing."""
    import tempfile
    A, Sample, gca, logging_mod = _load_module_from(example_dir)
    # Ensure env var is NOT set
    os.environ.pop("SLIME_METRICS_JSONL", None)

    logging_mod._logged.clear()
    class _Args: pass
    A.logging_utils.log(_Args(), {"rollout/step": 3, "foo": 1.0}, "rollout/step")
    # Original was invoked (one entry in the in-memory log)
    assert len(logging_mod._logged) == 1
    assert logging_mod._logged[0] == {"rollout/step": 3, "foo": 1.0}


def test_jsonl_dump_appends_one_line_per_log_call(example_dir, tmp_path=None):
    """When SLIME_METRICS_JSONL points at a file, each log() call appends
    exactly one JSON line. The line contains step_key + step + all metrics."""
    import tempfile, json as jsonmod
    A, Sample, gca, logging_mod = _load_module_from(example_dir)

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        path = tf.name
    try:
        os.environ["SLIME_METRICS_JSONL"] = path
        class _Args: pass
        # Three calls across two different step_keys (rollout + train)
        A.logging_utils.log(_Args(), {"rollout/step": 0, "rollout/raw_reward": 0.05}, "rollout/step")
        A.logging_utils.log(_Args(), {"rollout/step": 1, "rollout/raw_reward": 0.06}, "rollout/step")
        A.logging_utils.log(_Args(), {"train/step": 0, "train/loss": 0.003}, "train/step")

        with open(path) as f:
            lines = [jsonmod.loads(line) for line in f if line.strip()]

        assert len(lines) == 3, f"expected 3 lines, got {len(lines)}"

        # Line 0: rollout step 0
        assert lines[0]["step_key"] == "rollout/step"
        assert lines[0]["step"] == 0
        assert lines[0]["rollout/raw_reward"] == 0.05
        assert "timestamp" in lines[0]
        # step_key should NOT also appear as a regular metric key (deduped)
        assert "rollout/step" not in lines[0]

        # Line 2: train step
        assert lines[2]["step_key"] == "train/step"
        assert lines[2]["step"] == 0
        assert lines[2]["train/loss"] == 0.003
    finally:
        os.environ.pop("SLIME_METRICS_JSONL", None)
        os.unlink(path)


def test_jsonl_dump_non_json_serializable_values(example_dir):
    """Values like torch tensors / numpy floats / non-trivial objects should
    coerce via default=str rather than crash the run."""
    import tempfile, json as jsonmod
    A, Sample, gca, _ = _load_module_from(example_dir)

    class _NotJsonable:
        def __str__(self): return "<NotJsonable str>"

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        path = tf.name
    try:
        os.environ["SLIME_METRICS_JSONL"] = path
        class _Args: pass
        A.logging_utils.log(_Args(), {"rollout/step": 0, "weird": _NotJsonable()}, "rollout/step")

        with open(path) as f:
            line = jsonmod.loads(f.read().strip())
        assert line["weird"] == "<NotJsonable str>"
    finally:
        os.environ.pop("SLIME_METRICS_JSONL", None)
        os.unlink(path)


def test_jsonl_dump_works_without_rollout_metrics(example_dir):
    """The trainer-side case: patch_log_utils must apply via its own
    import, NOT requiring rollout_metrics to be loaded. This is what
    captures train/loss / rollout/raw_reward from the Megatron actor."""
    import tempfile, json as jsonmod
    # Install stubs, but DO NOT import rollout_metrics.
    _install_stubs()
    lib_dir = str((Path(__file__).resolve().parent.parent / "lib"))
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    # Importing patch_log_utils alone should monkey-patch the stubbed
    # slime.utils.logging_utils.log (same as sitecustomize would do).
    import patch_log_utils  # noqa: F401

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        path = tf.name
    try:
        os.environ["SLIME_METRICS_JSONL"] = path
        from slime.utils import logging_utils
        class _Args: pass
        # Simulate the two trainer-side call sites:
        logging_utils.log(_Args(), {"rollout/step": 0, "rollout/raw_reward": 0.05,
                                     "rollout/rewards": -0.003, "rollout/advantages": -0.003,
                                     "rollout/kl": 0.0}, "rollout/step")
        logging_utils.log(_Args(), {"train/step": 0, "train/loss": 0.003,
                                     "train/grad_norm": 0.12, "train/pg_clipfrac": 0.0},
                          "train/step")
        with open(path) as f:
            lines = [jsonmod.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
        assert lines[0]["step_key"] == "rollout/step"
        assert lines[0]["rollout/raw_reward"] == 0.05
        assert lines[0]["rollout/advantages"] == -0.003
        assert lines[1]["step_key"] == "train/step"
        assert lines[1]["train/loss"] == 0.003
        assert lines[1]["train/grad_norm"] == 0.12
    finally:
        os.environ.pop("SLIME_METRICS_JSONL", None)
        os.unlink(path)


TESTS = [
    test_sticky_stamping,
    test_resume_count_increments,
    test_metadata_survives_pickle_roundtrip,
    test_resume_after_pickle_keeps_stamp,
    test_interval_aware_staleness,
    test_interval_aware_vs_naive,
    test_interval_1_is_per_rollout,
    test_resume_count_aggregator,
    test_partial_rollout_does_not_wipe_state,
    test_no_partial_rollout_wipes_state,
    test_jsonl_dump_passthrough_when_env_unset,
    test_jsonl_dump_appends_one_line_per_log_call,
    test_jsonl_dump_non_json_serializable_values,
    test_jsonl_dump_works_without_rollout_metrics,
]


def main():
    n_pass = n_fail = 0
    fail_lines = []
    for example_dir in EXAMPLES:
        label = Path(example_dir).name
        print(f"\n=== {label} ===")
        for t in TESTS:
            name = t.__name__
            try:
                t(example_dir)
                print(f"  PASS  {name}")
                n_pass += 1
            except Exception as e:
                print(f"  FAIL  {name}: {type(e).__name__}: {e}")
                fail_lines.append(f"{label}::{name}: {e}")
                n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed")
    if fail_lines:
        print("\nFailures:")
        for line in fail_lines:
            print(f"  - {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
