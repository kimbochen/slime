#!/usr/bin/env python3
"""Tests for examples/swe-bench/metrics.py and log_patch.py.

All cheap unit tests — no Modal, no live anything. Construct synthetic
Samples and assert that metric computation produces the expected keys/values.
Also tests the JSONL append mechanism end-to-end.

Run from anywhere:
    python examples/swe-bench/tests/test_metrics.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE_DIR = HERE.parent
TELEMETRY_DIR = EXAMPLE_DIR / "telemetry"
SLIME_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(TELEMETRY_DIR))   # for `import metrics`, `import log_patch`
sys.path.insert(0, str(EXAMPLE_DIR))
sys.path.insert(0, str(SLIME_ROOT))


# ---------------------------------------------------------------------------
# Synthetic sample factory
# ---------------------------------------------------------------------------


class _FakeStatus:
    """Stand-in for Sample.Status enum members — only `.name` is read by metrics.py."""
    def __init__(self, name: str):
        self.name = name


class _FakeSample:
    """Duck-typed Sample. Avoids importing slime.utils.types (pulls in torch).
    metrics.py only reads .status.name, .reward, .response_length, .tokens, .metadata,
    .weight_versions, and optionally .group_index — that's the whole contract."""
    pass


def _sample(
    *,
    status: str = "COMPLETED",
    reward: float | None = None,
    response_length: int = 100,
    n_tokens: int = 1000,
    turn_count: int = 5,
    tool_call_count: int = 3,
    resume_count: int = 1,
    fail_reason: str | None = None,
    reward_fail_reason: str | None = None,
    inference_time_s: float | None = None,
    sandbox_time_s: float | None = None,
    weight_versions: list[int] | None = None,
    group_index: int | None = None,
):
    s = _FakeSample()
    s.status = _FakeStatus(status)
    s.reward = reward
    s.response_length = response_length
    s.tokens = [0] * n_tokens
    s.metadata = {
        "turn_count": turn_count,
        "tool_call_count": tool_call_count,
        "resume_count": resume_count,
    }
    if fail_reason:
        s.metadata["fail_reason"] = fail_reason
    if reward_fail_reason:
        s.metadata["reward_fail_reason"] = reward_fail_reason
    if inference_time_s is not None:
        s.metadata["inference_time_s"] = inference_time_s
    if sandbox_time_s is not None:
        s.metadata["sandbox_time_s"] = sandbox_time_s
    if weight_versions is not None:
        s.weight_versions = list(weight_versions)
    if group_index is not None:
        s.group_index = group_index
    return s


# ---------------------------------------------------------------------------
# metrics.py tests
# ---------------------------------------------------------------------------


def test_empty_samples_returns_empty_dict() -> None:
    import metrics
    assert metrics._compute_swebench_metrics([]) == {}


def test_resolution_rate_is_mean_reward() -> None:
    import metrics
    samples = [
        _sample(status="COMPLETED", reward=1.0),
        _sample(status="COMPLETED", reward=0.0),
        _sample(status="COMPLETED", reward=1.0),
        _sample(status="COMPLETED", reward=0.0),
    ]
    m = metrics._compute_swebench_metrics(samples)
    assert m["swebench/outcome/resolution_rate"] == 0.5


def test_resolution_rate_treats_none_reward_as_zero() -> None:
    import metrics
    samples = [
        _sample(status="COMPLETED", reward=1.0),
        _sample(status="ABORTED", reward=None),
    ]
    m = metrics._compute_swebench_metrics(samples)
    assert m["swebench/outcome/resolution_rate"] == 0.5     # 1.0 + 0 (None) / 2


def test_status_breakdown_fractions() -> None:
    import metrics
    samples = [
        _sample(status="COMPLETED", reward=1.0),
        _sample(status="COMPLETED", reward=0.0),
        _sample(status="ABORTED"),
        _sample(status="FAILED", fail_reason="create for instance 'x' failed: ..."),
    ]
    m = metrics._compute_swebench_metrics(samples)
    assert m["swebench/outcome/sample_status/completed_frac"] == 0.5
    assert m["swebench/outcome/sample_status/aborted_frac"] == 0.25
    assert m["swebench/outcome/sample_status/failed_frac"] == 0.25
    # Note: truncated_frac is intentionally absent (slime tracks rollout/truncated_ratio)


def test_trajectory_stats_have_mean_median_p95() -> None:
    import metrics
    samples = [
        _sample(turn_count=t, tool_call_count=t - 1, n_tokens=t * 100)
        for t in range(1, 21)   # turn_count 1..20
    ]
    m = metrics._compute_swebench_metrics(samples)
    assert m["swebench/trajectory/turns/mean"] == 10.5
    assert m["swebench/trajectory/turns/median"] == 10.5
    # p95 of 1..20: nearest-rank index = int(0.95*20) = 19 → values[19] = 20
    assert m["swebench/trajectory/turns/p95"] == 20


def test_response_per_turn_clamps_zero_turns() -> None:
    import metrics
    # Sample with turn_count=0 shouldn't divide by zero.
    samples = [
        _sample(response_length=100, turn_count=0),
        _sample(response_length=200, turn_count=4),
    ]
    m = metrics._compute_swebench_metrics(samples)
    # turn=0 → clamped to 1 → 100/1 = 100
    # turn=4 → 200/4 = 50
    # mean = 75
    assert m["swebench/trajectory/response_per_turn/mean"] == 75.0
    assert "swebench/trajectory/response_per_turn/p95" not in m   # p95 omitted by design


def test_sandbox_fail_reason_bucketing() -> None:
    import metrics
    samples = [
        _sample(status="FAILED", fail_reason="create for instance 'foo' failed: ImageNotFoundError"),
        _sample(status="FAILED", fail_reason="reattach to sb-abc failed: NotFoundError"),
        _sample(status="FAILED", fail_reason="sandbox sb-xyz died mid-rollout: ..."),
        _sample(status="FAILED", fail_reason="some unrecognized error message"),
        _sample(status="COMPLETED", reward=1.0),    # no fail_reason
    ]
    m = metrics._compute_swebench_metrics(samples)
    n = 5
    assert m["swebench/fail_reasons/sandbox/create_frac"] == 1 / n
    assert m["swebench/fail_reasons/sandbox/reattach_frac"] == 1 / n
    assert m["swebench/fail_reasons/sandbox/died_mid_rollout_frac"] == 1 / n
    assert m["swebench/fail_reasons/sandbox/other_frac"] == 1 / n


def test_reward_fail_reason_bucketing() -> None:
    import metrics
    samples = [
        _sample(reward=0.0, reward_fail_reason="no model_patch (status=COMPLETED)"),
        _sample(reward=0.0, reward_fail_reason="test_patch apply failed (rc=1): ..."),
        _sample(reward=0.0, reward_fail_reason="model_patch apply failed (rc=1): ..."),
        _sample(reward=0.0, reward_fail_reason="pytest exit 1; tail of output: ..."),
        _sample(reward=0.0, reward_fail_reason="reward sandbox failed: SandboxCreateError(...)"),
        _sample(reward=0.0, reward_fail_reason="something else"),
        _sample(reward=1.0),    # no reward_fail_reason
    ]
    m = metrics._compute_swebench_metrics(samples)
    n = 7
    assert m["swebench/fail_reasons/reward/no_model_patch_frac"] == 1 / n
    assert m["swebench/fail_reasons/reward/test_patch_failed_frac"] == 1 / n
    assert m["swebench/fail_reasons/reward/model_patch_failed_frac"] == 1 / n
    assert m["swebench/fail_reasons/reward/pytest_exit_nonzero_frac"] == 1 / n
    assert m["swebench/fail_reasons/reward/sandbox_error_frac"] == 1 / n
    assert m["swebench/fail_reasons/reward/other_frac"] == 1 / n


def test_resume_count_aggregation() -> None:
    import metrics
    samples = [
        _sample(resume_count=1),   # not resumed
        _sample(resume_count=2),   # resumed once
        _sample(resume_count=3),
        _sample(resume_count=5),
        _sample(resume_count=1),
    ]
    m = metrics._compute_swebench_metrics(samples)
    assert m["swebench/async/resume_count/mean"] == 2.4   # (1+2+3+5+1)/5
    assert m["swebench/async/resume_count/n_resumed"] == 3   # 2,3,5 > 1


def test_no_samples_have_fail_reasons_buckets_are_zero() -> None:
    """Even with zero failures, all bucket keys should be present (= 0.0)
    so wandb chart layout is consistent across rollouts."""
    import metrics
    samples = [_sample(status="COMPLETED", reward=1.0) for _ in range(5)]
    m = metrics._compute_swebench_metrics(samples)
    for bucket in ("create", "reattach", "died_mid_rollout", "other"):
        assert m[f"swebench/fail_reasons/sandbox/{bucket}_frac"] == 0.0


def test_inference_and_sandbox_time_stats() -> None:
    import metrics
    samples = [
        _sample(inference_time_s=1.0, sandbox_time_s=4.0),
        _sample(inference_time_s=2.0, sandbox_time_s=8.0),
        _sample(inference_time_s=3.0, sandbox_time_s=12.0),
    ]
    m = metrics._compute_swebench_metrics(samples)
    assert m["swebench/trajectory/inference_time_s/mean"] == 2.0
    assert m["swebench/trajectory/inference_time_s/median"] == 2.0
    assert m["swebench/trajectory/sandbox_time_s/mean"] == 8.0
    assert m["swebench/trajectory/sandbox_time_s/median"] == 8.0
    # sandbox_time_frac = sum(sandbox)/(sum(inference)+sum(sandbox)) = 24/30 = 0.8
    assert m["swebench/trajectory/sandbox_time_frac"] == 0.8


def test_timing_missing_metadata_defaults_to_zero() -> None:
    """Samples without inference_time_s/sandbox_time_s metadata (e.g., early
    failures before any model call) should contribute 0.0, not crash."""
    import metrics
    samples = [
        _sample(status="FAILED", fail_reason="create for instance 'x' failed: ..."),
        _sample(inference_time_s=2.0, sandbox_time_s=8.0),
    ]
    m = metrics._compute_swebench_metrics(samples)
    # mean over [0.0, 2.0] = 1.0, mean over [0.0, 8.0] = 4.0
    assert m["swebench/trajectory/inference_time_s/mean"] == 1.0
    assert m["swebench/trajectory/sandbox_time_s/mean"] == 4.0
    # frac: 8 / (2 + 8) = 0.8
    assert m["swebench/trajectory/sandbox_time_frac"] == 0.8


def test_sandbox_time_frac_omitted_when_total_zero() -> None:
    """If no sample has any timing recorded (all-zero batch), don't divide
    by zero — just omit the frac key."""
    import metrics
    samples = [_sample(status="FAILED", fail_reason="x") for _ in range(3)]
    m = metrics._compute_swebench_metrics(samples)
    assert "swebench/trajectory/sandbox_time_frac" not in m


def test_policy_staleness_uses_weight_version_times_interval() -> None:
    """staleness = max(0, rollout_id - weight_versions[0] × update_weights_interval).
    With rollout_id=12, interval=5:
      sample wv[0]=1 (dispatched in v=1 cycle): 12 - 5  = 7 (survived past v=1)
      sample wv[0]=2 (dispatched in v=2 cycle): 12 - 10 = 2 (survived past v=2)
      sample wv[0]=3 (dispatched in current cycle): 12 - 15 = max(0,-3) = 0 (fresh)
    """
    import metrics
    samples = [
        _sample(weight_versions=[1]),
        _sample(weight_versions=[2]),
        _sample(weight_versions=[3]),
    ]
    m = metrics._compute_swebench_metrics(samples, rollout_id=12, update_weights_interval=5)
    # values = [7, 2, 0]
    assert m["swebench/async/policy_staleness/mean"] == 9 / 3
    assert m["swebench/async/policy_staleness/median"] == 2.0


def test_policy_staleness_skips_samples_without_weight_versions() -> None:
    """Samples that died before calling SGLang have empty weight_versions; skip."""
    import metrics
    samples = [
        _sample(weight_versions=[1]),     # staleness 8 - 5 = 3
        _sample(weight_versions=[]),       # skipped
        _sample(weight_versions=[1]),      # staleness 8 - 5 = 3
    ]
    m = metrics._compute_swebench_metrics(samples, rollout_id=8, update_weights_interval=5)
    # values = [3, 3]
    assert m["swebench/async/policy_staleness/mean"] == 3.0


def test_policy_staleness_absent_when_rollout_id_unknown() -> None:
    """When _compute_swebench_metrics is called without rollout_id, omit keys."""
    import metrics
    samples = [_sample(weight_versions=[1])]
    m = metrics._compute_swebench_metrics(samples)
    assert "swebench/async/policy_staleness/mean" not in m


def test_policy_staleness_zero_for_fresh_samples_in_current_cycle() -> None:
    """Samples dispatched in the current weight version cycle have staleness 0
    (they haven't survived past their dispatch cycle's end)."""
    import metrics
    # At rollout 17, interval=5: v=4 cycle is rollouts 15-19 (since v=4 became
    # active at rollout 15, calculated as wv * interval = 4 * 5 = 20 — actually
    # the formula treats wv=4 as "version 4 active from rollout 20 onward",
    # so any rollout < 20 gives negative → clamped to 0 → "fresh")
    samples = [_sample(weight_versions=[4])]
    m = metrics._compute_swebench_metrics(samples, rollout_id=17, update_weights_interval=5)
    assert m["swebench/async/policy_staleness/mean"] == 0.0


def test_policy_staleness_uses_default_interval_1_when_unspecified() -> None:
    """Default update_weights_interval=1 means staleness = rollout_id - wv[0]
    (no scaling). Useful when args.update_weights_interval is missing."""
    import metrics
    samples = [_sample(weight_versions=[3])]
    m = metrics._compute_swebench_metrics(samples, rollout_id=10)  # interval defaults to 1
    assert m["swebench/async/policy_staleness/mean"] == 7.0  # 10 - 3*1


def test_group_max_staleness_aggregates_per_group_max() -> None:
    """group_max_staleness = mean across groups of each group's max staleness.
    With rollout_id=12, interval=5:
      group 0: wv [1, 1] → stale [7, 7] → max=7
      group 1: wv [2, 3] → stale [2, 0] → max=2
      Mean of group maxes = (7+2)/2 = 4.5"""
    import metrics
    samples = [
        _sample(weight_versions=[1], group_index=0),  # stale 7
        _sample(weight_versions=[1], group_index=0),  # stale 7 → group 0 max=7
        _sample(weight_versions=[2], group_index=1),  # stale 2
        _sample(weight_versions=[3], group_index=1),  # stale 0 → group 1 max=2
    ]
    m = metrics._compute_swebench_metrics(samples, rollout_id=12, update_weights_interval=5)
    assert m["swebench/async/group_max_staleness/mean"] == 4.5
    # spread: group 0 = 7-7 = 0; group 1 = 2-0 = 2; mean = 1.0
    assert m["swebench/async/group_staleness_spread/mean"] == 1.0
    assert "swebench/async/group_staleness_spread/p95" not in m


def test_group_max_staleness_skips_samples_without_group_index() -> None:
    """Samples missing group_index should be excluded from per-group aggregation
    (but still included in flat policy_staleness)."""
    import metrics
    samples = [
        _sample(weight_versions=[1], group_index=0),   # stale 7
        _sample(weight_versions=[2], group_index=0),   # stale 2 → group 0 max=7
        _sample(weight_versions=[1]),                   # stale 7, no group → flat only
    ]
    m = metrics._compute_swebench_metrics(samples, rollout_id=12, update_weights_interval=5)
    # Flat stats include all 3
    assert m["swebench/async/policy_staleness/mean"] == (7 + 2 + 7) / 3
    # Per-group only includes group 0
    assert m["swebench/async/group_max_staleness/mean"] == 7.0


def test_group_max_staleness_absent_when_no_groups_have_data() -> None:
    """If no samples have BOTH weight_versions AND group_index, per-group
    metrics should be omitted (not zeroed)."""
    import metrics
    samples = [
        _sample(weight_versions=[1]),                   # no group_index
        _sample(group_index=0),                         # no weight_versions
    ]
    m = metrics._compute_swebench_metrics(samples, rollout_id=10, update_weights_interval=5)
    # Flat staleness has the first sample
    assert m["swebench/async/policy_staleness/mean"] == 5.0  # 10 - 1*5
    # But per-group has nothing
    assert "swebench/async/group_max_staleness/mean" not in m
    assert "swebench/async/group_staleness_spread/mean" not in m


def test_group_staleness_spread_is_zero_for_uniform_group() -> None:
    """If all samples in a group have identical weight_versions[0], spread=0."""
    import metrics
    samples = [
        _sample(weight_versions=[1], group_index=0),
        _sample(weight_versions=[1], group_index=0),
        _sample(weight_versions=[1], group_index=0),
    ]
    m = metrics._compute_swebench_metrics(samples, rollout_id=10, update_weights_interval=5)
    assert m["swebench/async/group_max_staleness/mean"] == 5.0  # 10 - 1*5
    assert m["swebench/async/group_staleness_spread/mean"] == 0.0


# ---------------------------------------------------------------------------
# log_patch tests — require the full slime runtime (wandb, ray, etc.)
# Skipped automatically if slime imports fail.
# ---------------------------------------------------------------------------


def _slime_log_importable() -> bool:
    try:
        from slime.utils import logging_utils  # noqa: F401
        return True
    except Exception:
        return False


def test_log_patch_writes_jsonl_when_env_set() -> None:
    """When SLIME_METRICS_JSONL is set, every logging_utils.log call should
    append a JSON line to that file."""
    if not _slime_log_importable():
        print("SKIP: slime.utils.logging_utils not importable (missing wandb/ray/...)")
        return

    import log_patch   # triggers _patch()
    from slime.utils import logging_utils

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        jsonl_path = f.name

    try:
        os.environ["SLIME_METRICS_JSONL"] = jsonl_path

        class FakeArgs:
            use_wandb = False
            use_tensorboard = False
        args = FakeArgs()

        logging_utils.log(args, {"foo": 1.0, "bar": 2.0, "step": 7}, step_key="step")
        logging_utils.log(args, {"baz": 3.0, "step": 8}, step_key="step")

        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        r1 = json.loads(lines[0])
        r2 = json.loads(lines[1])
        assert r1["foo"] == 1.0 and r1["bar"] == 2.0 and r1["step"] == 7
        assert r1["step_key"] == "step"
        assert "timestamp" in r1
        assert r2["baz"] == 3.0 and r2["step"] == 8
    finally:
        os.environ.pop("SLIME_METRICS_JSONL", None)
        os.unlink(jsonl_path)


def test_log_patch_noop_when_env_unset() -> None:
    """When SLIME_METRICS_JSONL is NOT set, the patch should pass through
    silently — no file created, slime's log behaves identically."""
    if not _slime_log_importable():
        print("SKIP: slime.utils.logging_utils not importable")
        return

    import log_patch   # already patched from prior test
    from slime.utils import logging_utils

    os.environ.pop("SLIME_METRICS_JSONL", None)

    class FakeArgs:
        use_wandb = False
        use_tensorboard = False
    args = FakeArgs()

    # Should not raise, should not write anywhere we can observe.
    logging_utils.log(args, {"x": 1}, step_key="x")
    # (Nothing to assert beyond "didn't crash" — passthrough is correct.)


def main() -> int:
    test_fns = [
        # metrics.py
        test_empty_samples_returns_empty_dict,
        test_resolution_rate_is_mean_reward,
        test_resolution_rate_treats_none_reward_as_zero,
        test_status_breakdown_fractions,
        test_trajectory_stats_have_mean_median_p95,
        test_response_per_turn_clamps_zero_turns,
        test_sandbox_fail_reason_bucketing,
        test_reward_fail_reason_bucketing,
        test_resume_count_aggregation,
        test_no_samples_have_fail_reasons_buckets_are_zero,
        test_inference_and_sandbox_time_stats,
        test_timing_missing_metadata_defaults_to_zero,
        test_sandbox_time_frac_omitted_when_total_zero,
        test_policy_staleness_uses_weight_version_times_interval,
        test_policy_staleness_skips_samples_without_weight_versions,
        test_policy_staleness_absent_when_rollout_id_unknown,
        test_policy_staleness_zero_for_fresh_samples_in_current_cycle,
        test_policy_staleness_uses_default_interval_1_when_unspecified,
        test_group_max_staleness_aggregates_per_group_max,
        test_group_max_staleness_skips_samples_without_group_index,
        test_group_max_staleness_absent_when_no_groups_have_data,
        test_group_staleness_spread_is_zero_for_uniform_group,
        # log_patch
        test_log_patch_writes_jsonl_when_env_set,
        test_log_patch_noop_when_env_unset,
    ]
    failures = 0
    for fn in test_fns:
        try:
            fn()
        except Exception as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}")
            print(f"      {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {fn.__name__}")
    print(f"\n{len(test_fns) - failures}/{len(test_fns)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
