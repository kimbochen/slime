"""SWE-bench rollout metric logger for slime's --custom-rollout-log-function-path.

Aggregates per-sample state (status, reward, metadata) into a fixed set of
swebench/* metrics, calls logging_utils.log so the values reach wandb (and
the JSONL sidecar if log_patch is active), and returns False so slime's
default rollout/perf logging runs additively.

Metric namespace:
    swebench/outcome/                  — task results
    swebench/trajectory/               — model behavior shape + inference/sandbox time
    swebench/fail_reasons/             — categorized failure modes
    swebench/async/                    — resume count + policy staleness

We do NOT log throughput here — slime's default `perf/tokens_per_gpu_per_sec`,
`perf/effective_tokens_per_gpu_per_sec`, `perf/rollout_time` and trainer-side
`perf/wait_time_ratio` cover it.

Wire via:
    --custom-rollout-log-function-path metrics.log_rollout_data
"""

from __future__ import annotations

import logging
import statistics
from typing import Any


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Failure-reason bucketing — match substrings to fixed categories so the
# wandb keys are stable across runs even though the raw fail_reason strings
# carry varying details (exception messages, paths, exit codes).
#
# The substring patterns match the strings set in:
#   - sandbox.py (SandboxCreateError, SandboxReattachError, SandboxDiedError)
#   - generate.py (catches the above, sets sample.metadata["fail_reason"])
#   - reward.py (sets sample.metadata["reward_fail_reason"])
# ---------------------------------------------------------------------------

_SANDBOX_FAIL_BUCKETS = {
    "create for instance":  "create",
    "reattach to":          "reattach",
    "died mid-rollout":     "died_mid_rollout",
}
_SANDBOX_FAIL_BUCKET_NAMES = ["create", "reattach", "died_mid_rollout", "other"]

_REWARD_FAIL_BUCKETS = {
    "no model_patch":           "no_model_patch",
    "test_patch apply failed":  "test_patch_failed",
    "model_patch apply failed": "model_patch_failed",
    "pytest exit":              "pytest_exit_nonzero",
    "reward sandbox failed":    "sandbox_error",
}
_REWARD_FAIL_BUCKET_NAMES = [
    "no_model_patch", "test_patch_failed", "model_patch_failed",
    "pytest_exit_nonzero", "sandbox_error", "other",
]


def _bucket(reason: str, table: dict[str, str]) -> str:
    if not reason:
        return "other"
    for substr, label in table.items():
        if substr in reason:
            return label
    return "other"


def _p95(values: list[float]) -> float:
    """Nearest-rank p95 (no interpolation). Cheap and good enough for log charts."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(0.95 * len(s))))
    return float(s[idx])


def _stats(values: list[float], prefix: str, out: dict, include_p95: bool = True) -> None:
    if not values:
        return
    out[f"{prefix}/mean"] = float(statistics.mean(values))
    out[f"{prefix}/median"] = float(statistics.median(values))
    if include_p95:
        out[f"{prefix}/p95"] = _p95(values)


def _compute_swebench_metrics(samples, rollout_id: int | None = None, update_weights_interval: int = 1) -> dict[str, Any]:
    n = len(samples)
    if n == 0:
        return {}
    m: dict[str, Any] = {}

    # --- swebench/outcome/ ---
    rewards = [float(s.reward or 0.0) for s in samples]
    m["swebench/outcome/resolution_rate"] = sum(rewards) / n

    # 3-way status partition (truncated is covered by slime's rollout/truncated_ratio).
    for status in ("COMPLETED", "ABORTED", "FAILED"):
        count = sum(1 for s in samples if s.status.name == status)
        m[f"swebench/outcome/sample_status/{status.lower()}_frac"] = count / n

    # --- swebench/trajectory/ ---
    turn_counts = [int(s.metadata.get("turn_count", 0)) for s in samples]
    tool_counts = [int(s.metadata.get("tool_call_count", 0)) for s in samples]
    seq_lens = [len(s.tokens) for s in samples]

    _stats(turn_counts, "swebench/trajectory/turns", m)
    _stats(tool_counts, "swebench/trajectory/tool_calls", m)
    _stats(seq_lens, "swebench/trajectory/sequence_length", m)

    # response_per_turn: per-sample response_length / turn_count (clamp turn>=1).
    rpt = [s.response_length / max(1, s.metadata.get("turn_count", 1)) for s in samples]
    _stats(rpt, "swebench/trajectory/response_per_turn", m, include_p95=False)

    # --- swebench/fail_reasons/sandbox/ (all denominators = n samples) ---
    sandbox_counts = {b: 0 for b in _SANDBOX_FAIL_BUCKET_NAMES}
    for s in samples:
        reason = s.metadata.get("fail_reason", "")
        if reason:
            sandbox_counts[_bucket(reason, _SANDBOX_FAIL_BUCKETS)] += 1
    for bucket in _SANDBOX_FAIL_BUCKET_NAMES:
        m[f"swebench/fail_reasons/sandbox/{bucket}_frac"] = sandbox_counts[bucket] / n

    # --- swebench/fail_reasons/reward/ ---
    reward_counts = {b: 0 for b in _REWARD_FAIL_BUCKET_NAMES}
    for s in samples:
        reason = s.metadata.get("reward_fail_reason", "")
        if reason:
            reward_counts[_bucket(reason, _REWARD_FAIL_BUCKETS)] += 1
    for bucket in _REWARD_FAIL_BUCKET_NAMES:
        m[f"swebench/fail_reasons/reward/{bucket}_frac"] = reward_counts[bucket] / n

    # --- swebench/async/resume_count ---
    # resume_count is 1 after a sample's first generate() call, bumps on each resume.
    # n_resumed = count of samples that needed at least one resume (> 1 call).
    resume_counts = [int(s.metadata.get("resume_count", 1)) for s in samples]
    _stats(resume_counts, "swebench/async/resume_count", m)
    m["swebench/async/resume_count/n_resumed"] = sum(1 for c in resume_counts if c > 1)

    # --- swebench/trajectory/{inference,sandbox}_time_s + sandbox_time_frac ---
    # generate.py accumulates per-sample wall time spent in `await post(...)`
    # (SGLang inference) and `await run_tool(...)` (Modal sandbox round-trip).
    # Sum across batch lets us compute "where did time go?" at-a-glance.
    inf_times = [float(s.metadata.get("inference_time_s", 0.0)) for s in samples]
    sb_times = [float(s.metadata.get("sandbox_time_s", 0.0)) for s in samples]
    _stats(inf_times, "swebench/trajectory/inference_time_s", m)
    _stats(sb_times, "swebench/trajectory/sandbox_time_s", m)
    total = sum(inf_times) + sum(sb_times)
    if total > 0:
        m["swebench/trajectory/sandbox_time_frac"] = sum(sb_times) / total

    # --- swebench/async/policy_staleness ---
    # Staleness = max(0, rollout_id - weight_versions[0] × update_weights_interval).
    #
    # Why this formula: weight_versions[0] is SGLang's reported "weight version"
    # at the sample's first generate call (sticky, never overwritten). The
    # version counter ticks once per update_weights call (= every Nth rollout
    # where N = update_weights_interval). Multiplying by interval converts
    # version-count back into rollout-count, so the subtraction is on a
    # consistent scale.
    #
    # Semantic of the result:
    #   0     → sample was first dispatched in the CURRENT weight version cycle
    #            (still fresh, never survived past the end of its dispatch cycle)
    #   N > 0 → sample survived N rollouts past the end of its original version's
    #            cycle (i.e., it was aborted at some weight broadcast and is
    #            still in the system, or it was queued in the async pipeline
    #            past the next broadcast)
    #
    # Earlier attempts and why they're worse:
    #   - rollout_id - weight_versions[0]: unit mismatch, grew unboundedly
    #   - rollout_id - start_rollout_id: only set on aborted samples (slime
    #     stamps it only in abort()), so ~92% of samples were skipped from
    #     the metric, leaving it biased to problem samples only
    #
    # Per-group stats (group_max_staleness, group_staleness_spread) measure
    # WITHIN-GROUP consistency for GSPO/GRPO advantage normalization: if one
    # sample in a group has much higher staleness than its siblings, that
    # group's advantage computation mixes policy versions.
    if rollout_id is not None:
        from collections import defaultdict
        staleness = []
        by_group: dict[int, list[int]] = defaultdict(list)
        for s in samples:
            wv = getattr(s, "weight_versions", None) or []
            if not wv:
                continue
            st = max(0, int(rollout_id) - int(wv[0]) * int(update_weights_interval))
            staleness.append(st)
            gi = getattr(s, "group_index", None)
            if gi is not None:
                by_group[gi].append(st)
        _stats(staleness, "swebench/async/policy_staleness", m)
        if by_group:
            group_maxes = [max(vs) for vs in by_group.values()]
            group_spreads = [max(vs) - min(vs) for vs in by_group.values()]
            _stats(group_maxes, "swebench/async/group_max_staleness", m)
            _stats(group_spreads, "swebench/async/group_staleness_spread", m, include_p95=False)

    return m


def log_rollout_data(
    rollout_id: int,
    args: Any,
    samples: list,
    rollout_extra_metrics: dict | None,
    rollout_time: float,
) -> bool:
    """slime --custom-rollout-log-function-path entry.

    Returns False (additive): slime's default rollout/perf logging still runs.
    Errors are logged and swallowed — metrics computation must not break
    training.
    """
    # print() (not logger) so the message reliably surfaces through Ray's actor
    # stdout capture. logger.warning was getting eaten by something in slime's
    # logging config during 36556 and we had no signal whether this hook fired.
    print(f"[swebench-metrics] log_rollout_data fired: rollout_id={rollout_id}, n_samples={len(samples)}", flush=True)
    try:
        # Imports deferred so the pure metric computation is testable
        # without slime / wandb / ray / etc. installed.
        from slime.utils import logging_utils
        from slime.utils.metric_utils import compute_rollout_step

        uwi = int(getattr(args, "update_weights_interval", 1) or 1)
        metrics = _compute_swebench_metrics(samples, rollout_id=rollout_id, update_weights_interval=uwi)
        if not metrics:
            print(f"[swebench-metrics] empty metrics dict (n_samples={len(samples)}) — returning early", flush=True)
            return False
        metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
        logging_utils.log(args, metrics, step_key="rollout/step")
        print(f"[swebench-metrics] logged {len(metrics)} keys", flush=True)
    except Exception as e:
        print(f"[swebench-metrics] FAILED: {e!r}", flush=True)
        logger.warning(f"swe-bench metrics computation failed: {e!r}")
    return False
