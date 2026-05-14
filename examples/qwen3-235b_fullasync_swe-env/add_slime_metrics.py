"""
add_slime_metrics.py
====================

Wraps `generate_with_codingagent.generate` and instruments `coding_sandbox`
so we get per-rollout metrics that slime doesn't emit out of the box, and
exposes a `--custom-rollout-log-function-path`-compatible logger that
folds them into wandb alongside slime's defaults.

Public surface (referenced by run_swe.sh):
  generate                       — drop-in replacement for the
                                   `--custom-generate-function-path`
  reward_func                    — passthrough for the `--custom-rm-path`
  custom_rollout_log_function    — for `--custom-rollout-log-function-path`

Extra metrics emitted (on top of slime's stock `rollout/*`, `perf/*`):
  prompt_len/{mean,std,min,max,p50,p90,p95,p99}
  throughput/samples_per_sec
  throughput/effective_tokens_per_sec

  sandbox/acquire_time_per_sample_mean        (seconds; ~equal to image-pull + ctor)
  sandbox/tool_calls_per_sample_mean          (total tool dispatches per sample)
  sandbox/tool_time_per_sample_mean           (sum of tool wallclock per sample)

  tool_calls/run_command/count_per_sample_mean
  tool_calls/run_command/time_per_call_mean_s
  ... same six (run_command, read_file, write_file, apply_patch, run_tests, submit)

  swebench/submitted_rate         (fraction of samples that called the submit tool)
  swebench/tests_pass_rate        (fraction where /eval.sh exited 0 in the rollout sandbox)

  policy_staleness/{mean,std,min,max,p50,p90,p95,p99}
    where staleness_per_sample = consumer_rollout_id - sample.policy_version_at_dispatch.

The instrumentation uses an `asyncio`-aware `ContextVar` so per-sample
attribution works correctly even though many `generate(sample, ...)` calls
share one Modal sandbox pool concurrently.
"""

from __future__ import annotations

import contextvars
import threading
import time
from typing import Any

import generate_with_codingagent as _ca_mod
import coding_sandbox as _sandbox_mod
from slime.ray.rollout import (
    compute_metrics_from_samples,
    compute_perf_metrics_from_samples,
)
from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step, compute_statistics, dict_add_prefix
from slime.utils.types import Sample

# Policy version tracking for fully-async staleness measurement.
# Bumped at the end of each `custom_rollout_log_function` call (which fires
# after the trainer's update_weights). `generate()` stamps
# `sample.policy_version_at_dispatch = _consumer_version` at dispatch time.
# Max staleness per sample = (consumer rollout_id at log time) - that stamp.
_consumer_version = 0
_consumer_version_lock = threading.Lock()

# ContextVar keeps per-sample state across awaits without leaking between
# concurrent generate() coroutines.
_sample_metrics: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "_sample_metrics", default=None
)

# The 5 tools the agent can dispatch (`submit` doesn't go through the
# sandbox so it's tracked separately on the per-sample bucket).
_TOOLS_TIMED = ("run_command", "read_file", "write_file", "apply_patch", "run_tests")


# --- coding_sandbox instrumentation -----------------------------------------

_orig_acquire = _sandbox_mod.CodingSandboxPool.acquire
_orig_run_command = _sandbox_mod.CodingSandbox.run_command
_orig_read_file = _sandbox_mod.CodingSandbox.read_file
_orig_write_file = _sandbox_mod.CodingSandbox.write_file
_orig_apply_patch = _sandbox_mod.CodingSandbox.apply_patch
_orig_run_tests = _sandbox_mod.CodingSandbox.run_tests


def _bump_tool_call(tool_name: str, dt: float, errored: bool) -> None:
    """Record one tool dispatch: count++, time += dt, errors += int(errored)."""
    bucket = _sample_metrics.get()
    if bucket is None:
        return
    tc = bucket["tool_calls"].setdefault(
        tool_name, {"count": 0, "time_total": 0.0, "errors": 0}
    )
    tc["count"] += 1
    tc["time_total"] += dt
    if errored:
        tc["errors"] += 1


async def _instrumented_acquire(self, instance_id, **create_kwargs):
    bucket = _sample_metrics.get()
    t0 = time.monotonic()
    sandbox = await _orig_acquire(self, instance_id, **create_kwargs)
    if bucket is not None:
        bucket["acquire_time_total"] += time.monotonic() - t0
        bucket["acquires"] += 1
    return sandbox


def _make_tool_wrapper(orig_fn, tool_name: str):
    async def wrapper(self, *args, **kwargs):
        t0 = time.monotonic()
        errored = False
        try:
            res = await orig_fn(self, *args, **kwargs)
        except Exception:
            errored = True
            raise
        else:
            # CommandResult-shaped results expose `.ok` — count exit_code!=0 as error.
            if hasattr(res, "ok") and not res.ok:
                errored = True
        finally:
            _bump_tool_call(tool_name, time.monotonic() - t0, errored)
        return res
    return wrapper


_sandbox_mod.CodingSandboxPool.acquire = _instrumented_acquire
_sandbox_mod.CodingSandbox.run_command = _make_tool_wrapper(_orig_run_command, "run_command")
_sandbox_mod.CodingSandbox.read_file = _make_tool_wrapper(_orig_read_file, "read_file")
_sandbox_mod.CodingSandbox.write_file = _make_tool_wrapper(_orig_write_file, "write_file")
_sandbox_mod.CodingSandbox.apply_patch = _make_tool_wrapper(_orig_apply_patch, "apply_patch")
_sandbox_mod.CodingSandbox.run_tests = _make_tool_wrapper(_orig_run_tests, "run_tests")


# --- generate / reward passthroughs -----------------------------------------


def _new_bucket() -> dict:
    return {
        "acquire_time_total": 0.0,
        "acquires": 0,
        "tool_calls": {},   # tool_name -> {count, time_total, errors}
    }


async def generate(args, sample: Sample, sampling_params) -> Sample:
    # Mirror the retool wrapper: reset accumulated state so re-dispatched
    # samples (after weight-update aborts) don't carry stale token lists.
    #
    # EXCEPTION: when --partial-rollout is enabled, slime intentionally
    # carries the partial tokens/response/loss_mask across the
    # abort+re-dispatch cycle so the agent can resume. Clearing them
    # here would defeat partial-rollout entirely (and silently — any
    # resume-detection logic checks response_length > 0, which we'd
    # nuke). This guard is defensive: the base agent in this example
    # has an `assert not args.partial_rollout` of its own, but keeping
    # the wrapper consistent with the resumable variant avoids subtle
    # bugs if anyone disables that assert.
    if not getattr(args, "partial_rollout", False):
        sample.rollout_log_probs = None
        sample.tokens = []
        sample.response_length = 0
        sample.response = ""
        sample.loss_mask = []

    # NOTE: store cross-dispatch counters in sample.metadata. Plain
    # attributes (sample.foo = ...) are stripped by Ray serialization when
    # slime puts samples back into the buffer for partial-rollout — only
    # declared dataclass fields like sample.metadata survive the round
    # trip. Lifting these into metadata makes them actually preserved
    # across abort+re-dispatch cycles.
    if sample.metadata is None:
        sample.metadata = {}

    # Sticky-on-first-dispatch policy version stamp.
    if "policy_version_at_dispatch" not in sample.metadata:
        sample.metadata["policy_version_at_dispatch"] = _consumer_version

    # Track re-dispatch count. resume_count == 0 means fresh dispatch;
    # >= 1 means re-dispatched at least once after a prior abort.
    sample.metadata["resume_count"] = sample.metadata.get("resume_count", -1) + 1

    bucket = _new_bucket()
    token = _sample_metrics.set(bucket)
    try:
        sample = await _ca_mod.generate(args, sample, sampling_params)
    finally:
        _sample_metrics.reset(token)

    sample.coding_sandbox_metrics = bucket
    sample.prompt_token_count = max(0, len(sample.tokens) - sample.response_length)
    return sample


async def reward_func(args, sample, **kwargs):
    return await _ca_mod.reward_func(args, sample, **kwargs)


# --- custom rollout log function --------------------------------------------


def _aggregate_sandbox_metrics(samples: list[Sample]) -> dict[str, float]:
    buckets = [getattr(s, "coding_sandbox_metrics", None) for s in samples]
    buckets = [b for b in buckets if b]
    n_samples = len(buckets)
    if n_samples == 0:
        return {}

    total_acquire = sum(b["acquire_time_total"] for b in buckets)
    total_tool_time = sum(
        sum(tc["time_total"] for tc in b["tool_calls"].values()) for b in buckets
    )
    total_tool_calls = sum(
        sum(tc["count"] for tc in b["tool_calls"].values()) for b in buckets
    )

    out: dict[str, float] = {
        "sandbox/acquire_time_per_sample_mean": total_acquire / n_samples,
        "sandbox/tool_calls_per_sample_mean": total_tool_calls / n_samples,
        "sandbox/tool_time_per_sample_mean": total_tool_time / n_samples,
    }

    # Per-tool breakdown
    for tool in _TOOLS_TIMED:
        n_calls = sum(b["tool_calls"].get(tool, {}).get("count", 0) for b in buckets)
        n_errs = sum(b["tool_calls"].get(tool, {}).get("errors", 0) for b in buckets)
        t_total = sum(b["tool_calls"].get(tool, {}).get("time_total", 0.0) for b in buckets)
        out[f"tool_calls/{tool}/count_per_sample_mean"] = n_calls / n_samples
        if n_calls > 0:
            out[f"tool_calls/{tool}/time_per_call_mean_s"] = t_total / n_calls
            out[f"tool_calls/{tool}/error_rate"] = n_errs / n_calls

    return out


def _aggregate_swebench_outcomes(samples: list[Sample]) -> dict[str, float]:
    if not samples:
        return {}
    n = len(samples)
    submitted = sum(1 for s in samples if getattr(s, "submitted", False))
    passed = sum(1 for s in samples if getattr(s, "_swebench_tests_passed", False) is True)
    return {
        "swebench/submitted_rate": submitted / n,
        "swebench/tests_pass_rate": passed / n,
    }


def custom_rollout_log_function(
    rollout_id: int,
    args,
    samples: list[Sample],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    """Returns True so slime's default _log_rollout_data is skipped — we have
    already logged a superset of its keys."""
    log_dict: dict[str, Any] = {**(rollout_extra_metrics or {})}

    # 1) slime's default metrics
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(
        compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/"
    )

    # 2) prompt length distribution (slime tracks response_len only)
    prompt_lens = [
        getattr(s, "prompt_token_count", max(0, len(s.tokens) - s.response_length))
        for s in samples
    ]
    if prompt_lens:
        log_dict |= dict_add_prefix(compute_statistics(prompt_lens), "prompt_len/")

    # 3) throughput in sample units (slime gives token-units only)
    if rollout_time > 0:
        log_dict["throughput/samples_per_sec"] = len(samples) / rollout_time
        total_eff_tokens = sum(s.effective_response_length for s in samples)
        log_dict["throughput/effective_tokens_per_sec"] = total_eff_tokens / rollout_time

    # 4) Coding-sandbox metrics
    log_dict |= _aggregate_sandbox_metrics(samples)

    # 5) SWE-bench outcomes
    log_dict |= _aggregate_swebench_outcomes(samples)

    # 6) Policy staleness — measured in WEIGHT VERSIONS (not rollouts).
    # With --update-weights-interval=N, the policy only changes every N
    # rollouts; samples dispatched and consumed within the same N-rollout
    # window are 0 weight-versions stale, not (delta-rollouts) stale.
    #
    # Stamps live in sample.metadata so they survive Ray serialization
    # on partial-rollout re-dispatch cycles.
    interval = max(1, getattr(args, "update_weights_interval", 1) or 1)
    versions = [
        (s.metadata or {}).get("policy_version_at_dispatch", rollout_id)
        for s in samples
    ]
    stalenesses = [
        max(0, (rollout_id // interval) - (v // interval)) for v in versions
    ]
    if stalenesses:
        log_dict |= dict_add_prefix(
            compute_statistics(stalenesses), "policy_staleness/"
        )

    # 7) Resume counts — direct signal that the sample was re-dispatched.
    # Read from metadata for the same Ray-serialization reason.
    resume_counts = [
        (s.metadata or {}).get("resume_count", 0) for s in samples
    ]
    if resume_counts:
        log_dict |= dict_add_prefix(
            compute_statistics(resume_counts), "resume_count/"
        )
        log_dict["resume_count/n_resumed"] = sum(1 for c in resume_counts if c > 0)

    # Bump `_consumer_version` so samples dispatched after this point are
    # stamped with the post-update-weights policy version.
    global _consumer_version
    _consumer_version = max(_consumer_version, rollout_id + 1)

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step

    print(f"[add_slime_metrics] rollout {rollout_id}: {log_dict}", flush=True)
    logging_utils.log(args, log_dict, step_key="rollout/step")
    return True
