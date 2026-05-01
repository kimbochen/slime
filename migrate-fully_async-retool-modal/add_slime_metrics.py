"""
add_slime_metrics.py
====================

Wraps `generate_with_retool.generate` and instruments `modal_tool_sandbox`
so we get per-rollout metrics that slime doesn't emit out of the box, and
exposes a `--custom-rollout-log-function-path`-compatible logger that
folds them into wandb alongside slime's defaults.

Public surface (referenced by run_async.sh):
  generate                       — drop-in replacement for the
                                   `--custom-generate-function-path`
  reward_func                    — passthrough for the `--custom-rm-path`
  custom_rollout_log_function    — for `--custom-rollout-log-function-path`

Extra metrics emitted (on top of slime's stock `rollout/*`, `perf/*`):
  prompt_len/{mean,std,min,max,p50,p90,p95,p99}
  throughput/samples_per_sec
  throughput/effective_tokens_per_sec
  sandbox/acquire_time_per_call_mean       (seconds)
  sandbox/exec_time_per_call_mean          (seconds, exec only — acquire excluded)
  sandbox/total_time_per_call_mean         (seconds, acquire + exec)
  sandbox/calls_per_sample_mean
  sandbox/error_rate                       (fraction of execs that returned "Error: ...")
  sandbox/total_time_per_sample_mean       (seconds spent in sandbox per sample)

The instrumentation uses an `asyncio`-aware `ContextVar` so per-sample
attribution works correctly even though many `generate(sample, ...)` calls
share one Modal sandbox pool concurrently.
"""

from __future__ import annotations

import contextvars
import time
from typing import Any

import generate_with_retool as _retool_mod
import modal_tool_sandbox as _sandbox_mod
from slime.ray.rollout import (
    compute_metrics_from_samples,
    compute_perf_metrics_from_samples,
)
from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step, compute_statistics, dict_add_prefix
from slime.utils.types import Sample

# ContextVar keeps per-sample state across awaits without leaking between
# concurrent generate() coroutines.
_sample_metrics: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "_sample_metrics", default=None
)


# --- Modal sandbox instrumentation ------------------------------------------

_orig_acquire = _sandbox_mod._acquire
_orig_execute_code = _sandbox_mod.ModalPythonSandbox.execute_code


async def _instrumented_acquire():
    bucket = _sample_metrics.get()
    t0 = time.monotonic()
    sb = await _orig_acquire()
    if bucket is not None:
        bucket["acquire_time_total"] += time.monotonic() - t0
        bucket["acquires"] += 1
    return sb


async def _instrumented_execute_code(self, code: str) -> str:
    bucket = _sample_metrics.get()
    t0 = time.monotonic()
    result = await _orig_execute_code(self, code)
    if bucket is not None:
        bucket["execute_code_time_total"] += time.monotonic() - t0
        bucket["execute_code_calls"] += 1
        if isinstance(result, str) and result.startswith("Error:"):
            bucket["execute_code_errors"] += 1
    return result


_sandbox_mod._acquire = _instrumented_acquire
_sandbox_mod.ModalPythonSandbox.execute_code = _instrumented_execute_code


# --- generate / reward passthroughs -----------------------------------------


def _new_bucket() -> dict:
    return {
        "acquire_time_total": 0.0,
        "acquires": 0,
        "execute_code_time_total": 0.0,
        "execute_code_calls": 0,
        "execute_code_errors": 0,
    }


async def generate(args, sample: Sample, sampling_params) -> Sample:
    # IMPORTANT: When the fully-async worker aborts an in-flight rollout
    # (e.g., during update_weights), the sample is returned to the buffer
    # and re-dispatched. The retool generate function APPENDS to
    # sample.rollout_log_probs (and friends) without first resetting them,
    # so a retried sample carries stale state from the prior attempt and
    # ends up with len(rollout_log_probs) > len(response_token_ids), which
    # later fires AssertionError("Token/logp length mismatch ...") and
    # downstream `slice_log_prob_with_cp` asserts. Clear here.
    sample.rollout_log_probs = None
    sample.tokens = []
    sample.response_length = 0
    sample.response = ""
    sample.loss_mask = []

    bucket = _new_bucket()
    token = _sample_metrics.set(bucket)
    try:
        sample = await _retool_mod.generate(args, sample, sampling_params)
    finally:
        _sample_metrics.reset(token)

    sample.modal_metrics = bucket
    sample.prompt_token_count = max(0, len(sample.tokens) - sample.response_length)
    return sample


async def reward_func(args, sample, **kwargs):
    return await _retool_mod.reward_func(args, sample, **kwargs)


# --- custom rollout log function --------------------------------------------


def _aggregate_sandbox_metrics(samples: list[Sample]) -> dict[str, float]:
    buckets = [getattr(s, "modal_metrics", None) for s in samples]
    buckets = [b for b in buckets if b]
    if not buckets:
        return {}

    total_acquire = sum(b["acquire_time_total"] for b in buckets)
    total_exec = sum(b["execute_code_time_total"] for b in buckets)
    total_calls = sum(b["execute_code_calls"] for b in buckets)
    total_errors = sum(b["execute_code_errors"] for b in buckets)
    n_samples = len(buckets)

    out: dict[str, float] = {
        "calls_per_sample_mean": total_calls / n_samples,
        "total_time_per_sample_mean": (total_acquire + total_exec) / n_samples,
    }
    if total_calls > 0:
        out["acquire_time_per_call_mean"] = total_acquire / total_calls
        out["exec_time_per_call_mean"] = (total_exec - total_acquire) / total_calls
        out["total_time_per_call_mean"] = (total_acquire + total_exec) / total_calls
        out["error_rate"] = total_errors / total_calls
    return out


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

    # 4) Modal sandbox metrics
    log_dict |= dict_add_prefix(_aggregate_sandbox_metrics(samples), "sandbox/")

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step

    print(f"[add_slime_metrics] rollout {rollout_id}: {log_dict}", flush=True)
    logging_utils.log(args, log_dict, step_key="rollout/step")
    return True
