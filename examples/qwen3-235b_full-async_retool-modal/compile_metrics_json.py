#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Compile a slime training-run log into a single per-step + per-rollout JSON.

Usage:
  ./compile_metrics_json.py --log <path> [--out <path>]
  ./compile_metrics_json.py --job <id>  [--out <path>]

The schema is per-step / per-rollout granular with no cross-step aggregates,
so the reader can subset/aggregate as they like.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

# Per-step training metrics live in lines like:
#   ...] model.py:664 - step N: {'train/loss': ..., ...}
# Per-step perf metrics:
#   ...] train_metric_utils.py:44 - perf N: {'perf/...': ..., ...}
# Per-rollout metrics:
#   [add_slime_metrics] rollout N: {'rollout/...': ..., ...}
# Update_weights pushes:
#   [TS] timer.py:32 - Timer update_weights end (elapsed: 44.9s)
# Aborts:
#   Returned aborted group N to data buffer
_TS_PREFIX = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]"

_STEP_RE = re.compile(rf"{_TS_PREFIX}[^[]*model\.py:\d+ - step (\d+): (\{{[^}}]*\}})")
_PERF_RE = re.compile(rf"{_TS_PREFIX}[^[]*train_metric_utils\.py:\d+ - perf (\d+): (\{{[^}}]*\}})")
_ROLLOUT_RE = re.compile(r"\[add_slime_metrics\] rollout (\d+): (\{.*?\})\s*$", re.MULTILINE)
_UW_RE = re.compile(rf"{_TS_PREFIX}[^[]*Timer update_weights end \(elapsed: ([\d.]+)s\)")
_UW_START_RE = re.compile(r"Timer update_weights start")
_ABORT_RE = re.compile(r"Returned aborted group (\d+)")
_ARGS_RE = re.compile(r"Running entrypoint for job [^:]+:\s*(python3?\s+train_async\.py.*?)$", re.MULTILINE)


def parse_metric_dict(s: str) -> dict:
    """Parse a Python-repr dict-of-floats into a real dict (full float precision preserved)."""
    s = s.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def parse_args_dump(text: str) -> dict[str, str | float | bool]:
    """Pull --foo VAL pairs from the train_async.py entrypoint command into a dict."""
    m = _ARGS_RE.search(text)
    if not m:
        return {}
    cmd = m.group(1)
    flags: dict[str, str | float | bool | list] = {}
    tokens = cmd.split()
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            key = t[2:].replace("-", "_")
            # Boolean flag: next token is also a flag, or end
            if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                flags[key] = True
                i += 1
            else:
                # Multi-value flag (e.g. --eval-prompt-data aime <path>)
                vals = []
                j = i + 1
                while j < len(tokens) and not tokens[j].startswith("--"):
                    vals.append(tokens[j])
                    j += 1
                flags[key] = vals[0] if len(vals) == 1 else " ".join(vals)
                # Try numeric coercion
                if isinstance(flags[key], str):
                    try:
                        flags[key] = int(flags[key])
                    except ValueError:
                        try:
                            flags[key] = float(flags[key])
                        except ValueError:
                            pass
                i = j
        else:
            i += 1
    return flags


def find_bringup_timeline(text: str) -> dict:
    """Pull a few significant bring-up wallclock markers out of the log."""
    # First "Server is up and ready to roll!" — last engine ready-ish marker
    m = re.search(rf"{_TS_PREFIX}[^]]*The server is fired up and ready to roll!", text)
    sglang_first_ready = m.group(1) if m else None
    # successfully loaded checkpoint (Megatron actor weight load done)
    m = re.search(rf"{_TS_PREFIX}[^]]*successfully loaded checkpoint", text)
    megatron_loaded = m.group(1) if m else None
    # First update_weights end
    m = _UW_RE.search(text)
    first_uw_end = m.group(1) if m else None
    return {
        "sglang_first_engine_ready": sglang_first_ready,
        "megatron_actor_loaded": megatron_loaded,
        "first_update_weights_done": first_uw_end,
    }


def compile_run(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")

    # Per-step train + perf
    step_to_train = {int(m.group(2)): (m.group(1), parse_metric_dict(m.group(3))) for m in _STEP_RE.finditer(text)}
    step_to_perf = {int(m.group(2)): (m.group(1), parse_metric_dict(m.group(3))) for m in _PERF_RE.finditer(text)}

    # Per-rollout
    rollout_data = {int(m.group(1)): parse_metric_dict(m.group(2)) for m in _ROLLOUT_RE.finditer(text)}

    # Update-weights pushes (elapsed only — no version index, but the order = version-1)
    uw_pushes = [(m.group(1), float(m.group(2))) for m in _UW_RE.finditer(text)]

    # Aborts grouped by push-window (between successive Timer update_weights start markers)
    uw_starts = [m.start() for m in _UW_START_RE.finditer(text)]
    aborts_by_window = []
    for i in range(len(uw_starts)):
        end = uw_starts[i + 1] if i + 1 < len(uw_starts) else len(text)
        chunk = text[uw_starts[i]:end]
        aborts_by_window.append(len(_ABORT_RE.findall(chunk)))

    # Args
    args = parse_args_dump(text)
    interval = args.get("update_weights_interval", 1)

    # Build steps array
    steps = []
    n_steps = max(list(step_to_train.keys()) + list(step_to_perf.keys()) + [-1]) + 1
    for n in range(n_steps):
        train_ts, train = step_to_train.get(n, (None, {}))
        perf_ts, perf = step_to_perf.get(n, (None, {}))
        # weight version active during this trainer step
        # initial push -> V_1 active for steps 0..interval-1
        # post-step (interval-1) push -> V_2 active for steps interval..2*interval-1
        # ...
        weight_version = 1 + (n // interval) if isinstance(interval, int) else None
        steps_since_last_push = (n % interval) if isinstance(interval, int) else None
        # is_push_step: this is the step AFTER which a push fires (i.e. (n+1) % interval == 0)
        is_push_step = ((n + 1) % interval == 0) if isinstance(interval, int) else None

        steps.append({
            "step": n,
            "wallclock": train_ts or perf_ts,
            "weight_version": weight_version,
            "steps_since_last_push": steps_since_last_push,
            "is_push_step": is_push_step,
            "train": {
                k.replace("train/", ""): v for k, v in train.items() if k.startswith("train/")
            },
            "perf": {
                k.replace("perf/", ""): v for k, v in perf.items() if k.startswith("perf/")
            },
        })

    # Build rollouts array
    rollouts = []
    n_rollouts = max(list(rollout_data.keys()) + [-1]) + 1
    for n in range(n_rollouts):
        d = rollout_data.get(n, {})
        # Tag each rollout with the weight_version active when its samples were generated.
        # Rollout N's samples are consumed by step N, so they carry the actor weights
        # at the start of step N → V_(1 + N // interval)
        weight_version = 1 + (n // interval) if isinstance(interval, int) else None
        rollouts.append({
            "rollout_id": n,
            "weight_version_consumed_under": weight_version,
            "lengths": {k.replace("rollout/", "").replace("response_len/", "response_len_"):
                        v for k, v in d.items() if k.startswith("rollout/response_len")
                                                or k.startswith("rollout/zero_std")
                                                or k.startswith("rollout/repetition")
                                                or k.startswith("rollout/truncated")},
            "prompt_len": {k.replace("prompt_len/", ""): v for k, v in d.items() if k.startswith("prompt_len/")},
            "perf": {k.replace("perf/", ""): v for k, v in d.items() if k.startswith("perf/")},
            "throughput": {k.replace("throughput/", ""): v for k, v in d.items() if k.startswith("throughput/")},
            "sandbox": {k.replace("sandbox/", ""): v for k, v in d.items() if k.startswith("sandbox/")},
            "rollout_step": d.get("rollout/step"),
        })

    # Trim args to a useful subset for the config block
    keep_keys = [
        "actor_num_nodes", "actor_num_gpus_per_node", "rollout_num_gpus",
        "tensor_model_parallel_size", "pipeline_model_parallel_size",
        "context_parallel_size", "expert_model_parallel_size",
        "expert_tensor_parallel_size", "decoder_last_pipeline_num_layers",
        "rollout_num_gpus_per_engine", "sglang_mem_fraction_static",
        "rollout_batch_size", "n_samples_per_prompt", "global_batch_size",
        "rollout_max_response_len", "max_tokens_per_gpu", "rollout_temperature",
        "advantage_estimator", "use_kl_loss", "kl_loss_coef", "kl_loss_type",
        "entropy_coef", "eps_clip", "eps_clip_high", "use_tis", "tis_clip",
        "update_weights_interval", "keep_old_actor",
        "optimizer", "lr", "lr_decay_style", "weight_decay",
        "adam_beta1", "adam_beta2", "optimizer_cpu_offload",
        "overlap_cpu_optimizer_d2h_h2d", "use_precision_aware_optimizer",
        "recompute_granularity", "recompute_method", "recompute_num_layers",
        "use_dynamic_batch_size", "sequence_parallel",
        "hf_checkpoint", "ref_load",
        "num_rollout", "num_layers", "num_experts", "moe_router_topk",
    ]
    config = {k: args.get(k) for k in keep_keys if k in args}

    # Map slurm-style num_engines from rollout_num_gpus / rollout_num_gpus_per_engine
    rng = args.get("rollout_num_gpus")
    rngpe = args.get("rollout_num_gpus_per_engine")
    if isinstance(rng, int) and isinstance(rngpe, int) and rngpe > 0:
        config["num_engines_derived"] = rng // rngpe

    samples_per_step = None
    if isinstance(args.get("rollout_batch_size"), int) and isinstance(args.get("n_samples_per_prompt"), int):
        samples_per_step = args["rollout_batch_size"] * args["n_samples_per_prompt"]
    config["samples_per_step_derived"] = samples_per_step

    return {
        "n_steps": len(steps),
        "n_rollouts": len(rollouts),
        "config": config,
        "events": {
            "update_weights_pushes": [
                {"index": i, "version": i + 1, "timestamp": ts, "elapsed_s": elapsed}
                for i, (ts, elapsed) in enumerate(uw_pushes)
            ],
            "aborts_by_push_window": [
                {"window_index": i, "n_aborts": n}
                for i, n in enumerate(aborts_by_window)
            ],
            "bringup": find_bringup_timeline(text),
        },
        "steps": steps,
        "rollouts": rollouts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, help="Path to slime stdout log file.")
    p.add_argument("--job", type=int, help="Slurm job id; resolves to mnt/logs/infx-retool-async-<id>.out.")
    p.add_argument("--mnt", type=Path, default=Path("/home/primeteam/slime/mnt"))
    p.add_argument("--out", type=Path, help="Output JSON path. Default: <log basename>.json")
    args = p.parse_args()

    if args.job:
        args.log = args.mnt / "logs" / f"infx-retool-async-{args.job}.out"
    if not args.log or not args.log.exists():
        print(f"Log not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    out = args.out or args.log.with_suffix(".metrics.json")
    data = compile_run(args.log)

    out.write_text(json.dumps(data, indent=2))
    print(f"Wrote {out}: {data['n_steps']} steps, {data['n_rollouts']} rollouts, {len(data['events']['update_weights_pushes'])} pushes")


if __name__ == "__main__":
    main()
