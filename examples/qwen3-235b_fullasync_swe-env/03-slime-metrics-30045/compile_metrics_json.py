#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Compile a slime swe-env training-run log into a single per-step + per-rollout JSON.

Adapted from the qwen3-235b retool tests' compile_metrics_json.py for the SWE-bench
agentic-coding workload. The differences vs. the retool variant:

  - sandbox/* metric block is structured per tool (run_command, read_file, write_file,
    apply_patch, run_tests) instead of a single `execute_code` exec.
  - swebench/{submitted_rate, tests_pass_rate} added.
  - policy_staleness/* added.
  - tool_calls/<tool>/{count_per_sample_mean, time_per_call_mean_s, error_rate}
    captured per tool.

Usage:
  ./compile_metrics_json.py --log <path> [--out <path>]
  ./compile_metrics_json.py --job <id>  [--out <path>]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

_TS_PREFIX = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]"
_STEP_RE = re.compile(rf"{_TS_PREFIX}[^[]*model\.py:\d+ - step (\d+): (\{{[^}}]*\}})")
_PERF_RE = re.compile(rf"{_TS_PREFIX}[^[]*train_metric_utils\.py:\d+ - perf (\d+): (\{{[^}}]*\}})")
_DATA_RE = re.compile(rf"{_TS_PREFIX}[^[]*data\.py:\d+ - rollout (\d+): (\{{[^}}]*\}})")
_ROLLOUT_RE = re.compile(r"\[rollout_metrics\] rollout (\d+): (\{.*?\})\s*$", re.MULTILINE)
_UW_RE = re.compile(rf"{_TS_PREFIX}[^[]*Timer update_weights end \(elapsed: ([\d.]+)s\)")
_UW_START_RE = re.compile(r"Timer update_weights start")
_ABORT_RE = re.compile(r"Returned aborted group (\d+)")
_ARGS_RE = re.compile(r"Running entrypoint for job [^:]+:\s*(python3?\s+train_async\.py.*?)$", re.MULTILINE)
_FIRST_READY_RE = re.compile(rf"{_TS_PREFIX}[^]]*The server is fired up and ready to roll!")
_MEGATRON_LOADED_RE = re.compile(rf"{_TS_PREFIX}[^]]*successfully loaded checkpoint")

# Tools the agent dispatches through the sandbox (matches rollout_metrics._TOOLS_TIMED).
_TOOLS = ("run_command", "read_file", "write_file", "apply_patch", "run_tests")


def parse_metric_dict(s: str) -> dict:
    """Parse a Python-repr dict-of-floats into a real dict."""
    s = s.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def parse_args_dump(text: str) -> dict[str, object]:
    m = _ARGS_RE.search(text)
    if not m:
        return {}
    cmd = m.group(1)
    flags: dict[str, object] = {}
    tokens = cmd.split()
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            key = t[2:].replace("-", "_")
            if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                flags[key] = True
                i += 1
            else:
                vals: list[str] = []
                j = i + 1
                while j < len(tokens) and not tokens[j].startswith("--"):
                    vals.append(tokens[j])
                    j += 1
                val: object = vals[0] if len(vals) == 1 else " ".join(vals)
                if isinstance(val, str):
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                flags[key] = val
                i = j
        else:
            i += 1
    return flags


def find_bringup_timeline(text: str) -> dict:
    sglang_first_ready = (m := _FIRST_READY_RE.search(text)) and m.group(1)
    megatron_loaded = (m := _MEGATRON_LOADED_RE.search(text)) and m.group(1)
    first_uw = (m := _UW_RE.search(text)) and m.group(1)
    return {
        "sglang_first_engine_ready": sglang_first_ready,
        "megatron_actor_loaded": megatron_loaded,
        "first_update_weights_done": first_uw,
    }


def _project(d: dict, prefix: str) -> dict:
    """Take all keys under `prefix/` and return them with prefix stripped."""
    return {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}


def compile_run(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")

    step_to_train = {int(m.group(2)): (m.group(1), parse_metric_dict(m.group(3))) for m in _STEP_RE.finditer(text)}
    step_to_perf = {int(m.group(2)): (m.group(1), parse_metric_dict(m.group(3))) for m in _PERF_RE.finditer(text)}
    step_to_data = {int(m.group(2)): (m.group(1), parse_metric_dict(m.group(3))) for m in _DATA_RE.finditer(text)}
    rollout_data = {int(m.group(1)): parse_metric_dict(m.group(2)) for m in _ROLLOUT_RE.finditer(text)}

    uw_pushes = [(m.group(1), float(m.group(2))) for m in _UW_RE.finditer(text)]
    uw_starts = [m.start() for m in _UW_START_RE.finditer(text)]
    aborts_by_window: list[int] = []
    for i in range(len(uw_starts)):
        end = uw_starts[i + 1] if i + 1 < len(uw_starts) else len(text)
        aborts_by_window.append(len(_ABORT_RE.findall(text[uw_starts[i]:end])))

    args = parse_args_dump(text)
    interval = args.get("update_weights_interval", 1)
    if not isinstance(interval, int):
        interval = 1

    n_steps = max(list(step_to_train.keys()) + list(step_to_perf.keys()) + [-1]) + 1
    steps = []
    for n in range(n_steps):
        train_ts, train = step_to_train.get(n, (None, {}))
        perf_ts, perf = step_to_perf.get(n, (None, {}))
        data_ts, data = step_to_data.get(n, (None, {}))
        weight_version = 1 + (n // interval)
        steps.append({
            "step": n,
            "wallclock": train_ts or perf_ts or data_ts,
            "weight_version": weight_version,
            "steps_since_last_push": n % interval,
            "is_push_step": (n + 1) % interval == 0,
            "train": _project(train, "train/"),
            "perf": _project(perf, "perf/"),
            # Trainer-side per-rollout (raw_reward, response_lengths, log_probs, etc.)
            "rollout_consumed": _project(data, "rollout/"),
        })

    n_rollouts = max(list(rollout_data.keys()) + [-1]) + 1
    rollouts = []
    for n in range(n_rollouts):
        d = rollout_data.get(n, {})
        weight_version = 1 + (n // interval)
        rollouts.append({
            "rollout_id": n,
            "weight_version_consumed_under": weight_version,
            "lengths": {
                k.replace("rollout/", "").replace("response_len/", "response_len_"): v
                for k, v in d.items()
                if k.startswith(("rollout/response_len", "rollout/zero_std", "rollout/repetition", "rollout/truncated"))
            },
            "prompt_len": _project(d, "prompt_len/"),
            "perf": _project(d, "perf/"),
            "throughput": _project(d, "throughput/"),
            "sandbox": _project(d, "sandbox/"),
            "tool_calls": {
                tool: {
                    "count_per_sample_mean": d.get(f"tool_calls/{tool}/count_per_sample_mean"),
                    "time_per_call_mean_s": d.get(f"tool_calls/{tool}/time_per_call_mean_s"),
                    "error_rate": d.get(f"tool_calls/{tool}/error_rate"),
                }
                for tool in _TOOLS
            },
            "swebench": _project(d, "swebench/"),
            "policy_staleness": _project(d, "policy_staleness/"),
            "rollout_step": d.get("rollout/step"),
        })

    keep_keys = [
        "actor_num_nodes", "actor_num_gpus_per_node", "rollout_num_gpus",
        "tensor_model_parallel_size", "pipeline_model_parallel_size",
        "context_parallel_size", "expert_model_parallel_size",
        "expert_tensor_parallel_size", "decoder_last_pipeline_num_layers",
        "rollout_num_gpus_per_engine", "sglang_mem_fraction_static",
        "sglang_enable_dp_attention", "sglang_dp_size", "sglang_ep_size",
        "rollout_batch_size", "n_samples_per_prompt", "global_batch_size",
        "rollout_max_response_len", "max_tokens_per_gpu", "rollout_temperature",
        "advantage_estimator", "kl_loss_coef", "kl_loss_type",
        "entropy_coef", "eps_clip", "update_weights_interval",
        "optimizer", "lr", "lr_decay_style", "weight_decay",
        "adam_beta1", "adam_beta2", "optimizer_cpu_offload",
        "overlap_cpu_optimizer_d2h_h2d", "use_precision_aware_optimizer",
        "recompute_granularity", "recompute_method", "recompute_num_layers",
        "use_dynamic_batch_size", "sequence_parallel",
        "hf_checkpoint", "ref_load", "prompt_data",
        "num_rollout", "num_layers", "num_experts", "moe_router_topk",
        "custom_generate_function_path", "custom_rm_path",
        "custom_rollout_log_function_path",
    ]
    config = {k: args.get(k) for k in keep_keys if k in args}

    rng = args.get("rollout_num_gpus")
    rngpe = args.get("rollout_num_gpus_per_engine")
    if isinstance(rng, int) and isinstance(rngpe, int) and rngpe > 0:
        config["num_engines_derived"] = rng // rngpe

    samples_per_step = None
    if isinstance(args.get("rollout_batch_size"), int) and isinstance(args.get("n_samples_per_prompt"), int):
        samples_per_step = args["rollout_batch_size"] * args["n_samples_per_prompt"]
    config["samples_per_step_derived"] = samples_per_step
    config["max_effective_context_derived"] = (
        args["context_parallel_size"] * args["max_tokens_per_gpu"]
        if isinstance(args.get("context_parallel_size"), int) and isinstance(args.get("max_tokens_per_gpu"), int)
        else None
    )

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
                {"window_index": i, "n_aborts": n} for i, n in enumerate(aborts_by_window)
            ],
            "bringup": find_bringup_timeline(text),
        },
        "steps": steps,
        "rollouts": rollouts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, help="Path to slime stdout log file.")
    p.add_argument("--job", type=int, help="Slurm job id; resolves to mnt/logs/infx-swe-64k-<id>.out.")
    p.add_argument("--mnt", type=Path, default=Path("/home/sa-shared/kimbo/slime/mnt"))
    p.add_argument("--out", type=Path, help="Output JSON path. Default: <log basename>.metrics.json")
    args = p.parse_args()

    if args.job:
        args.log = args.mnt / "logs" / f"infx-swe-64k-{args.job}.out"
    if not args.log or not args.log.exists():
        print(f"Log not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    out = args.out or args.log.with_suffix(".metrics.json")
    data = compile_run(args.log)
    out.write_text(json.dumps(data, indent=2))
    print(
        f"Wrote {out}: {data['n_steps']} steps, {data['n_rollouts']} rollouts, "
        f"{len(data['events']['update_weights_pushes'])} pushes"
    )


if __name__ == "__main__":
    main()
