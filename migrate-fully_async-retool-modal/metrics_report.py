#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "rich"]
# ///
"""
metrics_report.py — slime + retool-modal metrics summarizer.

Parses a slime stdout log produced by `add_slime_metrics.custom_rollout_log_function`
plus slime's own `train_metric_utils.log_perf_data_raw` and produces a tight
per-step report of the metrics requested in conversation:

  Trainer:
    - num GPUs, parallelism (TP/PP/CP/EP/ETP/DP), global / micro batch
    - perf/train_wait_time, perf/actor_train_time, perf/step_time
    - perf/wait_time_ratio
    - perf/update_weights_time   (= weight broadcast time)
    - perf/actor_train_tflops, log_probs_tflops, ref_log_probs_tflops
    - samples/sec (TRAIN-ONLY: global_batch_size / actor_train_time)
    - tokens/sec  (TRAIN-ONLY: from slime's perf/actor_train_tok_per_s)

  Generator:
    - num GPUs, parallelism (per-engine TP, EP), samples_per_step / max_concurrency
    - throughput/samples_per_sec
    - avg_inference_time_per_sample (= rollout_time / num_samples)
    - sandbox/total_time_per_sample_mean (Modal acquire+exec; from add_slime_metrics)
    - prompt_len/{mean,p50,p95}      (TOKENS)
    - rollout/response_len/{mean,p50,p95} (TOKENS)
    - effective_response_len/* (TOKENS, observation tokens excluded)
    - rollout/multi_turn_metric/round_number_mean   (= avg #tool calls)

  System efficiency:
    - rollout/repetition_frac, rollout/truncated_ratio
    - rollout/zero_std/count_*
    - rollout/prefix_cache_hit_rate
    - sandbox/error_rate, sandbox/calls_per_sample_mean
    - generator_to_trainer_ratio = rollout_time / actor_train_time
      (>1 means rollout is the bottleneck — fully-async overlap is helping)
    - mfu_pct = perf/actor_train_tflops / hardware peak (H200 BF16 = 989 TFLOPS)

Run:
    chmod +x metrics_report.py
    ./metrics_report.py --log mnt/logs/infx-retool-async-23413.out

  (the shebang invokes `uv run --script`; uv builds an ephemeral venv.)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rich.console import Console
from rich.table import Table

# H200 BF16 peak (TFLOP/s, dense). Used for an MFU estimate.
H200_BF16_PEAK_TFLOPS = 989.0


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

# Lines we care about, examples:
#   "[add_slime_metrics] rollout 0: {'rollout/response_len/mean': ..., ...}"
#   "perf 0: {'perf/actor_train_time': 8.31, ...}"
#   And the leading "Running entrypoint for job ..." line carries every CLI flag.

_ROLLOUT_RE = re.compile(r"\[add_slime_metrics\] rollout (\d+):\s*(\{.*\})")
_PERF_RE = re.compile(r"^perf (\d+):\s*(\{.*\})", re.MULTILINE)
_ENTRY_RE = re.compile(r"Running entrypoint for job [^:]+:\s*python3?\s+train_async\.py\s+(.*?)$", re.MULTILINE)
_UPDW_RE = re.compile(r"Timer update_weights end \(elapsed: ([\d.]+)s\)")


def _safe_eval(s: str) -> dict:
    """Slime / our log function dump dicts via Python's repr; literal_eval is safe."""
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return {}


def parse_log(path: Path) -> tuple[dict, list[dict], list[dict], list[float]]:
    """Returns (config, rollout_metrics_by_id, perf_metrics_by_id, update_weights_seconds)."""
    text = path.read_text(errors="replace")

    config: dict = {}
    m = _ENTRY_RE.search(text)
    if m:
        config = _parse_cli(m.group(1))

    rollouts: dict[int, dict] = {}
    for m in _ROLLOUT_RE.finditer(text):
        rid = int(m.group(1))
        rollouts[rid] = _safe_eval(m.group(2))

    perfs: dict[int, dict] = {}
    for m in _PERF_RE.finditer(text):
        rid = int(m.group(1))
        perfs[rid] = _safe_eval(m.group(2))

    updw = [float(t) for t in _UPDW_RE.findall(text)]
    return config, rollouts, perfs, updw


def _parse_cli(cli: str) -> dict:
    """Best-effort parse of slime's `python3 train_async.py --foo 1 --bar baz` dump."""
    out: dict[str, Any] = {}
    tokens = cli.strip().split()
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            key = t[2:].replace("-", "_")
            # take everything until the next --flag as the value (handles space-separated value lists)
            vals: list[str] = []
            j = i + 1
            while j < len(tokens) and not tokens[j].startswith("--"):
                vals.append(tokens[j])
                j += 1
            if not vals:
                out[key] = True
            elif len(vals) == 1:
                out[key] = _coerce(vals[0])
            else:
                out[key] = [_coerce(v) for v in vals]
            i = j
        else:
            i += 1
    return out


def _coerce(s: str) -> Any:
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


# ---------------------------------------------------------------------------
# derived metrics + report
# ---------------------------------------------------------------------------


def _get(d: dict, *keys: str, default: float | None = None) -> float | None:
    for k in keys:
        if k in d:
            v = d[k]
            return float(v) if isinstance(v, (int, float)) else v
    return default


def _summary_row(rollout: dict, perf: dict, cfg: dict, updw: float | None) -> dict:
    actor_gpus = cfg.get("actor_num_nodes", 0) * cfg.get("actor_num_gpus_per_node", 0)
    rollout_gpus = cfg.get("rollout_num_gpus", 0)

    train_time = _get(perf, "perf/actor_train_time", default=0.0) or 0.0
    step_time = _get(perf, "perf/step_time", default=train_time + (_get(perf, "perf/train_wait_time") or 0.0))
    wait_time = _get(perf, "perf/train_wait_time", default=0.0) or 0.0
    rollout_time = _get(rollout, "perf/rollout_time", default=None)

    gbs = cfg.get("global_batch_size", 0)
    n_per_prompt = cfg.get("n_samples_per_prompt", 1)
    rollout_batch = cfg.get("rollout_batch_size", 0)
    samples_per_step = rollout_batch * n_per_prompt

    samples_per_sec_train_only = (gbs / train_time) if train_time > 0 else None
    tokens_per_sec_train_only = _get(perf, "perf/actor_train_tok_per_s")
    tflops = _get(perf, "perf/actor_train_tflops")
    mfu = (tflops / H200_BF16_PEAK_TFLOPS * 100) if (tflops and actor_gpus) else None

    samples_per_sec_gen = _get(rollout, "throughput/samples_per_sec")
    avg_infer = (rollout_time / samples_per_step) if (rollout_time and samples_per_step) else None
    sandbox_per_sample = _get(rollout, "sandbox/total_time_per_sample_mean")

    return {
        "actor_train_time_s": train_time,
        "step_time_s": step_time,
        "train_wait_time_s": wait_time,
        "wait_time_ratio": _get(perf, "perf/wait_time_ratio"),
        "weight_broadcast_s": updw,
        "actor_train_tflops": tflops,
        "log_probs_tflops": _get(perf, "perf/log_probs_tflops"),
        "ref_log_probs_tflops": _get(perf, "perf/ref_log_probs_tflops"),
        "samples_per_sec_train_only": samples_per_sec_train_only,
        "tokens_per_sec_train_only": tokens_per_sec_train_only,
        "mfu_pct": mfu,
        "rollout_time_s": rollout_time,
        "samples_per_sec_gen": samples_per_sec_gen,
        "avg_inference_time_per_sample_s": avg_infer,
        "avg_sandbox_time_per_sample_s": sandbox_per_sample,
        "sandbox_acquire_per_call_s": _get(rollout, "sandbox/acquire_time_per_call_mean"),
        "sandbox_exec_per_call_s": _get(rollout, "sandbox/exec_time_per_call_mean"),
        "sandbox_calls_per_sample": _get(rollout, "sandbox/calls_per_sample_mean"),
        "sandbox_error_rate": _get(rollout, "sandbox/error_rate"),
        "prompt_len_mean_tokens": _get(rollout, "prompt_len/mean"),
        "prompt_len_p50_tokens": _get(rollout, "prompt_len/p50"),
        "prompt_len_p95_tokens": _get(rollout, "prompt_len/p95"),
        "response_len_mean_tokens": _get(rollout, "rollout/response_len/mean"),
        "response_len_p50_tokens": _get(rollout, "rollout/response_len/p50"),
        "response_len_p95_tokens": _get(rollout, "rollout/response_len/p95"),
        "effective_response_len_mean_tokens": _get(rollout, "rollout/effective_response_len/mean"),
        "tool_calls_mean": _get(rollout, "rollout/multi_turn_metric/round_number_mean"),
        "prefix_cache_hit_rate": _get(rollout, "rollout/prefix_cache_hit_rate"),
        "truncated_ratio": _get(rollout, "rollout/truncated_ratio"),
        "repetition_frac": _get(rollout, "rollout/repetition_frac"),
        "gen_to_trainer_ratio": (rollout_time / train_time) if (rollout_time and train_time) else None,
    }


def _config_block(cfg: dict, console: Console) -> None:
    actor_gpus = cfg.get("actor_num_nodes", 0) * cfg.get("actor_num_gpus_per_node", 0)
    rollout_gpus = cfg.get("rollout_num_gpus", 0)

    t = Table(title="Trainer config", show_header=True, header_style="bold")
    t.add_column("key"); t.add_column("value")
    t.add_row("num_gpus", f"{actor_gpus}  ({cfg.get('actor_num_nodes')} × {cfg.get('actor_num_gpus_per_node')})")
    t.add_row(
        "parallelism",
        f"TP={cfg.get('tensor_model_parallel_size', 1)} "
        f"PP={cfg.get('pipeline_model_parallel_size', 1)} "
        f"CP={cfg.get('context_parallel_size', 1)} "
        f"EP={cfg.get('expert_model_parallel_size', 1)} "
        f"ETP={cfg.get('expert_tensor_parallel_size', 1)}"
    )
    t.add_row("global_batch_size", str(cfg.get("global_batch_size")))
    t.add_row("max_tokens_per_gpu", str(cfg.get("max_tokens_per_gpu")))
    t.add_row("recompute", f"{cfg.get('recompute_granularity')} / {cfg.get('recompute_method')} / {cfg.get('recompute_num_layers')}")
    console.print(t)

    t2 = Table(title="Generator config", show_header=True, header_style="bold")
    t2.add_column("key"); t2.add_column("value")
    n_engines = rollout_gpus // cfg.get("rollout_num_gpus_per_engine", 1) if cfg.get("rollout_num_gpus_per_engine") else None
    t2.add_row("num_gpus", str(rollout_gpus))
    t2.add_row("rollout_num_gpus_per_engine (= TP per engine)", str(cfg.get("rollout_num_gpus_per_engine")))
    t2.add_row("num_engines", str(n_engines))
    samples_per_step = cfg.get("rollout_batch_size", 0) * cfg.get("n_samples_per_prompt", 1)
    t2.add_row("samples_per_step (max concurrency)", f"{samples_per_step}  ({cfg.get('rollout_batch_size')} × {cfg.get('n_samples_per_prompt')})")
    t2.add_row("rollout_max_response_len (tokens)", str(cfg.get("rollout_max_response_len")))
    console.print(t2)


def _per_step_table(rollouts: dict, perfs: dict, cfg: dict, updws: list[float], console: Console) -> None:
    rows: list[dict] = []
    for rid in sorted(set(rollouts) | set(perfs)):
        rollout = rollouts.get(rid, {})
        perf = perfs.get(rid, {})
        # update_weights happens once per step; assume sequential — pick rid-th element if available
        updw = updws[rid] if rid < len(updws) else None
        rows.append({"rollout_id": rid, **_summary_row(rollout, perf, cfg, updw)})

    if not rows:
        console.print("[yellow]No rollout/perf events found in log yet.[/yellow]")
        return

    # Trainer-side per-step view
    t = Table(title="Trainer per-step (TRAIN-ONLY throughput excludes wait time)", show_header=True, header_style="bold")
    t.add_column("rollout"); t.add_column("step_s", justify="right"); t.add_column("train_s", justify="right")
    t.add_column("wait_s", justify="right"); t.add_column("wait_ratio", justify="right")
    t.add_column("updw_s", justify="right")
    t.add_column("samples/s", justify="right"); t.add_column("tok/s", justify="right")
    t.add_column("TFLOP/s", justify="right"); t.add_column("MFU%", justify="right")
    for r in rows:
        t.add_row(
            str(r["rollout_id"]),
            _f(r["step_time_s"], "{:.2f}"),
            _f(r["actor_train_time_s"], "{:.2f}"),
            _f(r["train_wait_time_s"], "{:.2f}"),
            _f(r["wait_time_ratio"], "{:.1%}"),
            _f(r["weight_broadcast_s"], "{:.2f}"),
            _f(r["samples_per_sec_train_only"], "{:.1f}"),
            _f(r["tokens_per_sec_train_only"], "{:.0f}"),
            _f(r["actor_train_tflops"], "{:.0f}"),
            _f(r["mfu_pct"], "{:.1f}"),
        )
    console.print(t)

    # Generator-side per-step view
    t = Table(title="Generator per-step (lengths in TOKENS)", show_header=True, header_style="bold")
    t.add_column("rollout"); t.add_column("rollout_s", justify="right")
    t.add_column("samples/s", justify="right"); t.add_column("infer_s/sample", justify="right")
    t.add_column("sandbox_s/sample", justify="right"); t.add_column("calls/sample", justify="right")
    t.add_column("err%", justify="right")
    t.add_column("prompt_len μ", justify="right"); t.add_column("resp_len μ", justify="right")
    t.add_column("eff_resp μ", justify="right"); t.add_column("tool_rounds μ", justify="right")
    for r in rows:
        t.add_row(
            str(r["rollout_id"]),
            _f(r["rollout_time_s"], "{:.2f}"),
            _f(r["samples_per_sec_gen"], "{:.2f}"),
            _f(r["avg_inference_time_per_sample_s"], "{:.2f}"),
            _f(r["avg_sandbox_time_per_sample_s"], "{:.2f}"),
            _f(r["sandbox_calls_per_sample"], "{:.2f}"),
            _f(r["sandbox_error_rate"], "{:.1%}"),
            _f(r["prompt_len_mean_tokens"], "{:.0f}"),
            _f(r["response_len_mean_tokens"], "{:.0f}"),
            _f(r["effective_response_len_mean_tokens"], "{:.0f}"),
            _f(r["tool_calls_mean"], "{:.2f}"),
        )
    console.print(t)

    # System efficiency
    t = Table(title="System efficiency", show_header=True, header_style="bold")
    t.add_column("rollout"); t.add_column("gen/train ratio", justify="right")
    t.add_column("prefix_cache_hit", justify="right"); t.add_column("truncated", justify="right")
    t.add_column("repetition", justify="right"); t.add_column("p95 prompt", justify="right")
    t.add_column("p95 resp", justify="right")
    for r in rows:
        t.add_row(
            str(r["rollout_id"]),
            _f(r["gen_to_trainer_ratio"], "{:.2f}"),
            _f(r["prefix_cache_hit_rate"], "{:.1%}"),
            _f(r["truncated_ratio"], "{:.1%}"),
            _f(r["repetition_frac"], "{:.1%}"),
            _f(r["prompt_len_p95_tokens"], "{:.0f}"),
            _f(r["response_len_p95_tokens"], "{:.0f}"),
        )
    console.print(t)

    # Aggregate (last-3 mean) summary
    last3 = rows[-3:]
    if len(last3) >= 1:
        agg = {k: np.mean([float(r[k]) for r in last3 if r.get(k) is not None])
               for k in last3[0] if k != "rollout_id"}
        t = Table(title=f"Steady-state mean (last {len(last3)} rollout(s))", show_header=True, header_style="bold")
        t.add_column("metric"); t.add_column("value", justify="right")
        for k, v in agg.items():
            if np.isnan(v):
                continue
            fmt = "{:.2%}" if any(s in k for s in ("ratio", "frac", "rate", "_pct")) else "{:.3f}"
            t.add_row(k, fmt.format(v) if not np.isnan(v) else "—")
        console.print(t)


def _f(v: Any, fmt: str = "{}") -> str:
    if v is None:
        return "—"
    try:
        if isinstance(v, float) and np.isnan(v):
            return "—"
        return fmt.format(v)
    except Exception:
        return str(v)


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--log", type=Path, help="Path to slime stdout log file (e.g. mnt/logs/infx-retool-async-23413.out)")
    p.add_argument("--job", type=int, help="Slurm job id; resolves --log to mnt/logs/infx-retool-async-<job>.out")
    p.add_argument("--mnt", type=Path, default=Path("/home/sa-shared/kimbo/slime/mnt"),
                   help="Slime mnt root for --job lookup (default: /home/sa-shared/kimbo/slime/mnt)")
    p.add_argument("--latest", action="store_true", help="Print only the most recent rollout")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of tables")
    args = p.parse_args(argv)

    if args.log is None and args.job is not None:
        args.log = args.mnt / "logs" / f"infx-retool-async-{args.job}.out"
    if args.log is None:
        p.error("Pass --log <path> or --job <slurm-id>")
    if not args.log.exists():
        p.error(f"Log not found: {args.log}")

    cfg, rollouts, perfs, updws = parse_log(args.log)

    if args.latest:
        if rollouts:
            last = max(rollouts.keys())
            rollouts = {last: rollouts[last]}
        if perfs:
            last = max(perfs.keys())
            perfs = {last: perfs[last]}

    if args.json:
        rows = []
        for rid in sorted(set(rollouts) | set(perfs)):
            rows.append({
                "rollout_id": rid,
                **_summary_row(rollouts.get(rid, {}), perfs.get(rid, {}), cfg,
                               updws[rid] if rid < len(updws) else None),
            })
        print(json.dumps({"config": cfg, "rows": rows}, indent=2, default=str))
        return 0

    console = Console()
    console.print(f"[dim]Log:[/dim] {args.log}")
    console.print(f"[dim]Found:[/dim] {len(rollouts)} rollout-metric blocks, {len(perfs)} perf blocks, "
                  f"{len(updws)} update_weights timings")
    if cfg:
        _config_block(cfg, console)
    _per_step_table(rollouts, perfs, cfg, updws, console)
    return 0


if __name__ == "__main__":
    sys.exit(main())
