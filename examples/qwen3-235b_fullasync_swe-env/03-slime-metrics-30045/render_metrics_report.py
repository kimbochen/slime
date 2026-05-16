#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["rich>=13.0"]
# ///
"""Render metrics.json into a readable text report (tables) matching the
swe-env-64k-30045 schema.

Usage:
  ./render_metrics_report.py --in metrics.json --out metrics_report.txt
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from rich.console import Console
from rich.table import Table


def fmt(x, n=2):
    if x is None: return "—"
    if isinstance(x, float): return f"{x:.{n}f}"
    return str(x)


def cfg_tables(c: dict):
    t = Table(title="Trainer config", show_lines=False)
    t.add_column("key"); t.add_column("value")
    pp = c.get("pipeline_model_parallel_size")
    tp = c.get("tensor_model_parallel_size")
    cp = c.get("context_parallel_size")
    ep = c.get("expert_model_parallel_size")
    etp = c.get("expert_tensor_parallel_size")
    actor_n = c.get("actor_num_nodes")
    actor_g = c.get("actor_num_gpus_per_node")
    t.add_row("actor_gpus", f"{actor_n} nodes × {actor_g} = {actor_n*actor_g}" if actor_n else "—")
    t.add_row("parallelism", f"TP={tp} PP={pp} CP={cp} EP={ep} ETP={etp}")
    t.add_row("global_batch_size", fmt(c.get("global_batch_size")))
    t.add_row("max_tokens_per_gpu", fmt(c.get("max_tokens_per_gpu")))
    t.add_row("effective_context_cap", f"{c.get('max_effective_context_derived')} tok (CP × max_tok/gpu)")
    t.add_row("recompute", f"{c.get('recompute_granularity')} / {c.get('recompute_method')} / {c.get('recompute_num_layers')}")
    t.add_row("advantage_estimator", fmt(c.get("advantage_estimator")))
    t.add_row("eps_clip", fmt(c.get("eps_clip"), 6))
    t.add_row("kl_loss_coef", fmt(c.get("kl_loss_coef")))
    t.add_row("optimizer_cpu_offload", fmt(c.get("optimizer_cpu_offload")))
    yield t

    t = Table(title="Generator config")
    t.add_column("key"); t.add_column("value")
    rng = c.get("rollout_num_gpus")
    rngpe = c.get("rollout_num_gpus_per_engine")
    t.add_row("rollout_gpus", fmt(rng))
    t.add_row("gpus_per_engine (= TP)", fmt(rngpe))
    t.add_row("num_engines", fmt(c.get("num_engines_derived")))
    t.add_row("samples_per_step (max concurrency)", f"{c.get('samples_per_step_derived')}  ({c.get('rollout_batch_size')} × {c.get('n_samples_per_prompt')})")
    t.add_row("rollout_max_response_len (tok)", fmt(c.get("rollout_max_response_len")))
    t.add_row("dp_attention / dp_size", f"{c.get('sglang_enable_dp_attention')} / {c.get('sglang_dp_size')}")
    t.add_row("ep_size", fmt(c.get("sglang_ep_size")))
    yield t


def trainer_per_step_table(steps: list[dict]):
    t = Table(title="Trainer per-step  (TRAIN time excludes wait)")
    for col in ["rollout", "step_s", "train_s", "wait_s", "wait_%", "updw_s",
                "log_probs_s", "lp_tflops", "at_tflops", "raw_reward", "trunc"]:
        t.add_column(col, justify="right")
    for s in steps:
        p = s["perf"]; r = s["rollout_consumed"]
        step_time = (p.get("log_probs_time", 0) or 0) + (p.get("actor_train_time", 0) or 0) + (p.get("update_weights_time", 0) or 0)
        wait_pct = 100 * (p.get("train_wait_time", 0) or 0) / max(step_time + (p.get("train_wait_time", 0) or 0), 1)
        t.add_row(
            str(s["step"]),
            fmt(step_time + p.get("train_wait_time", 0)),
            fmt(p.get("actor_train_time")),
            fmt(p.get("train_wait_time")),
            f"{wait_pct:.0f}%",
            fmt(p.get("update_weights_time")),
            fmt(p.get("log_probs_time")),
            fmt(p.get("log_probs_tflops"), 1),
            fmt(p.get("actor_train_tflops"), 1),
            fmt(r.get("raw_reward"), 4),
            fmt(r.get("truncated"), 3),
        )
    yield t


def per_rollout_table(rs: list[dict]):
    t = Table(title="Per-rollout generator + sandbox metrics")
    cols = ["roll", "rollout_s", "resp_μ", "resp_max", "trunc",
            "tool_calls/μ", "sandbox_s/μ",
            "apply_patch_err", "run_tests_err", "submitted", "tests_pass"]
    for c in cols: t.add_column(c, justify="right")
    for r in rs:
        L = r["lengths"]; P = r["perf"]; S = r["sandbox"]; TC = r["tool_calls"]; SW = r["swebench"]
        t.add_row(
            str(r["rollout_id"]),
            fmt(P.get("rollout_time"), 0),
            fmt(L.get("response_len_mean"), 0),
            fmt(L.get("response_len_max"), 0),
            fmt(L.get("truncated_ratio"), 3),
            fmt(S.get("tool_calls_per_sample_mean"), 1),
            fmt(S.get("tool_time_per_sample_mean"), 1),
            fmt(TC.get("apply_patch", {}).get("error_rate"), 3),
            fmt(TC.get("run_tests", {}).get("error_rate"), 3),
            fmt(SW.get("submitted_rate"), 3),
            fmt(SW.get("tests_pass_rate"), 3),
        )
    yield t


def per_tool_aggregate(rs: list[dict]):
    """Per-tool aggregate (mean across rollouts)."""
    t = Table(title="Per-tool aggregate  (mean across 20 rollouts)")
    for c in ["tool", "count/sample", "time/call_s", "error_rate"]:
        t.add_column(c, justify="right")
    tools = ["run_command", "read_file", "write_file", "apply_patch", "run_tests"]
    for tool in tools:
        cps = [r["tool_calls"][tool]["count_per_sample_mean"] for r in rs if r["tool_calls"][tool]["count_per_sample_mean"] is not None]
        tpc = [r["tool_calls"][tool]["time_per_call_mean_s"] for r in rs if r["tool_calls"][tool]["time_per_call_mean_s"] is not None]
        err = [r["tool_calls"][tool]["error_rate"] for r in rs if r["tool_calls"][tool]["error_rate"] is not None]
        m = lambda xs: sum(xs)/len(xs) if xs else None
        t.add_row(tool, fmt(m(cps), 2), fmt(m(tpc), 2), fmt(m(err), 3))
    yield t


def steady_state(steps: list[dict], rs: list[dict]):
    """Mean of last N=3 trainer steps and rollouts."""
    t = Table(title="Steady-state mean (last 3 rollouts / 3 trainer steps)")
    t.add_column("metric"); t.add_column("value", justify="right")
    last3_p = steps[-3:] if len(steps) >= 3 else steps
    last3_r = rs[-3:] if len(rs) >= 3 else rs
    def mp(key):
        vs = [s["perf"].get(key) for s in last3_p if s["perf"].get(key) is not None]
        return sum(vs)/len(vs) if vs else None
    def mr(*keys):
        vs = []
        for r in last3_r:
            d = r
            for k in keys:
                d = d.get(k, {}) if isinstance(d, dict) else None
                if d is None: break
            if isinstance(d, (int, float)): vs.append(d)
        return sum(vs)/len(vs) if vs else None

    rows = [
        ("log_probs_time_s",         mp("log_probs_time")),
        ("actor_train_time_s",       mp("actor_train_time")),
        ("update_weights_time_s",    mp("update_weights_time")),
        ("train_wait_time_s",        mp("train_wait_time")),
        ("log_probs_tflops",         mp("log_probs_tflops")),
        ("actor_train_tflops",       mp("actor_train_tflops")),
        ("rollout_time_s",           mr("perf", "rollout_time")),
        ("response_len_mean",        mr("lengths", "response_len_mean")),
        ("response_len_max",         mr("lengths", "response_len_max")),
        ("truncated_ratio",          mr("lengths", "truncated_ratio")),
        ("tool_calls_per_sample",    mr("sandbox", "tool_calls_per_sample_mean")),
        ("sandbox_time_per_sample_s", mr("sandbox", "tool_time_per_sample_mean")),
        ("apply_patch_error_rate",   mr("tool_calls", "apply_patch", "error_rate")),
        ("run_tests_error_rate",     mr("tool_calls", "run_tests", "error_rate")),
        ("swebench_submitted_rate",  mr("swebench", "submitted_rate")),
        ("swebench_tests_pass_rate", mr("swebench", "tests_pass_rate")),
        ("policy_staleness_mean",    mr("policy_staleness", "mean")),
        ("policy_staleness_max",     mr("policy_staleness", "max")),
    ]
    for k, v in rows:
        if isinstance(v, float):
            t.add_row(k, f"{v:.4f}")
        else:
            t.add_row(k, fmt(v))
    yield t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, default=Path("metrics.json"))
    p.add_argument("--out", type=Path, default=Path("metrics_report.txt"))
    args = p.parse_args()
    data = json.loads(args.inp.read_text())

    console = Console(record=True, width=140)
    console.print(f"Log-derived: {data['n_steps']} steps, {data['n_rollouts']} rollouts, "
                  f"{len(data['events']['update_weights_pushes'])} update_weights pushes")
    for tg in cfg_tables(data["config"]): console.print(tg)
    for tg in trainer_per_step_table(data["steps"]): console.print(tg)
    for tg in per_rollout_table(data["rollouts"]): console.print(tg)
    for tg in per_tool_aggregate(data["rollouts"]): console.print(tg)
    for tg in steady_state(data["steps"], data["rollouts"]): console.print(tg)

    args.out.write_text(console.export_text())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
