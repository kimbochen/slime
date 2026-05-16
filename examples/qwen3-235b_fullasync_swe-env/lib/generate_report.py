#!/usr/bin/env python3
"""Generate a markdown report from a slime SWE-env training log.

Usage:
    python generate_report.py <slurm.out> <output_dir>

Writes:
    <output_dir>/RESULTS.md      — per-step + per-rollout markdown
    <output_dir>/metrics.json    — structured dump
"""
from __future__ import annotations
import json
import re
import statistics
import sys
from pathlib import Path

H200_PEAK_BF16_TFLOPS = 990.0


def parse_perf_lines(log_path: Path) -> list[dict]:
    pattern = re.compile(r"train_metric_utils\.py:44 - perf (\d+): (\{.*\})")
    steps = []
    for line in log_path.read_text(errors="replace").splitlines():
        m = pattern.search(line)
        if not m:
            continue
        step_n = int(m.group(1))
        body = m.group(2).replace("'", '"')
        try:
            d = json.loads(body)
        except Exception:
            continue
        steps.append({"step": step_n, "perf": d})
    steps.sort(key=lambda x: x["step"])
    return steps


def parse_rollout_lines(log_path: Path) -> list[dict]:
    pattern = re.compile(r"rollout_metrics\] rollout (\d+): (\{.*\})")
    rollouts = []
    for line in log_path.read_text(errors="replace").splitlines():
        m = pattern.search(line)
        if not m:
            continue
        r_n = int(m.group(1))
        body = m.group(2).replace("'", '"')
        try:
            d = json.loads(body)
        except Exception:
            continue
        rollouts.append({"rollout_id": r_n, "metrics": d})
    rollouts.sort(key=lambda x: x["rollout_id"])
    return rollouts


def parse_trainer_rewards(log_path: Path) -> list[dict]:
    pattern = re.compile(r"data\.py:208 - rollout (\d+): (\{.*\})")
    rewards = []
    for line in log_path.read_text(errors="replace").splitlines():
        m = pattern.search(line)
        if not m:
            continue
        r_n = int(m.group(1))
        body = m.group(2).replace("'", '"')
        try:
            d = json.loads(body)
        except Exception:
            continue
        rewards.append({"rollout_id": r_n, "trainer_data": d})
    rewards.sort(key=lambda x: x["rollout_id"])
    return rewards


def render_report(
    steps: list[dict],
    rollouts: list[dict],
    rewards: list[dict],
    job_id: str,
    log_path: Path,
    out_path: Path,
    title: str = "SWE-env run",
) -> str:
    lines: list[str] = []
    lines.append(f"# {title} — job {job_id}")
    lines.append("")
    lines.append(f"- Log: `{log_path}`")
    lines.append(f"- Steps captured: {len(steps)}")
    lines.append(f"- Rollouts captured: {len(rollouts)}")
    lines.append("")

    if steps:
        lines.append("## Per-step performance")
        lines.append("")
        lines.append("| step | wait (s) | train (s) | step (s) | train_tflops | MFU(in-step) | wallclock_MFU | wait_ratio |")
        lines.append("|-----:|---------:|----------:|---------:|-------------:|-------------:|--------------:|-----------:|")
        for s in steps:
            p = s["perf"]
            wait = p.get("perf/train_wait_time", 0)
            train_t = p.get("perf/train_time", 0)
            step_t = p.get("perf/step_time", 0)
            tflops = p.get("perf/actor_train_tflops", 0)
            wr = p.get("perf/wait_time_ratio", 0)
            actor_t = p.get("perf/actor_train_time", 0)
            mfu_in = tflops / H200_PEAK_BF16_TFLOPS * 100 if tflops else 0
            wc_mfu = (mfu_in * (actor_t / step_t)) if step_t else 0
            lines.append(
                f"| {s['step']:>4d} "
                f"| {wait:>8.0f} "
                f"| {train_t:>9.0f} "
                f"| {step_t:>8.0f} "
                f"| {tflops:>12.1f} "
                f"| {mfu_in:>11.2f}% "
                f"| {wc_mfu:>12.2f}% "
                f"| {wr:>10.2f} |"
            )
        lines.append("")

        # aggregate stats over steady-state (skip step 0)
        steady = steps[1:] if len(steps) > 1 else steps
        if steady:
            tflops = [s["perf"].get("perf/actor_train_tflops", 0) for s in steady]
            waits = [s["perf"].get("perf/train_wait_time", 0) for s in steady]
            trains = [s["perf"].get("perf/train_time", 0) for s in steady]
            wrs = [s["perf"].get("perf/wait_time_ratio", 0) for s in steady]
            lines.append("### Steady-state aggregate (step 1+)")
            lines.append("")
            lines.append(f"- mean train_tflops: **{statistics.mean(tflops):.1f}** (MFU {statistics.mean(tflops)/H200_PEAK_BF16_TFLOPS*100:.2f}%)")
            lines.append(f"- mean wait_time: {statistics.mean(waits):.0f}s")
            lines.append(f"- mean train_time: {statistics.mean(trains):.0f}s")
            lines.append(f"- mean wait_ratio: {statistics.mean(wrs):.2f}")
            lines.append("")

    if rollouts:
        lines.append("## Per-rollout metrics")
        lines.append("")
        lines.append("| rollout | rt (s) | tok/gpu/s | sandbox_acq (s) | resp_len_mean | resp_len_max | zero_std | tool_calls | policy_staleness_max |")
        lines.append("|--------:|-------:|----------:|----------------:|--------------:|-------------:|---------:|-----------:|---------------------:|")
        for r in rollouts:
            m = r["metrics"]
            rt = m.get("perf/rollout_time", 0)
            tps = m.get("perf/tokens_per_gpu_per_sec", 0)
            sbx = m.get("sandbox/acquire_time_per_sample_mean", 0)
            rlm = m.get("rollout/response_len/mean", 0)
            rlx = m.get("rollout/response_len/max", 0)
            zs = m.get("rollout/zero_std/count_0.1", 0)
            tc = m.get("sandbox/tool_calls_per_sample_mean", 0)
            psm = m.get("policy_staleness/max", 0)
            lines.append(
                f"| {r['rollout_id']:>7d} "
                f"| {rt:>6.0f} "
                f"| {tps:>9.1f} "
                f"| {sbx:>15.1f} "
                f"| {rlm:>13.0f} "
                f"| {rlx:>12.0f} "
                f"| {zs:>5}/8 "
                f"| {tc:>10.1f} "
                f"| {psm:>20} |"
            )
        lines.append("")

    if rewards:
        lines.append("## Trainer-side reward signal")
        lines.append("")
        lines.append("| rollout | raw_reward | rewards (post-norm) | log_probs | rollout_log_probs | kl |")
        lines.append("|--------:|-----------:|--------------------:|----------:|------------------:|---:|")
        for r in rewards:
            d = r["trainer_data"]
            rr = d.get("rollout/raw_reward", 0)
            rw = d.get("rollout/rewards", 0)
            lp = d.get("rollout/log_probs", 0)
            rlp = d.get("rollout/rollout_log_probs", 0)
            kl = d.get("rollout/kl", 0)
            lines.append(
                f"| {r['rollout_id']:>7d} "
                f"| {rr*100:>9.2f}% "
                f"| {rw:>+19.4f} "
                f"| {lp:>+9.4f} "
                f"| {rlp:>+17.4f} "
                f"| {kl:>+4.4f} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines))
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, help="slurm stdout log (e.g. mnt/logs/infx-swe-31079.out)")
    ap.add_argument("out_dir", type=Path, help="experiment dir to write metrics.json + RESULTS.md into")
    ap.add_argument(
        "--title",
        default="SWE-env run",
        help='Markdown title prefix (e.g. "Resumable SWE-env run", "Base SWE-env (no partial-rollout)")',
    )
    args = ap.parse_args()

    log_path = args.log
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    m = re.search(r"-(\d+)\.out$", log_path.name)
    job_id = m.group(1) if m else "unknown"

    steps = parse_perf_lines(log_path)
    rollouts = parse_rollout_lines(log_path)
    rewards = parse_trainer_rewards(log_path)

    metrics_json = {
        "job_id": job_id,
        "log_path": str(log_path),
        "n_steps": len(steps),
        "n_rollouts": len(rollouts),
        "steps": steps,
        "rollouts": rollouts,
        "trainer_rewards": rewards,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2))

    md = render_report(steps, rollouts, rewards, job_id, log_path, out_dir / "RESULTS.md", title=args.title)
    print(md)


if __name__ == "__main__":
    main()
