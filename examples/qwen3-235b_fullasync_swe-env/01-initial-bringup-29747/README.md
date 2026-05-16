# 01 — Initial SWE-bench Verified bring-up (job 29747)

**Status:** ✅ working baseline
**Lib SHA:** `8368a19f` (commit that introduced the example)
**Dataset:** `swebench_verified/sample_10.jsonl` (10 instances)

## Hypothesis

End-to-end validation: can we run Qwen3-235B-A22B-Thinking against SWE-bench
Verified via fully-async Megatron + 8 SGLang TP=8 engines on 12 H200 nodes,
with a custom Modal-backed coding sandbox and a 6-tool agent loop?

## Configuration deltas vs canonical

| Knob | Value at this experiment | Notes |
|---|---|---|
| Trainer parallelism | TP=4 PP=4 CP=2 EP=8 ETP=1 | not yet reshaped for 64K context |
| Effective context cap | 32K (CP × max_tokens_per_gpu = 2 × 16384) | |
| `--rollout-max-response-len` | 32768 | |
| Agent `max_turns` | 30 | |
| Dataset | Verified sample_10 | |
| `--partial-rollout` | OFF | |
| `--update-weights-interval` | (not set; default 1) | every-step weight push |
| Metric instrumentation | none | bare slime defaults |

## What was tested

12-node Slurm bring-up. 4 actor + 8 rollout. Verified that:

1. Modal sandboxes spawn from `ghcr.io/epoch-research/swe-bench.eval.x86_64.*`
   per-instance images at the expected throughput.
2. The agent's 6 tools (`run_command`, `read_file`, `write_file`,
   `apply_patch`, `run_tests`, `submit`) round-trip through the sandbox
   correctly.
3. `reward_func` runs the instance's `/eval.sh` inside the same sandbox
   the agent used and returns 1.0 on green / 0.0 on red.
4. Fully-async (`train_async.py`) drives the rollout/training overlap.

## Result

Bring-up was green. ~5% raw_reward at baseline. Trajectories truncated at
32K context for the longest samples — motivation for **02-context-scaling**.

## How to reproduce

```bash
sbatch 01-initial-bringup-29747/run_swe.sbatch
```

Uses `lib/` for the agent runtime. The launcher in this dir adds `../../lib`
to PYTHONPATH so `--custom-generate-function-path rollout_metrics.generate`
resolves into the canonical library.
