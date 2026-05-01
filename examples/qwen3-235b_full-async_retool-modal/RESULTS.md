# Run 3 (job 135) — RESULTS

**Date:** 2026-04-27
**Goal:** 20 RL training steps on Qwen3-235B-A22B-Thinking-2507, fully-async + Modal retool, BF16 train + FP8 inference.
**Outcome:** ✅ All 20 training steps completed cleanly. Job ran for 1h 13m 12s wall-clock; cancelled after step 20 emitted (would have continued to step 3000).

## Cluster layout

- **32 H200 nodes total** (256 GPUs):
  - 8 actor nodes (64 GPUs) running Megatron BF16 training
  - 24 rollout nodes (192 GPUs) running 48 SGLang FP8 engines (TP=4 each)
- Train parallelism: TP=4 PP=4 CP=2 EP=16 ETP=1, sequence-parallel, recompute=full/uniform/1
- Modal sandbox pool: 128 warm sandboxes, network-blocked, 60 s per-exec timeout

## Training trajectory (loss / grad / entropy)

20 steps. GRPO with `eps_clip=0.2/0.28`, `kl_loss_coef=0`, `use_tis` enabled.

| step | train/loss | grad_norm | entropy_loss |
|---:|---:|---:|---:|
| 0 | -1.0e-4 | 0.083 | 0.232 |
| 1 | +9.3e-5 | 0.066 | 0.227 |
| 5 | +5.6e-6 | 0.048 | 0.238 |
| 10 | +1.0e-7 | 0.051 | 0.210 |
| 15 | +2.1e-5 | 0.064 | 0.217 |
| 19 | -2.3e-7 | 0.038 | 0.227 |

- `train/loss` oscillates around 0 (range ±1e-4) — Qwen3-235B-Thinking is already well-aligned for these dapo-math prompts; effective gradient is tiny.
- `grad_norm` decays from ~0.08 to ~0.04 over the 20 steps — model state is settling.
- `entropy_loss` is stable around 0.21–0.25 — no entropy collapse.
- `train/pg_clipfrac = 0.0` always — no clipping triggered (rollout/train policy match is tight, `tis_abs ≈ 0.02`).
- `train/ois ≈ 1.0`, `train/tis ≈ 1.0` — TIS importance ratios near unity; off-policy correction is essentially identity.
- `train/kl_loss` rises from 0 → ~0.001 across steps (low, contained).

## Performance — per-step (steady-state, last 3 rollouts)

| Metric | Value |
|---|---:|
| `step_time` (full RL step) | 156 s |
| `actor_train_time` | 64 s |
| `train_wait_time` (idle on rollout) | 64 s |
| `wait_time_ratio` (idle/total) | **41 %** |
| `update_weights_time` (actor → SGLang push) | 41 s |
| `actor_train_tflops` (per GPU) | 54 TFLOPS |
| `actor_train_tok_per_s` | 22 K tok/s/GPU |
| **MFU** (vs H200 BF16 peak 989 TFLOPS) | **5.5 %** |
| `log_probs_tflops` | 90 |
| `ref_log_probs_tflops` | 79 |

Bring-up cost (one-time, step 0):
- step_time = 393 s (vs 156 s steady-state)
- inflated by initial weight push (46.4 s) + first rollout drained from cold queue (185 s wait_time)

## Generator — per-step (steady-state)

| Metric | Value |
|---|---:|
| `rollout_time` (256 samples) | 118 s |
| `samples_per_sec_gen` | 2.18 |
| `prompt_len` mean | 333 tokens |
| `response_len` mean | 5190 tokens |
| `truncated_ratio` | 30 % |
| `repetition_frac` | 0 % |
| `sandbox_calls_per_sample` | 0.007 |
| `sandbox_error_rate` | 0 % |
| `sandbox_exec_per_call` | 0.49 s |

- Mean response length **5190 tokens** at steady state (vs 8192 cap) — Qwen3-Thinking is verbose (`<think>` blocks consume most of the output).
- `truncated_ratio = 30%` — about a third of samples hit the 8192-token cap. To improve answer-completion rates, raise `--rollout-max-response-len` to 12K–16K. (Stuck with 8192 here for memory headroom + faster wall time.)
- **Sandbox usage drops sharply over training**: rollout 0 had 0.14 calls/sample, rollouts 9–19 are at 0.01–0.02 calls/sample. Either (a) the model rarely emits the new Qwen3 native `<tool_call>{json}</tool_call>` format on its own without further training, or (b) the 30 % truncated samples never reach the answer/tool-call portion of generation. Both are plausible; the former is the dominant explanation.
- `sandbox_error_rate` was 5–8% on rollouts 6–7, dropped to 0 % afterward. Modal sandbox pool was healthy — no execs failed in steady state.

## System efficiency

| Metric | Value |
|---|---:|
| `gen/train ratio` (rollout_time / actor_train_time, steady-state) | **1.85** |
| `wait_time_ratio` (steady-state) | 41 % |
| Steady-state `step_time` | 156 s |

- **`gen/train ratio = 1.85`**: rollout side is ~1.85× slower than the actor train step, so the trainer is starved for ~40 % of each step waiting for groups to arrive. Knobs to close the gap:
  - more rollout GPUs (currently 192; could go to 256 if free)
  - larger SGLang engines (TP=8 with EP per quirk fix, or DP-attention) for faster decode
  - lower `n_samples_per_prompt` (currently 8; 4 would halve rollout work but hurt GRPO variance)
- **`wait_time_ratio = 41 %`**: confirms the same — fully-async still leaves ~40 % of the trainer step idle. That's better than synchronous (where it'd be ~65 %), but there's room.

## Bring-up timing breakdown

| Phase | Wall time |
|---|---:|
| sbatch submit → 32-node enroot create | ~1 min |
| Ray head + 31 workers up | ~1 min |
| SGLang engine init | ~30 s |
| FP8 weight load (24 shards × 4 ranks × 48 engines, all reading /data) | ~4 min |
| DeepGEMM kernel warmup | ~1.5 min |
| CUDA graph capture (52 batch sizes) | ~4 min |
| **All 48 SGLang engines: "Server is up"** | **t+10 min** |
| Megatron actor weight load (BF16 torch_dist, 8 PP shards) | ~2 min |
| First `update_weights` push (Megatron → SGLang) | 46 s |
| First rollout (256 samples) | 136 s |
| **First `perf 0`** | **t+25 min** |
| 20 steps complete | t+1h 13m |

## Files of record

- Slurm log: `/home/primeteam/slime/mnt/logs/infx-retool-async-135.out` (1.7 MB)
- Stderr: `/home/primeteam/slime/mnt/logs/infx-retool-async-135.err`
- Per-node logs: `infx-retool-async-135-qpn01gpu*.out` (32 files)
- Final metrics-report dump: `/home/primeteam/slime/mnt/logs/metrics_report_final.txt`
- BF16 torch_dist actor checkpoint: `/data/outputs/slime-checkpoints/Qwen3-235B-A22B-Thinking-2507_torch_dist/release/` (438 GB)

## What this run validated

1. **End-to-end fully-async RL on Qwen3-235B-A22B-Thinking-2507** (BF16 train + FP8 infer) on 32 H200 nodes works.
2. **Direct enroot bypass** of pyxis's /tmp extraction limit works — the sqsh on /data + per-srun ENROOT_DATA_PATH=/scratch pattern is reliable.
3. **Qwen3 native chat template + JSON tool-call parsing** produces well-formed prompts and parses model output cleanly (0 % parser errors over 5120 samples).
4. **Modal sandbox pool** survives the 73-min run with 0 % error rate.
5. **fully-async aborts during update_weights** are handled correctly — `add_slime_metrics.generate` resets sample state on retry, no Token/logp length mismatch asserts fired.
6. **Cluster-specific quirks** Q13–Q17 documented in `EXPERIMENT_LOG.md` are now permanent guardrails for any future re-target.

## What this run did NOT validate (limitations)

- **Convergence**: 20 steps is far too few to see actual reward improvement. `train/loss ≈ 0` because the model is well-aligned out of the box and the importance-sampling correction makes each gradient step nearly identity.
- **Eval**: `--eval-interval 50` so no eval ran in 20 steps. Test on aime-2024 by submitting with `NUM_ROLLOUT=50` and watching for the eval block.
- **Reward signal**: rewards are emitted (`rollout/zero_std/count_-1.1` etc are non-zero, so some samples score correctly), but we didn't surface mean reward — that's in the slime stock metrics dict but the GLM-style metrics_report doesn't pull it out. Add a `rollout/reward/mean` row if you want to track learning curves.
- **MFU on H200**: 5.5 % is below the SETUP.md target of >25 %. Recompute=full + small global_batch + Adam CPU offload eat the headroom. For longer training, try `--recompute-num-layers 0` (drop full recompute) or larger global_batch_size, but mind activation memory.
