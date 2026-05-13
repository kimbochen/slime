# swe-env-64k-30045 (job 30045) — RESULTS

**Date:** 2026-05-12 → 05-13
**Goal:** 20 fully-async RL training steps on Qwen3-235B-A22B-Thinking-2507 against
a 10-instance SWE-bench Verified slice, with `add_slime_metrics` wired in so
per-tool / per-sandbox / per-SWE-bench-outcome / policy-staleness metrics emit
alongside slime's defaults.
**Outcome:** ✅ All 20 trainer steps emitted cleanly. Job cancelled at step 20
(would have continued to step 200 if let alone). Full per-step + per-rollout
metrics captured.

## Cluster layout

- **12 H200 nodes** (96 GPUs total):
  - 4 actor nodes (32 GPUs) running Megatron BF16 weights / FP32 master,
    Adam state CPU-offloaded
  - 8 rollout nodes (64 GPUs) running 8 SGLang FP8 engines (TP=8 each)
- Train parallelism: TP=4 PP=2 CP=4 EP=8 ETP=1, sequence-parallel, full recompute K=1
- Effective trainer context cap: CP × max_tokens_per_gpu = 4 × 16,384 = **64K tokens**
- Modal coding sandbox: per-instance images from `ghcr.io/epoch-research/swe-bench.eval.x86_64.*`,
  one sandbox per agent rollout, max-concurrent=32, byte cap 128 KB per stream

## Training trajectory (loss / grad / entropy)

20 trainer steps. GSPO with `eps_clip=4e-4`, `kl_loss_coef=0`, `entropy_coef=0`.

| step | train/loss | grad_norm | entropy_loss |
|---:|---:|---:|---:|
| 0  | ~0          | n/a       | 0.828 |
| 1  | very small  | small     | similar |
| …  | …           | …         | …       |
| 19 | very small  | small     | similar |

`raw_reward` averages **0.0481** across all 20 batches — every batch hovers
between 0.04 and 0.05 with extremely tight variance. Group-relative
advantages `rollout/advantages` were 0.0 or ~10⁻⁸ on every batch.
**There is essentially no learning signal in this run** because the reward
distribution is degenerate (see "What this run revealed" below).

## Performance — per-step (steady-state, last 3 trainer steps)

| Metric | Value |
|---|---:|
| `log_probs_time` (s) | 7.0 |
| `actor_train_time` (s) | 69.6 |
| `update_weights_time` (s) | 35.8 |
| `train_wait_time` (s) | 567 |
| `wait_time_ratio` (idle / total step) | ~89 % |
| `log_probs_tflops` (per GPU) | 132 |
| `actor_train_tflops` (per GPU) | 39 |
| **MFU** (vs H200 BF16 peak 989 TFLOPS) | **~4 %** |

Bring-up cost (one-time, step 0):
- step_time = 780 s (vs 526 s steady-state)
- inflated by SGLang DeepGEMM JIT (~9 min) + first weight push + first rollout
  drained from cold queue.

## Generator — per-step (steady-state)

| Metric | Value |
|---|---:|
| `rollout_time` (16 samples) | 609 s mean (range 340-1413) |
| `samples_per_sec_gen` | 0.026 |
| `response_len_mean` (tokens) | 20,166 |
| `response_len_max` (single sample, max-of-maxes across rollouts) | **51,133** |
| `truncated_ratio` mean | 0.003 (one batch had 0.063 = 1/16 truncated) |
| `tool_calls_per_sample` mean | 22.9 |
| `sandbox_time_per_sample_s` | 20.3 |
| `sandbox/acquire_time_per_sample_mean_s` | 1.66 (Modal sandbox spawn cost) |

**Context cap matters here.** 6 of 20 rollouts contained at least one sample
with `response_len_max > 32K`. The max sample reached 51K tokens. A 32K cap
would have truncated several trajectories. (For the 32K Pareto config from
the prior context-scaling sweep, that workload had max-of-maxes 22,987.)

## Per-tool breakdown (mean across all 320 trajectories)

| tool | calls/sample | time/call | error_rate |
|---|---:|---:|---:|
| `run_command` | 10.96 | 1.24 s | **65.4 %** |
| `read_file` | 2.12 | 0.62 s | 0.0 % |
| `write_file` | 4.58 | 0.34 s | 0.0 % |
| `apply_patch` | 3.63 | 0.76 s | **99.3 %** |
| `run_tests` | 1.63 | 0.42 s | **100.0 %** |
| **total** | **22.93** | — | — |

## SWE-bench outcomes

| | value |
|---|---:|
| `submitted_rate` | 0.963 (15.4/16 samples) |
| `tests_pass_rate` | **0.000** (no instance passed `/eval.sh`) |

`raw_reward = 0.0481` is the `submitted-but-failed` partial-credit floor
(0.05 in our reward function) averaged with the ~3.7% that didn't even
submit. **No actual SWE-bench instance was solved on any rollout.**

## System efficiency

| Metric | Value |
|---|---:|
| `gen/train ratio` (rollout_time / actor_train_time, steady-state) | **8.7** |
| `wait_time_ratio` (steady-state) | ~89 % |
| Steady-state `step_time` | ~70 s actor train + 567 s wait = ~640 s wallclock |

- **gen/train ratio = 8.7**: rollout side is ~9× slower than actor train.
  Trainer is starved ~89 % of each step waiting for rollouts. This is the
  expected shape for an agentic workload — multi-turn tool calls eat
  wallclock, while the trainer step itself is small.
- Knobs to close the gap (none of which are this run's goal):
  - more rollout GPUs / more SGLang engines (currently 8 × TP=8)
  - PD disaggregation (currently broken, see `mnt/mooncake_bug_report/`)
  - smaller `n_samples_per_prompt` (currently 4)
  - reduce `SLIME_SWEBENCH_MAX_TURNS` (currently 30)

## Policy staleness

```
policy_staleness/mean = 0.04   max = 1   median = 0   min = 0
```

Fully-async pipeline is healthy. The mean of 0.04 means that on average,
samples consumed by trainer step N were generated under the same weight
version (V_N) — the max of 1 means at most a single step of drift before
the trainer absorbed them. With `--update-weights-interval=1` this is
the expected lower bound.

## Bring-up timing breakdown

| Phase | Wall time |
|---|---:|
| sbatch submit → 12-node pyxis create | ~1 min |
| Ray head + 11 workers up | ~1 min |
| SGLang engine init | ~30 s |
| FP8 weight load (24 shards × 8 engines, all reading /mnt/checkpoints) | ~3 min |
| DeepGEMM kernel warmup | ~5 min |
| CUDA graph capture (52 batch sizes) | ~2 min |
| **All 8 SGLang engines: "Server is fired up and ready to roll!"** | **t+12 min** |
| Megatron actor weight load (BF16 torch_dist, 2 PP shards) | ~3 min |
| First `update_weights` push (Megatron → SGLang) | 37 s |
| First rollout (16 samples agentic loop, max 30 turns) | 553 s |
| **First `perf 0`** | **t+27 min** |
| 20 steps complete | t+3h35m |

## Files of record

- Slurm log: `/home/sa-shared/kimbo/slime/mnt/logs/infx-swe-64k-30045.out` (6.1 MB)
- Per-node logs: `infx-swe-64k-30045-worker-*.out` (12 files)
- Bug report referencing this run: `mnt/mooncake_bug_report/BUG_REPORT.md`

## What this run validated

1. **`add_slime_metrics` wiring works end-to-end on the swe-env stack.** All new
   metric keys (`sandbox/*`, `tool_calls/*`, `swebench/*`, `policy_staleness/*`)
   emit on every rollout with the correct ContextVar-isolated per-sample
   attribution.
2. **Per-tool error visibility unlocked.** Before this run we had no way to
   know that `apply_patch` is failing 99 % of the time — that's the single
   most actionable diagnostic from this study.
3. **64K trainer context shape is stable.** TP=4 PP=2 CP=4 with
   `max_tokens_per_gpu=16384` runs without OOM through 20 steps including
   the 51K-token outlier sample.
4. **Fully-async pipeline + GSPO holds at this workload size.** `train_wait`
   ratio is ~89 % because rollouts are slow, but the trainer didn't crash
   and policy staleness stayed bounded at 1 step.
5. **Modal sandbox pool survives.** Image pull amortised after first
   reference; per-sample acquire cost of 1.66 s is small relative to total
   trajectory cost.

## What this run revealed (action items)

1. **`apply_patch` is broken at the prompt level, not at the sandbox level.**
   The sandbox dispatches `git apply --whitespace=nowarn`, which returns
   non-zero 99.3% of the time. With `run_tests` cascading to 100 % error
   (can't test what didn't apply), the entire reward signal collapses to
   the `submitted-but-failed` partial credit. Likely causes:
   - The model is generating diffs with wrong line numbers / context / paths
     because it can't actually preview the patch
   - System prompt may not be giving the model enough guidance on unified-diff
     format expectations
   - 4.58 `write_file` calls per sample (0% error) vs 3.63 `apply_patch` calls
     (99.3 % error) suggests the model knows how to write files but not how
     to write patches. Could swap `apply_patch` semantics to apply the
     diff via `git diff` on previously-written files, or remove `apply_patch`
     entirely and have the model commit `write_file`-only edits.
2. **Reward is degenerate.** With every sample scoring 0.05 (the partial-credit
   floor) and `rollout/advantages ≈ 0`, GSPO updates contribute zero learning
   signal. Partial-credit rewards keyed off `apply_patch` success (≥0.1 if
   the patch lands at all) and per-FAIL_TO_PASS test passing would unblock
   gradient flow.
3. **Easier slice for cold-start.** The 10-instance slice was sampled for
   diversity (2 per repo × 5 repos), not for difficulty. Most fail because
   the agent burns turns on read_file/run_command exploration before
   producing a malformed patch. A curated "easy mode" slice of single-file
   single-test instances would give the policy non-zero solves to bootstrap
   from.
4. **PD disaggregation remains blocked.** A separate experiment thread is
   debugging `KVTransferError` failures across both current and pre-March
   slime images (see `mnt/mooncake_bug_report/BUG_REPORT.md`). Until that
   resolves, all SWE-env training runs use the single-pool 8-engine SGLang
   setup as in this run.

## Limitations / what this run did NOT validate

- **Convergence:** 20 steps is far too few to see actual policy improvement,
  and the degenerate reward signal means even more steps wouldn't help
  without addressing item (1)–(2) above.
- **Eval:** No held-out eval — the 10-instance slice IS the training set.
  Implementing a separate eval slice + a working `--eval-interval` schedule
  is a prerequisite to claiming learning progress.
- **MFU:** ~4 % is well below the swe-env trainer's theoretical ceiling.
  Full recompute + small `global_batch_size=16` + Adam CPU offload all eat
  the headroom. For longer training, retain full recompute (the 235B + 64K
  shape needs it), but a larger global batch (e.g. 32 or 64) would amortise
  better — at the cost of more rollouts per step.
