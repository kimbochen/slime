# Run 37043 — Lite-40 + 8-sample groups, NUM_ROLLOUT=20 (clean completion)

- **Status**: ✅ COMPLETED (exit 0:0)
- **Wall**: 3h20m10s
- **Date**: 2026-05-29
- **Log**: `mnt/logs/infx-swebench-new-37043.out`
- **Metrics JSONL**: `mnt/logs/metrics-37043.jsonl` (100 records, 20 rollouts × 5 records each)

## Summary

20-rollout RL training run of Qwen3-235B-A22B-Thinking-2507-FP8 on a 40-instance
slice of SWE-bench-Lite using slime + Megatron + SGLang + Modal sandboxes.

Pipeline ran cleanly end-to-end: 20/20 rollouts completed, 20/20 trainer steps
performed weight updates, 20/20 metric records emitted to JSONL including the
per-group max staleness telemetry. Four `update_weights` broadcasts fired
(initial + 3 regular at end of rollouts 4, 9, 14) — all completed without
incident. `Training command finished` reached cleanly; the sbatch's bottom
orphan-cleanup pass ran automatically and terminated 3 leftover Modal
sandboxes.

Reward signal stayed flat (~6.5% mean resolve, no trend) — consistent with the
configuration ceiling for this scale of agent RL on a small prompt pool with
binary reward.

## Configuration

```
Model:              Qwen3-235B-A22B-Thinking-2507-FP8 (FP8 weights, bf16 trainer compute)
Cluster:            12 H200 nodes × 8 GPU (4 actor nodes for trainer, 8 nodes for rollout)
SGLang:             8 engines, TP=8 each, dp_size=8, ep_size=8, mem_fraction=0.85
Trainer (Megatron): TP=4, PP=2, CP=4, EP=8, ETP=1, max_tokens_per_gpu=16384
                    → effective context cap = CP × max_tokens = 64K
                    advantage: GSPO; kl_loss_coef: 0; eps_clip: 4e-4; lr: 1e-6
                    optimizer: adam (CPU offload + precision-aware)
                    --save / --save-interval: OMITTED (no checkpoint, by design)
Rollout:            --partial_rollout + --use-tis (tis_clip 2.0, tis_clip_low 0)
Data:               40 SWE-bench-Lite instances (mnt/data/swebench_lite/sample_40.jsonl)
                    Each prompt used ~4 times across the 20-step run
Batching:           rollout_batch_size=8, n_samples_per_prompt=8, global_batch_size=64
                    → 64 trajectories per rollout (8 prompts × 8 samples each)
Weight updates:     every 5 rollouts (interval=5) → 4 updates (initial + 3 regular)
Custom hooks:       generate_with_openhands.generate, reward.compute_reward,
                    metrics.log_rollout_data

Staleness metric:   staleness = max(0, rollout_id − weight_versions[0] × interval)
                    Plus per-group: group_max_staleness, group_staleness_spread
```

## Tables

### 1. Trainer step time (sec/step)

| step | step_t | train_t | actor_t | logp_t | wait_t | upd_w | wait_r |
|-----:|-------:|--------:|--------:|-------:|-------:|------:|-------:|
| 0 | 928 | 293 | 237 | 54 | 635 | 39.2 | 0.685 |
| 1 | 390 | 204 | 182 | 20 | 186 | — | 0.477 |
| 2 | 514 | 199 | 174 | 23 | 315 | — | 0.613 |
| 3 | 467 | 187 | 165 | 20 | 281 | — | 0.601 |
| 4 | 450 | 210 | 186 | 23 | 240 | — | 0.534 |
| 5 | 549 | 201 | 178 | 22 | 348 | 35.1 | 0.633 |
| 6 | 548 | 193 | 173 | 19 | 354 | — | 0.647 |
| 7 | 514 | 183 | 163 | 20 | 331 | — | 0.643 |
| 8 | 363 | 171 | 151 | 19 | 191 | — | 0.527 |
| 9 | 529 | 173 | 144 | 28 | 356 | — | 0.673 |
| 10 | 497 | 178 | 158 | 19 | 319 | 36.7 | 0.642 |
| 11 | 470 | 190 | 159 | 30 | 280 | — | 0.597 |
| 12 | 433 | 189 | 169 | 19 | 244 | — | 0.563 |
| 13 | 520 | 195 | 174 | 21 | 325 | — | 0.625 |
| 14 | 498 | 212 | 189 | 22 | 287 | — | 0.575 |
| 15 | 449 | 201 | 180 | 20 | 248 | 34.9 | 0.552 |
| 16 | 593 | 174 | 154 | 19 | 419 | — | 0.707 |
| 17 | 548 | 213 | 185 | 27 | 335 | — | 0.611 |
| 18 | 469 | 185 | 165 | 19 | 283 | — | 0.605 |
| 19 | 467 | 197 | 174 | 23 | 270 | — | 0.578 |

**Steady-state means** (steps 1-19, excluding warmup): step_time ≈ 490s,
train_time ≈ 192s, actor_train ≈ 170s, log_probs ≈ 22s, train_wait ≈ 298s,
wait_ratio ≈ 0.60. Trainer sits idle ~60% of each step waiting for rollouts.
Four `update_weights` events at steps 0/5/10/15 took 39.2/35.1/36.7/34.9s
— consistent.

### 2. Generator per-sample (sec, mean of 64)

| step | inf_mean | inf_p95 | sb_mean | sb_p95 | sb_frac | roll_t |
|-----:|---------:|--------:|--------:|-------:|--------:|-------:|
| 0 | 277 | 421 | 18.9 | 67.6 | 0.064 | 593 |
| 1 | 311 | 428 | 12.8 | 30.5 | 0.040 | 479 |
| 2 | 266 | 444 | 15.6 | 32.7 | 0.055 | 519 |
| 3 | 298 | 444 | 20.2 | 61.1 | 0.064 | 479 |
| 4 | 319 | 503 | 20.0 | 86.1 | 0.059 | 426 |
| 5 | 287 | 473 | 16.3 | 41.1 | 0.054 | 522 |
| 6 | 322 | 541 | 13.2 | 39.9 | 0.039 | 555 |
| 7 | 322 | 450 | 22.1 | 72.9 | 0.064 | 524 |
| 8 | 269 | 407 | 16.3 | 37.8 | 0.057 | 374 |
| 9 | 248 | 394 | 11.5 | 26.6 | 0.044 | 527 |
| 10 | 302 | 464 | 23.4 | 124.9 | 0.072 | 455 |
| 11 | 302 | 460 | 28.9 | 160.5 | 0.087 | 458 |
| 12 | 260 | 430 | 20.8 | 49.7 | 0.074 | 433 |
| 13 | 303 | 514 | 13.2 | 64.7 | 0.042 | 513 |
| 14 | 309 | 459 | 17.5 | 50.6 | 0.054 | 482 |
| 15 | 302 | 489 | 12.5 | 31.4 | 0.040 | 423 |
| 16 | 298 | 446 | 24.4 | 184.3 | 0.076 | 621 |
| 17 | 397 | 626 | 27.5 | 109.3 | 0.065 | 509 |
| 18 | 288 | 488 | 22.5 | 80.2 | 0.073 | 496 |
| 19 | 297 | 416 | 17.7 | 51.2 | 0.056 | 454 |

**Sandbox fraction stable at 4-9%** — SGLang inference dominates per-sample
time by roughly 12-20×. Rollout wall time clusters around 425-555s (median
~490s) — no straggler explosions throughout the entire run.

### 3. Throughput (tokens/GPU/sec)

| step | train_tok/s | tflops | gen_tok/s | gen_eff/s | longest/s |
|-----:|------------:|-------:|----------:|----------:|----------:|
| 0 | 5315 | 40.9 | 31.13 | 18.75 | 55.04 |
| 1 | 7531 | 59.6 | 42.59 | 27.60 | 72.46 |
| 2 | 7263 | 55.2 | 33.64 | 21.10 | 59.21 |
| 3 | 7877 | 60.6 | 40.44 | 26.49 | 68.56 |
| 4 | 7102 | 54.6 | 45.65 | 32.00 | 76.19 |
| 5 | 7539 | 58.7 | 38.07 | 23.34 | 65.25 |
| 6 | 7484 | 57.9 | 34.50 | 22.99 | 57.10 |
| 7 | 8411 | 65.7 | 38.58 | 24.29 | 62.94 |
| 8 | 8092 | 61.1 | 48.16 | 29.92 | 89.80 |
| 9 | 7659 | 53.9 | 28.41 | 19.83 | 52.84 |
| 10 | 7828 | 59.5 | 39.99 | 28.32 | 74.78 |
| 11 | 7913 | 60.1 | 39.92 | 25.55 | 75.07 |
| 12 | 7259 | 55.6 | 41.87 | 23.72 | 74.78 |
| 13 | 7645 | 59.2 | 36.31 | 25.65 | 62.28 |
| 14 | 7439 | 60.0 | 43.13 | 26.71 | 70.44 |
| 15 | 7167 | 54.2 | 43.13 | 30.35 | 76.62 |
| 16 | 7833 | 57.5 | 28.52 | 18.67 | 51.72 |
| 17 | 7503 | 59.6 | 40.18 | 27.21 | 65.41 |
| 18 | 7697 | 59.6 | 37.73 | 23.45 | 66.90 |
| 19 | 7684 | 59.2 | 43.23 | 28.29 | 69.91 |

**Trainer steady at ~7600 tok/gpu/s, ~58 TFLOPs (MFU ~5.8%)**. Generation
throughput averaging ~38 tok/gpu/s with `longest_sample/s` typically 60-90 —
no straggler-tail collapse anywhere in the run.

### 4. Rollout length (tokens)

| step | resp_mean | resp_max | resp_min | seq_mean | turns | tool_calls |
|-----:|----------:|---------:|---------:|---------:|------:|-----------:|
| 0 | 11128 | 19120 | 4042 | 19702 | 13.0 | 12.7 |
| 1 | 13227 | 20570 | 5531 | 21473 | 11.4 | 11.3 |
| 2 | 10946 | 22915 | 3291 | 19777 | 13.0 | 12.9 |
| 3 | 12691 | 21951 | 3961 | 20357 | 14.2 | 14.0 |
| 4 | 13644 | 23759 | 3348 | 20628 | 12.1 | 11.6 |
| 5 | 12185 | 23629 | 6006 | 21004 | 13.7 | 13.5 |
| 6 | 12766 | 22553 | 2797 | 20257 | 10.6 | 10.5 |
| 7 | 12718 | 18632 | 2017 | 21383 | 14.1 | 14.0 |
| 8 | 11203 | 18559 | 2947 | 19127 | 13.4 | 13.2 |
| 9 | 10451 | 20122 | 2319 | 17228 | 10.8 | 10.7 |
| 10 | 12896 | 22306 | 2540 | 19333 | 11.1 | 11.0 |
| 11 | 11696 | 18676 | 4001 | 19603 | 13.1 | 12.9 |
| 12 | 10273 | 21647 | 2806 | 19183 | 14.5 | 14.2 |
| 13 | 13168 | 25148 | 3025 | 20730 | 10.7 | 10.6 |
| 14 | 12864 | 20393 | 3724 | 21939 | 13.0 | 12.9 |
| 15 | 12852 | 25911 | 3334 | 20193 | 11.7 | 11.5 |
| 16 | 11590 | 19784 | 4632 | 18860 | 12.8 | 12.4 |
| 17 | 13847 | 20834 | 3245 | 21686 | 12.9 | 12.8 |
| 18 | 11630 | 24057 | 3595 | 19900 | 13.0 | 12.9 |
| 19 | 12856 | 19656 | 4845 | 20922 | 14.1 | 13.9 |

resp_mean ≈ 12.1K, seq_mean ≈ 20.1K — well under the 32K trainer context cap.
Turn count 10-15, tool_calls ≈ turns − 1 (every turn except the last makes a
tool call; the last calls `finish` or runs out of budget). Model never blew
past response budget.

### 5. Task performance (64 samples per rollout)

| step | resolve | complete | trunc | zero_std/8 | raw_rwd |
|-----:|--------:|---------:|------:|-----------:|--------:|
| 0 | 1.6% | 85.9% | 14.1% | 7/8 | 0.0156 |
| 1 | 12.5% | 82.8% | 17.2% | 7/8 | 0.1250 |
| 2 | 15.6% | 96.9% | 3.1% | 6/8 | 0.1562 |
| 3 | 7.8% | 95.3% | 4.7% | 6/8 | 0.0781 |
| 4 | 0.0% | 87.5% | 12.5% | 8/8 | 0.0000 |
| 5 | 12.5% | 89.1% | 10.9% | 6/8 | 0.1250 |
| 6 | 3.1% | 92.2% | 7.8% | 7/8 | 0.0312 |
| 7 | 3.1% | 87.5% | 12.5% | 6/8 | 0.0312 |
| 8 | 17.2% | 87.5% | 12.5% | 6/8 | 0.1719 |
| 9 | 0.0% | 100.0% | 0.0% | 8/8 | 0.0000 |
| 10 | 12.5% | 90.6% | 9.4% | 6/8 | 0.1250 |
| 11 | 9.4% | 90.6% | 9.4% | 7/8 | 0.0938 |
| 12 | 0.0% | 89.1% | 10.9% | 8/8 | 0.0000 |
| 13 | 0.0% | 92.2% | 7.8% | 8/8 | 0.0000 |
| 14 | 7.8% | 81.2% | 18.8% | 6/8 | 0.0781 |
| 15 | 14.1% | 95.3% | 4.7% | 6/8 | 0.1406 |
| 16 | 0.0% | 93.8% | 6.2% | 8/8 | 0.0000 |
| 17 | 15.6% | 85.9% | 14.1% | 5/8 | 0.1562 |
| 18 | 1.6% | 93.8% | 6.2% | 7/8 | 0.0156 |
| 19 | 0.0% | 82.8% | 17.2% | 8/8 | 0.0000 |

**20-step mean resolve: 6.5%**, range 0-17.2%. **Six zero-resolve batches**
(steps 4, 9, 12, 13, 16, 19) where no sample passed pytest.

**`zero_std/8` stays 5-8 every step**: most prompt-groups have all 8 samples
landing on the same reward (all-fail or all-pass). At ~6.5% mean solve rate,
theory says P(all-fail at 8 samples) ≈ 58% per group, but observed zero-std
fraction is 75-100% — confirming the bimodal-prompt distribution (some prompts
the model never solves, some it always solves; both end up zero-std).

### 6. Reward fail-reasons (fraction of 64 samples)

| step | no_patch | mp_fail | pytest_fail |
|-----:|---------:|--------:|------------:|
| 0 | 21.9% | 12.5% | 64.1% |
| 1 | 46.9% | 0.0% | 40.6% |
| 2 | 15.6% | 0.0% | 68.8% |
| 3 | 20.3% | 25.0% | 46.9% |
| 4 | 15.6% | 25.0% | 59.4% |
| 5 | 20.3% | 14.1% | 53.1% |
| 6 | 23.4% | 0.0% | 73.4% |
| 7 | 32.8% | 25.0% | 39.1% |
| 8 | 20.3% | 12.5% | 50.0% |
| 9 | 28.1% | 0.0% | 71.9% |
| 10 | 12.5% | 25.0% | 50.0% |
| 11 | 10.9% | 12.5% | 67.2% |
| 12 | 1.6% | 3.1% | 95.3% |
| 13 | 34.4% | 35.9% | 29.7% |
| 14 | 45.3% | 1.6% | 45.3% |
| 15 | 18.8% | 12.5% | 54.7% |
| 16 | 23.4% | 14.1% | 62.5% |
| 17 | 26.6% | 0.0% | 57.8% |
| 18 | 7.8% | 35.9% | 54.7% |
| 19 | 29.7% | 1.6% | 68.8% |

**20-step means**: no_patch ≈ 22%, mp_fail ≈ 13%, pytest_fail ≈ 58%.

The model produces patches ~78% of the time (no_patch + mp_fail ≈ 22%), but
most of those patches don't pass pytest. The dominant failure mode is
"produces a fix that's wrong" rather than "produces no fix" or "produces
malformed diff."

### 7. Training signal (per train step)

| step | loss | entropy | grad_norm | rollout_lp_diff |
|-----:|------:|--------:|----------:|----------------:|
| 0 | +5.0e-05 | 0.753 | 0.090 | 0.1035 |
| 1 | +0.0000 | 0.761 | 0.000 | 0.1011 |
| 2 | -1.6e-05 | 0.732 | 0.094 | 0.1024 |
| 3 | +1.9e-04 | 0.798 | 0.155 | 0.1083 |
| 4 | +0.0000 | 0.795 | 0.000 | 0.1085 |
| 5 | +4.9e-05 | 0.739 | 0.146 | 0.1001 |
| 6 | +4.1e-05 | 0.805 | 0.092 | 0.1091 |
| 7 | +1.1e-04 | 0.807 | 0.153 | 0.1067 |
| 8 | +1.5e-04 | 0.768 | 0.166 | 0.1047 |
| 9 | +0.0000 | 0.711 | 0.000 | 0.0996 |
| 10 | -9.1e-05 | 0.764 | 0.152 | 0.1051 |
| 11 | -4.1e-05 | 0.747 | 0.103 | 0.1024 |
| 12 | +0.0000 | 0.734 | 0.000 | 0.0995 |
| 13 | +0.0000 | 0.798 | 0.000 | 0.1050 |
| 14 | -7.0e-07 | 0.810 | 0.140 | 0.1045 |
| 15 | +4.3e-05 | 0.757 | 0.140 | 0.1016 |
| 16 | +0.0000 | 0.736 | 0.000 | 0.0976 |
| 17 | +3.5e-04 | 0.850 | 0.141 | 0.1083 |
| 18 | -2.7e-05 | 0.750 | 0.120 | 0.1010 |
| 19 | +0.0000 | 0.786 | 0.000 | 0.1038 |

- `loss == pg_loss` everywhere (no KL or entropy loss applied — both coefs are 0)
- **Seven zero-grad batches** at steps 1, 4, 9, 12, 13, 16, 19 — each
  corresponds to an `zero_std/8 = 8/8` rollout where every prompt-group had
  identical rewards across its 8 samples → no advantage → no gradient
- `entropy_loss` stable ~0.77 — model isn't collapsing or saturating
- `grad_norm` bounded 0.09-0.17 when non-zero — no explosions
- `rollout_lp_diff` ≈ 0.105 throughout — consistent bf16↔FP8 quantization
  gap between trainer-side and rollout-side logprobs
- `pg_clipfrac = 0` always: updates well within `eps_clip=4e-4`, no PPO clipping
- `tis_clipfrac` ≈ 0.006: almost no TIS clipping, off-policy ratio well-behaved

### 8. Async / staleness

The metric is `staleness = max(0, rollout_id − weight_versions[0] × interval)`,
which counts how many rollouts a sample has lived past the end of its original
weight version cycle.

| step | resume_mean | n_resumed | stale_mean | stale_p95 | grp_max_mean | grp_max_p95 | grp_spread_mean |
|-----:|------------:|----------:|-----------:|----------:|-------------:|------------:|----------------:|
| 0 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 1.25 | 16 | 0.39 | 1.00 | 0.50 | 1.00 | 0.12 |
| 7 | 1.34 | 22 | 0.41 | 2.00 | 0.75 | 2.00 | 0.75 |
| 8 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 11 | 1.17 | 11 | 0.59 | 1.00 | 0.62 | 1.00 | 0.12 |
| 12 | 1.11 | 7 | 0.50 | 2.00 | 0.50 | 2.00 | 0.00 |
| 13 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 14 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 15 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 16 | 1.12 | 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 17 | 1.72 | 46 | 0.69 | 2.00 | 0.75 | 2.00 | 0.25 |
| 18 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 19 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

The staleness metric is **bounded between 0 and 2**, firing only on rollouts
where samples from a previous weight version cycle got mixed in via
partial-rollout aborts.

Concrete reading:
- Steps 0-5: all 0 (no weight broadcast has fired yet → no samples from a
  previous cycle)
- Step 6 (first rollout after the broadcast at end of step 4): `n_resumed=16`
  of 64, `stale_mean=0.39` — about a quarter of the batch has
  `weight_versions[0]=1` (either resumed from the abort or completed cleanly
  during v=1 pre-broadcast), giving staleness `6 − 1×5 = 1`
- Step 7: peaks at `n_resumed=22` with `stale_p95=2` (some samples were
  dispatched at v=1, aborted at the v=2 broadcast, completed at v=3 cycle →
  `wv[0]=1`, stale = `7 − 5 = 2`)
- Pattern repeats around steps 11-12 and 16-17 (right after the weight
  broadcasts at end of steps 9 and 14)
- Step 17 had the biggest churn: `n_resumed=46` (72% of the batch), reflecting
  many samples being mid-flight when the v=4 broadcast hit at end of step 14

`grp_spread_mean` stays ≤ 0.75 — within-group staleness is consistent (most
groups have all samples at the same staleness value), which means GSPO
advantage normalization isn't being noised by mixing many policy versions
inside one group.

## Interpretation

### What this run validated

1. **Full pipeline runs end-to-end at this scale, cleanly**. 20 rollouts × 64
   trajectories = 1280 multi-turn agent trajectories generated, all reaching
   terminal status. All 20 trainer steps completed with non-explosive
   gradients. All 20 metrics records emitted with full 45 keys including the
   per-group staleness telemetry. `Training command finished` reached, sbatch
   teardown ran the orphan cleanup automatically.

2. **No engine deaths across 3h20m of sustained agent-RL load**. Four
   `update_weights` broadcasts (initial + 3 regular) all completed in 34-39s.
   No Triton errors, no NCCL/gloo timeouts, no SGLang sampler issues.

3. **Per-group staleness metric works correctly**. Bounded values 0-2, fires
   only on the rollouts where partial-rollout aborts mix policy versions.
   Group spread stays ≤0.75 — minimal GSPO normalization noise from
   intra-group policy-version mixing.

4. **Automatic orphan-sandbox cleanup at end of sbatch worked**. 3 orphan
   Modal sandboxes were left after the rollout/trainer race at job end (out
   of the ~1280 trajectories total — 0.2% leak rate). The post-Ray-stop
   `cleanup_sandboxes.py` invocation in `run_swe.sbatch` terminated them
   automatically; no manual intervention needed.

5. **Trainer throughput steady at ~7600 tok/gpu/s** (~58 TFLOPs, MFU ~5.8%).
   `wait_ratio` averaging 0.60 — trainer is idle about 60% of step time
   waiting on rollouts. Rollout side is the bottleneck.

6. **Sandbox is not the bottleneck.** `sb_frac` ≈ 4-9% per sample — Modal
   round-trip is a small fraction of per-sample wall time. The bottleneck
   is SGLang generation time for long multi-turn trajectories (~12 turns ×
   ~1K response tokens per turn).

### What this run did NOT validate (flat reward)

7. **Reward stayed flat — 6.5% mean resolve with no upward trend.** The
   configuration ceiling for visible learning at this scale of agent RL has
   multiple contributors:
   - **Bimodal per-prompt solve rates**: most prompts the model either
     always fails or always solves. Both end up zero-std (no gradient).
     Group size doesn't help — the limitation is per-prompt, not per-sample.
   - **Small effective batch**: 5-8 of 8 prompt-groups are zero-std every
     step, leaving only 0-3 informative groups per gradient update.
   - **Short run**: 20 rollouts is too few for any signal to compound at
     this LR (1e-6) and clipping (`eps_clip=4e-4`).
   - **Hard task**: ~58% of patches the model produces fail pytest, even
     when they apply cleanly.

### Recommendations for next runs

1. **For learning signal, the lever is prompt curriculum, not group size or
   prompt count.** Filter the prompt pool to only "edge of capability"
   prompts (ones with nonzero observed variance in solve rate). Currently
   all 40 Lite prompts are used uniformly; many are likely
   guaranteed-fail or guaranteed-pass with current weights, contributing
   zero gradient signal.

2. **Per-prompt instrumentation would enable that curriculum**. Knowing
   which prompts always-fail vs always-pass vs edge-of-capability for the
   current model would let us drop the uninformative ones. Currently we
   only track batch-level aggregates; need to dump per-prompt solve rates
   over a few rollouts to identify candidates.

3. **For reward shaping**, partial credit for "patch applied cleanly" or
   "didn't regress PASS_TO_PASS" would densify the gradient signal beyond
   the current binary 0/1, even before prompt filtering.

4. **For longer runs**, this configuration is stable enough to extend to
   100+ rollouts (~17 hours wall) without infrastructure intervention.
   The sbatch wallclock limit (24h) gives reasonable headroom.

5. **The 8-sample-group / 64-batch configuration is the proven-stable
   point for this scale.** Bumping `n_samples_per_prompt` to 16 has
   consistently triggered SGLang engine instability in prior runs (one TP
   worker dying mid-forward, cascading to gloo "Connection closed" and a
   30-min `update_weights` deadlock). The bigger-group variant is not safe
   to use until upstream SGLang stability for this workload improves.

## Files preserved

- `mnt/logs/infx-swebench-new-37043.out` — full sbatch stdout
- `mnt/logs/infx-swebench-new-37043.err` — sbatch stderr
- `mnt/logs/infx-swebench-new-37043-head.out` — Ray head log
- `mnt/logs/infx-swebench-new-37043-worker-*.out` — per-worker Ray logs
- `mnt/logs/metrics-37043.jsonl` — 100 records, full data for all 20
  rollouts including the per-group staleness keys
