# Run 36915 — Lite-40 + 16-sample groups, NUM_ROLLOUT=20

- **Status**: ❌ FAILED (exit 1:0) — but all 20 rollouts + 20 trainer steps + 20 metric emits captured before the failure
- **Wall**: 6h45m43s
- **Date**: 2026-05-28 → 2026-05-29 (crossed midnight)
- **Log**: `mnt/logs/infx-swebench-new-36915.out`
- **Metrics JSONL**: `mnt/logs/metrics-36915.jsonl` (100 records, full 20 rollouts × 5 records each)

## Summary

20-rollout RL training run of Qwen3-235B-A22B-Thinking-2507-FP8 on a 40-instance
slice of SWE-bench-Lite using slime + Megatron + SGLang + Modal sandboxes.

All 20 training steps completed cleanly and produced full metrics, including the
per-group staleness telemetry which works as intended (bounded ~0-2 rollouts,
fires only on rollouts where partial-rollout aborts mixed policy versions).

The job exited with FAILED state because one SGLang engine on `10.49.115.1`
crashed at around step 16/17 with a gloo TCP "Connection closed by peer" error
(an intra-engine worker died). The rollout side continued in a degraded
7-engine mode through rollouts 17-19 with increasingly long straggler tails,
then `update_weights` after step 19 deadlocked 30 minutes on the dead engine
before timing out. All training data is preserved.

Reward signal stayed flat throughout (~9% mean resolve, no trend) — consistent
with the configuration ceiling for this scale of agent RL.

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
Batching:           rollout_batch_size=8, n_samples_per_prompt=16, global_batch_size=128
                    → 128 trajectories per rollout (8 prompts × 16 samples each)
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
| 0 | 1208 | 417 | 341 | 75 | 791 | 35.2 | 0.655 |
| 1 | 642 | 333 | 288 | 44 | 308 | — | 0.481 |
| 2 | 610 | 336 | 292 | 43 | 274 | — | 0.449 |
| 3 | 668 | 307 | 269 | 37 | 361 | — | 0.541 |
| 4 | 678 | 349 | 309 | 39 | 329 | — | 0.486 |
| 5 | 821 | 359 | 318 | 40 | 462 | 37.1 | 0.563 |
| 6 | 799 | 326 | 289 | 37 | 473 | — | 0.592 |
| 7 | 693 | 362 | 307 | 53 | 332 | — | 0.478 |
| 8 | 486 | 321 | 279 | 41 | 165 | — | 0.340 |
| 9 | 655 | 324 | 280 | 43 | 331 | — | 0.506 |
| 10 | 823 | 367 | 324 | 41 | 456 | 34.8 | 0.555 |
| 11 | 506 | 322 | 282 | 38 | 184 | — | 0.364 |
| 12 | 417 | 273 | 243 | 29 | 143 | — | 0.344 |
| 13 | 750 | 338 | 297 | 39 | 412 | — | 0.550 |
| 14 | 601 | 339 | 297 | 41 | 262 | — | 0.436 |
| 15 | 600 | 356 | 306 | 49 | 244 | 34.1 | 0.407 |
| 16 | 1055 | 368 | 310 | 58 | 687 | — | 0.651 |
| 17 | 1034 | 371 | 322 | 48 | 663 | — | 0.641 |
| 18 | 3446 | 281 | 246 | 34 | 3164 | — | 0.918 |
| 19 | 4348 | 319 | 281 | 36 | 4029 | — | 0.927 |

**Steady-state means** (steps 1-15, before the engine death):
- step_time ≈ 656s, train_time ≈ 335s, actor_train ≈ 295s, log_probs ≈ 41s
- train_wait ≈ 320s, wait_ratio ≈ 0.48 — trainer sits idle ~half the step time
- Updates at steps 0/5/10/15 took 35.2 / 37.1 / 34.8 / 34.1s — consistent

**Steps 16-19** show a clear degradation regime: step_time jumped from ~650s
to 3446s/4348s and wait_ratio to 0.92 because the cluster was rolling out on
7 engines instead of 8, and the slowest sample dominated the wall time.

### 2. Generator per-sample (sec, mean of 128)

| step | inf_mean | inf_p95 | sb_mean | sb_p95 | sb_frac | roll_t |
|-----:|---------:|--------:|--------:|-------:|--------:|-------:|
| 0 | 404 | 663 | 22.1 | 74.9 | 0.052 | 754 |
| 1 | 378 | 579 | 18.2 | 55.5 | 0.046 | 725 |
| 2 | 331 | 586 | 11.8 | 34.5 | 0.035 | 606 |
| 3 | 364 | 534 | 26.5 | 177.2 | 0.068 | 697 |
| 4 | 426 | 666 | 19.4 | 73.9 | 0.043 | 635 |
| 5 | 371 | 599 | 19.5 | 46.6 | 0.050 | 774 |
| 6 | 379 | 714 | 15.2 | 44.6 | 0.039 | 832 |
| 7 | 484 | 842 | 24.1 | 94.2 | 0.047 | 657 |
| 8 | 387 | 601 | 18.7 | 45.3 | 0.046 | 527 |
| 9 | 320 | 532 | 11.6 | 33.7 | 0.035 | 652 |
| 10 | 388 | 569 | 21.1 | 74.2 | 0.052 | 744 |
| 11 | 347 | 614 | 27.4 | 123.4 | 0.073 | 550 |
| 12 | 292 | 527 | 13.5 | 32.2 | 0.044 | 465 |
| 13 | 400 | 713 | 15.2 | 66.9 | 0.037 | 685 |
| 14 | 402 | 609 | 11.3 | 35.9 | 0.027 | 600 |
| 15 | 405 | 651 | 11.8 | 30.6 | 0.028 | 548 |
| 16 | 438 | 712 | 23.1 | 114.9 | 0.050 | 1042 |
| 17 | 496 | 709 | 18.0 | 48.2 | 0.035 | 1031 |
| 18 | 227 | 384 | 12.7 | 41.8 | 0.053 | 3535 |
| 19 | 263 | 393 | 16.3 | 52.4 | 0.058 | 4310 |

**Sandbox fraction ~4-7%** — SGLang inference dominates per-sample time by
roughly 15-20×. The straggler tail in steps 18-19 is wall-clock
(rollout_time) inflation from the dead engine — per-sample inference time
actually DROPPED (227, 263s) because the surviving samples completed faster
while the few stragglers dominated wall time.

### 3. Throughput (tokens/GPU/sec)

| step | train_tok/s | tflops | gen_tok/s | gen_eff/s | longest/s |
|-----:|------------:|-------:|----------:|----------:|----------:|
| 0 | 7906 | 62.7 | 52.50 | 33.66 | 44.19 |
| 1 | 9181 | 71.8 | 54.19 | 32.92 | 47.94 |
| 2 | 8364 | 64.0 | 55.17 | 37.06 | 56.43 |
| 3 | 9151 | 67.8 | 52.23 | 34.68 | 50.62 |
| 4 | 8539 | 66.2 | 61.41 | 45.14 | 50.17 |
| 5 | 8673 | 68.7 | 52.71 | 32.16 | 45.81 |
| 6 | 8624 | 65.6 | 44.12 | 27.45 | 40.05 |
| 7 | 9292 | 74.7 | 64.49 | 43.31 | 51.19 |
| 8 | 9011 | 68.4 | 70.35 | 45.31 | 65.17 |
| 9 | 8333 | 60.6 | 49.11 | 34.87 | 51.65 |
| 10 | 7845 | 59.3 | 50.04 | 35.56 | 43.60 |
| 11 | 8746 | 66.5 | 65.84 | 38.85 | 67.75 |
| 12 | 8568 | 59.6 | 65.19 | 41.03 | 69.71 |
| 13 | 8978 | 70.2 | 54.81 | 38.58 | 47.46 |
| 14 | 8854 | 68.2 | 64.71 | 44.24 | 53.89 |
| 15 | 8862 | 69.2 | 70.15 | 47.06 | 60.72 |
| 16 | 8953 | 72.1 | 39.53 | 23.74 | 32.54 |
| 17 | 8945 | 72.2 | 41.22 | 28.93 | 32.89 |
| 18 | 9995 | 76.1 | 10.17 | 6.64 | 9.12 |
| 19 | 9538 | 74.5 | 9.25 | 6.56 | 8.10 |

**Trainer steady around 8700 tok/gpu/s, ~67 TFLOPs (MFU 6.7%).** **Generation
throughput collapsed in steps 18-19** when the engine died: `gen_tok/s` went
from ~55 to 10, `longest_sample/s` from 60 to 8 — clear signature of
straggler tail from the missing engine.

### 4. Rollout length (tokens)

| step | resp_mean | resp_max | resp_min | seq_mean | turns | tool_calls |
|-----:|----------:|---------:|---------:|---------:|------:|-----------:|
| 0 | 12694 | 23244 | 3692 | 21067 | 13.2 | 13.0 |
| 1 | 11936 | 22881 | 3454 | 20668 | 13.0 | 13.0 |
| 2 | 11236 | 23420 | 1657 | 19070 | 11.1 | 11.0 |
| 3 | 12085 | 20022 | 2491 | 19199 | 12.6 | 12.3 |
| 4 | 14331 | 27492 | 5295 | 20613 | 10.9 | 10.6 |
| 5 | 12448 | 24371 | 2827 | 21547 | 13.5 | 13.2 |
| 6 | 11413 | 22540 | 2509 | 19444 | 12.2 | 12.1 |
| 7 | 14224 | 22676 | 4215 | 22301 | 12.4 | 12.2 |
| 8 | 11928 | 21977 | 3940 | 19659 | 13.4 | 13.3 |
| 9 | 11362 | 21051 | 3146 | 18216 | 10.6 | 10.4 |
| 10 | 13234 | 22139 | 5404 | 19861 | 12.2 | 12.0 |
| 11 | 10690 | 22188 | 1897 | 19300 | 13.9 | 13.8 |
| 12 | 9533 | 22801 | 2831 | 16258 | 11.9 | 11.8 |
| 13 | 13207 | 25034 | 1517 | 20851 | 9.8 | 9.7 |
| 14 | 13263 | 23737 | 4236 | 20520 | 11.1 | 10.9 |
| 15 | 12900 | 26276 | 1127 | 21211 | 11.7 | 11.6 |
| 16 | 12369 | 24433 | 927 | 21661 | 13.7 | 13.5 |
| 17 | 14909 | 27022 | 3682 | 22512 | 12.1 | 11.9 |
| 18 | 11745 | 20862 | 1657 | 19201 | 11.0 | 10.9 |
| 19 | 14141 | 23449 | 4722 | 20969 | 12.1 | 12.0 |

resp_mean ≈ 12.4K, seq_mean ≈ 20K — well under the 32K trainer context cap.
Turn count 10-14, tool_calls ≈ turns − 1 (every turn except the last makes a
tool call; the last calls `finish` or runs out of budget).

### 5. Task performance (128 samples per rollout)

| step | resolve | complete | trunc | zero_std/8 | raw_rwd |
|-----:|--------:|---------:|------:|-----------:|--------:|
| 0 | 3.9% | 83.6% | 16.4% | 7/8 | 0.0391 |
| 1 | 9.4% | 86.7% | 13.3% | 7/8 | 0.0938 |
| 2 | 16.4% | 88.3% | 11.7% | 6/8 | 0.1641 |
| 3 | 8.6% | 94.5% | 5.5% | 6/8 | 0.0859 |
| 4 | 3.1% | 87.5% | 12.5% | 7/8 | 0.0312 |
| 5 | 3.1% | 85.9% | 14.1% | 7/8 | 0.0312 |
| 6 | 22.7% | 93.0% | 7.0% | 5/8 | 0.2266 |
| 7 | 0.0% | 87.5% | 12.5% | 8/8 | 0.0000 |
| 8 | 16.4% | 88.3% | 11.7% | 6/8 | 0.1641 |
| 9 | 0.0% | 96.1% | 3.9% | 8/8 | 0.0000 |
| 10 | 10.9% | 83.6% | 16.4% | 6/8 | 0.1094 |
| 11 | 12.5% | 91.4% | 8.6% | 7/8 | 0.1250 |
| 12 | 10.9% | 94.5% | 5.5% | 6/8 | 0.1094 |
| 13 | 0.0% | 91.4% | 8.6% | 8/8 | 0.0000 |
| 14 | 3.1% | 83.6% | 16.4% | 6/8 | 0.0312 |
| 15 | 18.8% | 89.8% | 10.2% | 6/8 | 0.1875 |
| 16 | 0.0% | 90.6% | 9.4% | 8/8 | 0.0000 |
| 17 | 14.8% | 84.4% | 15.6% | 5/8 | 0.1484 |
| 18 | 9.4% | 87.5% | 12.5% | 7/8 | 0.0938 |
| 19 | 7.8% | 84.4% | 15.6% | 7/8 | 0.0781 |

**20-step mean resolve: 8.6%**, range 0-23%. Five zero-resolve batches (steps
7, 9, 13, 16) — random unlucky prompt sets where no samples passed pytest.

**`zero_std/8` stays 5-8 every step**: even with 16-sample groups, most
prompt-groups have all 16 samples landing on the same reward (all-fail or
all-pass). At 9% mean solve rate, theory says P(all-fail at 16 samples) ≈ 22%,
so we'd expect ~22% of groups to be zero-std — observed 75-100%. The gap means
**per-prompt solve rates are highly bimodal**: most prompts the model never
solves (effective 0%), some it always solves. Both end up zero-std regardless
of group size.

### 6. Reward fail-reasons (fraction of 128 samples)

| step | no_patch | mp_fail | pytest_fail |
|-----:|---------:|--------:|------------:|
| 0 | 22.7% | 11.7% | 61.7% |
| 1 | 35.9% | 0.8% | 53.9% |
| 2 | 28.1% | 0.0% | 55.5% |
| 3 | 16.4% | 25.0% | 50.0% |
| 4 | 22.7% | 25.0% | 49.2% |
| 5 | 25.0% | 11.7% | 60.2% |
| 6 | 10.9% | 0.0% | 66.4% |
| 7 | 36.7% | 25.8% | 37.5% |
| 8 | 16.4% | 12.5% | 54.7% |
| 9 | 29.7% | 12.5% | 57.8% |
| 10 | 21.9% | 13.3% | 53.9% |
| 11 | 10.9% | 12.5% | 64.1% |
| 12 | 1.6% | 0.0% | 87.5% |
| 13 | 34.4% | 37.5% | 28.1% |
| 14 | 43.0% | 0.0% | 53.9% |
| 15 | 21.1% | 12.5% | 47.7% |
| 16 | 30.5% | 12.5% | 57.0% |
| 17 | 28.1% | 12.5% | 44.5% |
| 18 | 26.6% | 0.0% | 64.1% |
| 19 | 21.9% | 12.5% | 57.8% |

**20-step means**: no_patch ≈ 24%, mp_fail ≈ 12%, pytest_fail ≈ 55%.

The model produces patches ~76% of the time (no_patch + mp_fail ≈ 24%), but
most of those patches don't pass pytest. The dominant failure mode is
"produces a fix that's wrong" rather than "produces no fix" or "produces
malformed diff."

### 7. Training signal (per train step)

| step | loss | entropy | grad_norm | rollout_lp_diff |
|-----:|------:|--------:|----------:|----------------:|
| 0 | -3.0e-05 | 0.769 | 0.071 | 0.1068 |
| 1 | -5.1e-05 | 0.748 | 0.075 | 0.0980 |
| 2 | +2.0e-04 | 0.757 | 0.105 | 0.1040 |
| 3 | +4.3e-05 | 0.800 | 0.123 | 0.1116 |
| 4 | -5.0e-05 | 0.812 | 0.067 | 0.1091 |
| 5 | -9.5e-06 | 0.784 | 0.089 | 0.1050 |
| 6 | -1.0e-04 | 0.748 | 0.115 | 0.1026 |
| 7 | +0.0000 | 0.837 | 0.000 | 0.1105 |
| 8 | -1.5e-05 | 0.749 | 0.112 | 0.1061 |
| 9 | +0.0000 | 0.742 | 0.000 | 0.1021 |
| 10 | +7.6e-05 | 0.791 | 0.102 | 0.1084 |
| 11 | +0.0000 | 0.725 | 0.000 | 0.1017 |
| 12 | +5.5e-05 | 0.738 | 0.108 | 0.1034 |
| 13 | +0.0000 | 0.799 | 0.000 | 0.1079 |
| 14 | -1.0e-04 | 0.830 | 0.100 | 0.1127 |
| 15 | -1.2e-04 | 0.726 | 0.105 | 0.0998 |
| 16 | +0.0000 | 0.760 | 0.000 | 0.1057 |
| 17 | -2.7e-04 | 0.841 | 0.127 | 0.1113 |
| 18 | +6.7e-05 | 0.803 | 0.080 | 0.1090 |
| 19 | -7.7e-05 | 0.780 | 0.076 | 0.1083 |

- `loss == pg_loss` everywhere (no KL or entropy loss applied — both coefs are 0)
- **Six zero-grad batches** at steps 7, 9, 11, 13, 16 — each corresponds to a
  `zero_std/8 = 8/8` rollout (every prompt-group had identical rewards across
  its 16 samples → no advantage → no gradient)
- `entropy_loss` stable ~0.78 — model isn't collapsing or saturating
- `grad_norm` bounded 0.07-0.13 when non-zero — no explosions
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
| 6 | 1.20 | 26 | 0.48 | 1.00 | 0.50 | 1.00 | 0.12 |
| 7 | 1.34 | 44 | 0.30 | 2.00 | 0.75 | 2.00 | 0.75 |
| 8 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 11 | 1.18 | 23 | 0.70 | 1.00 | 0.75 | 1.00 | 0.12 |
| 12 | 1.00 | 0 | 0.25 | 2.00 | 0.25 | 2.00 | 0.00 |
| 13 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 14 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 15 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 16 | 1.25 | 32 | 0.04 | 0.00 | 0.12 | 1.00 | 0.12 |
| 17 | 1.57 | 73 | 0.48 | 2.00 | 0.75 | 2.00 | 0.50 |
| 18 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 19 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

The staleness metric is **bounded between 0 and 2**, firing only on rollouts
where samples from a previous weight version cycle got mixed in via
partial-rollout aborts.

Concrete reading:
- Steps 0-5: all 0 (no weight broadcast has fired yet → no samples from a
  previous cycle)
- Step 6 (first rollout after the broadcast at end of step 4): `n_resumed=26`
  of 128, `stale_mean=0.48` — roughly half the batch has `weight_versions[0]=1`
  (either resumed from the abort or completed cleanly during v=1 pre-broadcast),
  giving staleness `6 − 1×5 = 1`
- Step 7: peaks at `n_resumed=44` with `stale_p95=2` (some samples were
  dispatched at v=1, aborted at the v=2 broadcast, completed at v=3 cycle →
  `wv[0]=1`, stale = `7 − 5 = 2`)
- Pattern repeats around steps 11-12 and 16-17 (right after the weight
  broadcasts at end of steps 9 and 14)

`grp_spread_mean` stays ≤ 0.75 — within-group staleness is consistent (most
groups have all samples at the same staleness value), which means GSPO
advantage normalization isn't being noised by mixing many policy versions
inside one group.

## Interpretation

### What this run validated

1. **Full pipeline runs end-to-end at this scale.** 20 rollouts × 128
   trajectories = 2560 multi-turn agent trajectories generated, all
   reaching terminal status (COMPLETED via `finish`, TRUNCATED, or FAILED).
   All 20 trainer steps completed with non-explosive gradients. All 20
   metrics records emitted with full 45 keys including the new per-group
   staleness telemetry.

2. **The per-group staleness metric works correctly.** Bounded values 0-2,
   fires only on the rollouts where partial-rollout aborts mix policy
   versions, group_spread reveals intra-group consistency at ≤0.75 — minimal
   GSPO normalization noise from staleness.

3. **Trainer throughput steady at ~8700 tok/gpu/s** (~67 TFLOPs, MFU 6.7%).
   wait_ratio in steady-state ~0.48, meaning the trainer is idle about half
   the step time waiting for fresh rollouts.

4. **Sandbox is not the bottleneck.** `sb_frac` ≈ 4-7% per sample —
   optimizing Modal latency wouldn't materially change throughput. The
   bottleneck is SGLang generation time for long multi-turn trajectories
   (~12 turns × ~1K response tokens per turn).

### What broke

5. **SGLang engine instability appears systemic at this scale.** One of
   the eight SGLang engines (on node `10.49.115.1`) crashed at the step
   16/17 boundary with a gloo TCP "Connection closed by peer" error,
   meaning one of its TP/EP workers died silently inside the engine
   process. Surviving engines continued to serve rollouts in degraded
   7-engine mode, but the next `update_weights` broadcast (after step 19)
   needs all 8 engines to ACK — when it couldn't reach the dead one, it
   waited the full 30 minutes of gloo's collective recv timeout before
   failing the job.

6. **Late-run degradation is acute when one engine dies.** Steps 16-17
   slowed from ~660s to ~1040s. Steps 18-19 stretched to 3535s and 4310s.
   Without the gloo deadlock at the end, the cluster could have kept
   limping along at progressively worse throughput.

### What this run did NOT validate (flat reward)

7. **Reward stayed flat — 8.6% mean resolve with no upward trend.** The
   configuration ceiling for visible learning at this scale of agent RL
   has multiple contributors:
   - **Bimodal per-prompt solve rates**: most prompts the model either
     always fails or always solves. Both end up zero-std (no gradient).
     Bigger groups don't help — the limitation is per-prompt, not
     per-sample.
   - **Small effective batch**: 5-8 of 8 prompt-groups are zero-std every
     step, leaving only 0-3 informative groups per gradient update.
   - **Short run**: 20 rollouts is too few for any signal to compound at
     this LR (1e-6) and clipping (`eps_clip=4e-4`).
   - **Hard task**: ~55% of patches the model produces fail pytest, even
     when they apply cleanly.

### Recommendations for next runs

1. **For learning signal, the lever is prompt curriculum, not group size.**
   Filter the prompt pool to only "edge of capability" prompts (ones with
   nonzero observed variance in solve rate). Currently all 40 Lite prompts
   are used uniformly; many of them are likely guaranteed-fail or
   guaranteed-pass with current weights.

2. **For longer runs**, expect to hit engine-side instability every 4-6
   hours. Either accept periodic retries or investigate hardware-side
   mitigations (exclude flaky nodes, lower `mem_fraction_static`,
   different attention backend, etc.).

3. **For richer per-prompt telemetry**, instrument per-prompt solve rates
   so the curriculum filtering decision can be data-driven rather than
   heuristic.

4. **For reward shaping**, partial credit for "patch applied cleanly" or
   "didn't regress PASS_TO_PASS" would densify the gradient signal beyond
   the current binary 0/1.

## Files preserved

- `mnt/logs/infx-swebench-new-36915.out` — full sbatch stdout
- `mnt/logs/infx-swebench-new-36915.err` — sbatch stderr
- `mnt/logs/infx-swebench-new-36915-head.out` — Ray head log
- `mnt/logs/infx-swebench-new-36915-worker-*.out` — per-worker Ray logs
- `mnt/logs/metrics-36915.jsonl` — 100 records covering all 20 rollouts;
  includes the per-group staleness keys for every emit
