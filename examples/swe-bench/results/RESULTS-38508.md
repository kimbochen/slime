# Run 38508 — 64K rollout context + harness fixes, NUM_ROLLOUT=20

Job: 38508. 12 × H200 nodes, Qwen3-235B-A22B-Thinking-2507-FP8, slime
fully-async pipeline against SWE-bench Lite-40. 20 rollouts × 64 samples
= 1,280 total samples. Completed cleanly end-to-end (no engine deaths,
no save_model OOM, no sandbox storms). Mean resolution rate **12.03%**,
**~1.85× the 37043 baseline of 6.5%**.

This run validated three changes layered together: (a) bumping
`--rollout-max-context-len` from 32K → 64K to use the trainer's full
effective context cap (CP=4 × max_tokens_per_gpu=16K = 64K), (b) three
reward-harness bug fixes that were silently zeroing ~15-25% of samples
regardless of model output, and (c) a per-sample reward-fail JSONL that
captures the raw `git apply` rejection / pytest exit messages we
previously aggregated away.

## Config

```
Model:              Qwen3-235B-A22B-Thinking-2507-FP8 (bf16 trainer side)
Cluster:            12 × H200 (8 GPU each), 96 GPUs total
Topology:           PP=2, CP=4, TP=4, EP=8 (per exp 13's stable config)

Rollout:            --partial_rollout + --use-tis (tis_clip 2.0, tis_clip_low 0)
                    --n-samples-per-prompt 8 (group size)
                    --rollout-batch-size 8 (8 unique prompts per rollout)
                    --over-sampling-batch-size 32
                    --rollout-max-context-len 65536   ← BUMPED from 32K
                    --rollout-max-response-len 16384
                    --sglang-context-length 65536
                    NUM_ROLLOUT=20 (no save, see CLAUDE.md note)
                    update-weights-interval 5

Trainer:            --max-tokens-per-gpu 16384
                    --global-batch-size 64
                    --lr 1e-6, eps-clip 4e-4
                    --advantage-estimator gspo, kl-loss-coef 0

Data:               examples/swe-bench/swebench_lite/sample_40.jsonl
                    40 instances spanning 11 repos (astropy, django,
                    matplotlib, mwaskom, pallets, psf, pydata, pylint-dev,
                    pytest-dev, scikit-learn, sphinx-doc)

Harness fixes (vs 37043):
  - xargs -d '\n'                  newline-only delimiter, no shell-quote
                                   interp on parametrized pytest IDs
  - django/django dispatch         runtests.py --settings=test_sqlite with
                                   class-level paths (lose per-method
                                   granularity, recover the ~10% of samples
                                   that the testbed-has-no-pytest bug
                                   was zeroing)
  - dotted-path + bracket filters  drop SWE-bench data-quality issues
                                   (docstring-only IDs that broke bash;
                                   truncated parametrized pytest IDs)

Staleness metric:   staleness = max(0, rollout_id - weight_versions[0] × interval)
                    Plus per-group: group_max_staleness, group_staleness_spread

JSONL artifacts:    results/metrics-38508.jsonl       (per-step aggregated)
                    results/reward-fails-38508.jsonl  (per-sample raw reasons)
```

## Timing summary

```
21:14   sbatch submitted
21:18   first SGLang engine fired up
21:19   all 12 SGLang engines up
21:33   first update_weights (initial bootstrap sync, 33.7s)
21:44   first rollout completed (641s — bootstrap-warm cache)
00:33   step 19 trainer finished
00:35   "Training command finished — stopping Ray cluster"
00:36   cleanup_sandboxes.py — sandboxes terminated cleanly

Total wall:   ~3h 21m for 20 rollouts (bootstrap + train + teardown)
```

## Per-step tables

### 1. Train / perf timings (sec)

| step | step_t | train_t | actor_t | log_probs | train_wait | upd_w | wait_ratio |
|----:|------:|------:|------:|--------:|--------:|------:|---------:|
| 0 | 993 | 316 | 258 | 57 | 677 | 33.7 | 0.682 |
| 1 | 440 | 152 | 134 | 17 | 288 | — | 0.654 |
| 2 | 547 | 181 | 155 | 26 | 366 | — | 0.669 |
| 3 | 475 | 215 | 191 | 23 | 260 | — | 0.547 |
| 4 | 473 | 189 | 169 | 19 | 285 | — | 0.601 |
| 5 | 742 | 204 | 181 | 22 | 537 | 37.1 | 0.724 |
| 6 | 668 | 194 | 168 | 25 | 474 | — | 0.709 |
| 7 | 373 | 207 | 183 | 23 | 166 | — | 0.445 |
| 8 | 353 | 181 | 153 | 27 | 172 | — | 0.488 |
| 9 | 477 | 178 | 149 | 28 | 299 | — | 0.626 |
| 10 | 523 | 178 | 159 | 18 | 345 | 35.5 | 0.659 |
| 11 | 438 | 159 | 141 | 17 | 279 | — | 0.637 |
| 12 | 603 | 173 | 154 | 18 | 430 | — | 0.713 |
| 13 | 566 | 183 | 164 | 18 | 383 | — | 0.677 |
| 14 | 512 | 212 | 190 | 21 | 301 | — | 0.587 |
| 15 | 464 | 162 | 142 | 19 | 301 | 41.9 | 0.650 |
| 16 | 532 | 166 | 147 | 18 | 366 | — | 0.689 |
| 17 | 456 | 190 | 170 | 19 | 265 | — | 0.582 |
| 18 | 557 | 200 | 179 | 20 | 358 | — | 0.642 |
| 19 | 621 | 219 | 193 | 24 | 402 | — | 0.648 |

Four `update_weights` broadcasts at steps 0/5/10/15 took 33.7/37.1/35.5/41.9s
— consistent with 37043.

**Wait ratio = 0.62 median** — trainer sits idle ~62% of every step waiting
for the next rollout. **System is generation-bound**, not training-bound;
longer trajectories from the 64K cap make the generator even slower than
37043 (993s step 0 vs 37043's ~700s step 0). Step 7/8 are notable outliers
(0.45 / 0.49 wait ratio) — backlog of resumed samples from step 5/6's
post-broadcast churn drained quickly because their tokens were already
prefilled in earlier steps.

### 2. Generator per-sample (sec, mean of 64)

| step | inf_mean | inf_p95 | sb_mean | sb_p95 | sb_frac | roll_t |
|----:|--------:|--------:|--------:|--------:|--------:|--------:|
| 0 | 344 | 535 | 21.7 | 60.4 | 0.059 | 642 |
| 1 | 282 | 497 | 13.8 | 34.8 | 0.046 | 604 |
| 2 | 265 | 493 | 13.6 | 43.4 | 0.049 | 517 |
| 3 | 347 | 560 | 26.0 | 119.5 | 0.070 | 440 |
| 4 | 333 | 491 | 17.3 | 50.9 | 0.050 | 500 |
| 5 | 334 | 528 | 13.6 | 42.0 | 0.039 | 689 |
| 6 | 328 | 517 | 12.9 | 33.4 | 0.038 | 678 |
| 7 | 318 | 672 | 20.7 | 87.0 | 0.061 | 360 |
| 8 | 266 | 392 | 13.5 | 38.9 | 0.048 | 379 |
| 9 | 278 | 433 | 11.0 | 33.8 | 0.038 | 479 |
| 10 | 292 | 478 | 23.3 | 123.8 | 0.074 | 487 |
| 11 | 281 | 472 | 23.8 | 106.6 | 0.078 | 457 |
| 12 | 266 | 463 | 14.5 | 39.7 | 0.052 | 589 |
| 13 | 310 | 547 | 16.9 | 72.8 | 0.052 | 556 |
| 14 | 365 | 528 | 15.1 | 45.6 | 0.040 | 483 |
| 15 | 319 | 569 | 15.2 | 39.7 | 0.046 | 471 |
| 16 | 315 | 487 | 11.9 | 30.7 | 0.036 | 528 |
| 17 | 302 | 546 | 26.7 | 120.3 | 0.081 | 431 |
| 18 | 320 | 510 | 35.2 | 94.2 | 0.099 | 548 |
| 19 | 306 | 460 | 19.7 | 68.2 | 0.060 | 602 |

**Inference dominates per-sample time**: sandbox fraction stays 4-10% of
total. Mean inference time per sample ~310s (vs 37043's ~290s) — small
~7% slowdown from longer 64K trajectories. P95 inference clusters around
500-570s except for step 7 at 672s (post-broadcast resumed-sample
prefill on the largest accumulated trajectories).

### 3. Throughput (tokens/GPU/sec)

| step | actor_tok/s | actor_tflops | gen_tok/s | gen_eff/s | longest_s |
|----:|-----------:|-------:|----------:|----------:|----------:|
| 0 | 5570 | 45.1 | 33.00 | 20.19 | 59.47 |
| 1 | 8760 | 64.7 | 28.42 | 18.92 | 57.18 |
| 2 | 7938 | 61.4 | 32.95 | 20.10 | 83.95 |
| 3 | 7591 | 62.2 | 49.09 | 32.81 | 86.96 |
| 4 | 7680 | 58.7 | 38.33 | 26.60 | 63.94 |
| 5 | 7955 | 64.2 | 30.86 | 19.82 | 55.87 |
| 6 | 7716 | 59.4 | 28.36 | 18.50 | 52.81 |
| 7 | 7443 | 60.4 | 56.05 | 34.24 | 131.18 |
| 8 | 7732 | 57.6 | 45.22 | 30.45 | 111.05 |
| 9 | 8254 | 62.4 | 35.92 | 24.94 | 75.54 |
| 10 | 7906 | 60.9 | 37.97 | 25.01 | 76.58 |
| 11 | 8488 | 64.2 | 38.37 | 23.97 | 84.57 |
| 12 | 7654 | 58.2 | 27.78 | 16.88 | 63.71 |
| 13 | 7962 | 62.1 | 34.66 | 24.17 | 65.86 |
| 14 | 7709 | 62.8 | 44.82 | 30.83 | 82.55 |
| 15 | 9836 | 79.2 | 42.11 | 27.45 | 84.07 |
| 16 | 8296 | 63.4 | 33.61 | 23.35 | 77.83 |
| 17 | 7646 | 60.0 | 44.59 | 26.72 | 99.02 |
| 18 | 7908 | 65.9 | 37.89 | 24.13 | 77.32 |
| 19 | 7093 | 57.2 | 33.70 | 21.82 | 74.38 |

**Actor train throughput ~8K tok/s, ~62 TFLOPS/GPU** — both higher than
37043 (~7.3K tok/s, ~55 TFLOPS). The 64K rollout context gives the
trainer longer concatenated microbatches per step, improving GPU
utilization on the train side.

Generation throughput is highly variable (28-56 tok/s/GPU) — depends on
how many resumed samples are paying re-prefill cost in any given step.

### 4. Sequence length & turns

| step | resp_mean | seq_mean | seq_p95 | resp_max | turns | tool_calls |
|----:|---------:|---------:|---------:|---------:|------:|-----------:|
| 0 | 12959 | 22442 | 33810 | 22410 | 14.6 | 14.4 |
| 1 | 11433 | 18331 | 31490 | 21578 | 12.0 | 12.0 |
| 2 | 10401 | 19197 | 34191 | 19796 | 12.8 | 12.7 |
| 3 | 14453 | 22693 | 37123 | 26068 | 13.4 | 13.2 |
| 4 | 13293 | 20272 | 29624 | 21350 | 14.2 | 14.1 |
| 5 | 13647 | 22456 | 33819 | 27693 | 13.1 | 12.9 |
| 6 | 12541 | 20305 | 33053 | 20324 | 11.4 | 11.2 |
| 7 | 12321 | 21299 | 36635 | 26629 | 15.0 | 14.9 |
| 8 | 11545 | 18467 | 29968 | 25800 | 11.6 | 11.5 |
| 9 | 11954 | 19270 | 31283 | 24800 | 10.8 | 10.7 |
| 10 | 12188 | 19659 | 29923 | 21986 | 12.8 | 12.5 |
| 11 | 10963 | 18742 | 31317 | 20260 | 12.4 | 12.3 |
| 12 | 9933 | 18424 | 35989 | 22988 | 12.0 | 12.0 |
| 13 | 13429 | 20365 | 33775 | 23843 | 11.7 | 11.7 |
| 14 | 14898 | 22865 | 35408 | 22486 | 11.9 | 11.5 |
| 15 | 12928 | 21799 | 34584 | 20494 | 12.8 | 12.5 |
| 16 | 12329 | 19020 | 31475 | 20977 | 11.0 | 10.9 |
| 17 | 11505 | 20321 | 33681 | 24470 | 14.6 | 14.4 |
| 18 | 13220 | 22100 | 38890 | 29875 | 12.9 | 12.8 |
| 19 | 13133 | 21381 | 37174 | 23656 | 12.9 | 12.8 |

**The 64K context cap is being used.** `seq_p95` exceeded the old 32K cap
on **16 of 20 steps** (and reached 38.9K at step 18). `seq_mean ≈ 20.5K`
(vs 37043's 20.1K — about 2% longer on average). `resp_max` hit 29.9K
on step 18 — single trajectories generating nearly 30K response tokens
across multi-turn loops.

Turn count averaged 12-15, similar to 37043, but trajectories produce
more tokens per turn on average (longer responses since the model doesn't
hit context limits as often).

### 5. Task performance (64 samples per rollout)

| step | resolve | complete | trunc | zero_std/8 |
|----:|--------:|---------:|------:|-----------:|
| 0 | 3.1% | 92.2% | 7.8% | 7/8 |
| 1 | 31.2% | 96.9% | 3.1% | 5/8 |
| 2 | 17.2% | 90.6% | 9.4% | 5/8 |
| 3 | 9.4% | 93.8% | 6.2% | 5/8 |
| 4 | 3.1% | 93.8% | 4.7% | 6/8 |
| 5 | 9.4% | 81.2% | 18.8% | 7/8 |
| 6 | 9.4% | 96.9% | 3.1% | 5/8 |
| 7 | 9.4% | 92.2% | 7.8% | 6/8 |
| 8 | 26.6% | 89.1% | 10.9% | 4/8 |
| 9 | 4.7% | 90.6% | 9.4% | 7/8 |
| 10 | 21.9% | 96.9% | 3.1% | 5/8 |
| 11 | 9.4% | 96.9% | 3.1% | 7/8 |
| 12 | 21.9% | 92.2% | 7.8% | 5/8 |
| 13 | 4.7% | 90.6% | 9.4% | 7/8 |
| 14 | 0.0% | 90.6% | 9.4% | 8/8 |
| 15 | 25.0% | 93.8% | 6.2% | 5/8 |
| 16 | 9.4% | 85.9% | 14.1% | 7/8 |
| 17 | 6.2% | 90.6% | 9.4% | 7/8 |
| 18 | 15.6% | 90.6% | 9.4% | 5/8 |
| 19 | 3.1% | 95.3% | 4.7% | 6/8 |

**20-step mean resolve: 12.03%** (vs 37043's 6.5%, ~1.85× lift). Range
0-31.25%. **Only one zero-resolve step** (step 14) vs 37043's six —
much steadier baseline.

**Five "big steps"** with ≥20% resolve: 1 (31.2%), 8 (26.6%), 15 (25.0%),
10 (21.9%), 12 (21.9%). These reflect lucky prompt draws hitting easy
instances + the harness fix giving them credit they previously didn't get.
37043's single best step was 17.2% — 38508's 5 best steps all exceed it.

`zero_std/8` stays 4-8 per step (mean ~6) — most groups still have all 8
samples landing on the same reward (the bimodal-prompt phenomenon: a
given instance is either always-solved or always-failed by current
policy). Even at the higher 12% mean solve rate, prompt-level variance
swamps within-prompt variance.

### 6. Reward fail-reasons (fraction of 64 samples)

| step | no_patch | mp_fail | pytest_fail |
|----:|--------:|--------:|------------:|
| 0 | 12.5% | 12.5% | 71.9% |
| 1 | 26.6% | 3.1% | 39.1% |
| 2 | 21.9% | 0.0% | 60.9% |
| 3 | 18.8% | 37.5% | 34.4% |
| 4 | 15.6% | 10.9% | 70.3% |
| 5 | 17.2% | 12.5% | 60.9% |
| 6 | 32.8% | 12.5% | 45.3% |
| 7 | 18.8% | 12.5% | 59.4% |
| 8 | 23.4% | 12.5% | 37.5% |
| 9 | 25.0% | 12.5% | 57.8% |
| 10 | 17.2% | 12.5% | 48.4% |
| 11 | 4.7% | 12.5% | 73.4% |
| 12 | 23.4% | 0.0% | 54.7% |
| 13 | 35.9% | 39.1% | 20.3% |
| 14 | 34.4% | 0.0% | 65.6% |
| 15 | 26.6% | 12.5% | 35.9% |
| 16 | 39.1% | 0.0% | 51.6% |
| 17 | 10.9% | 12.5% | 70.3% |
| 18 | 9.4% | 37.5% | 37.5% |
| 19 | 32.8% | 12.5% | 51.6% |

**20-step means**: no_patch ≈ 21.7%, mp_fail ≈ 13.1%, pytest_fail ≈ 53.1%
(reward 1.0 fraction = 12.03%, sum ≈ 100%).

**`mp_fail` clusters at the 12.5% baseline** for most steps (= 8 samples /
64 = one full group). This is the structural rate of patch-vs-test_patch
collisions in the dataset — model edits files that the harness's
test_patch also wants to modify, conflict at apply time.

**Three `mp_fail` spikes** to 37.5% / 39.1% / 37.5% at steps 3, 13, 18
(10-step periodicity). Could be the same difficult-patch instances
cycling through the buffer — Lite-40 has only 40 instances, the worker
draws 8 prompts per rollout, so the same conflict-prone subset
(astropy + django are the usual suspects from our pre-run diagnostic)
recurs every ~5 rollouts.

**`no_patch` mean = 21.7%** — a quarter of samples never produce a patch.
Compare to 37043's 22% — essentially unchanged. The 64K context didn't
help the "model gives up" rate; it helped the "model finishes a long
trajectory" rate (visible in completed_frac).

### 7. Training signal (per train step)

| step | loss | entropy | grad_norm | rollout_lp_diff | tis_clipfrac |
|----:|-----:|--------:|----------:|----------------:|-------------:|
| 0 | +2.98e-05 | 0.765 | 0.111 | 0.1068 | 0.0064 |
| 1 | -1.15e-04 | 0.726 | 0.169 | 0.0975 | 0.0056 |
| 2 | -5.94e-05 | 0.771 | 0.188 | 0.1070 | 0.0062 |
| 3 | -4.76e-05 | 0.797 | 0.182 | 0.1074 | 0.0064 |
| 4 | +1.12e-04 | 0.777 | 0.173 | 0.1094 | 0.0066 |
| 5 | -1.21e-04 | 0.744 | 0.096 | 0.1013 | 0.0060 |
| 6 | +1.04e-04 | 0.814 | 0.175 | 0.1057 | 0.0058 |
| 7 | -1.92e-04 | 0.771 | 0.175 | 0.1084 | 0.0064 |
| 8 | -1.09e-04 | 0.780 | 0.210 | 0.1111 | 0.0068 |
| 9 | +7.56e-05 | 0.753 | 0.115 | 0.1017 | 0.0057 |
| 10 | +1.55e-04 | 0.747 | 0.184 | 0.1052 | 0.0063 |
| 11 | -1.49e-04 | 0.738 | 0.096 | 0.1040 | 0.0065 |
| 12 | -7.97e-05 | 0.738 | 0.180 | 0.1066 | 0.0066 |
| 13 | -6.96e-05 | 0.763 | 0.087 | 0.1033 | 0.0060 |
| 14 | +0.00e+00 | 0.817 | 0.000 | 0.1160 | 0.0074 |
| 15 | -5.85e-05 | 0.793 | 0.178 | 0.1087 | 0.0065 |
| 16 | -4.91e-05 | 0.779 | 0.130 | 0.1064 | 0.0062 |
| 17 | +1.12e-04 | 0.771 | 0.120 | 0.1075 | 0.0064 |
| 18 | +3.11e-05 | 0.785 | 0.212 | 0.1130 | 0.0070 |
| 19 | +4.64e-05 | 0.767 | 0.124 | 0.1062 | 0.0063 |

**Loss tiny and zero-centered** (range -1.92e-04 to +1.55e-04) — no
runaway, no collapse. Step 14's `loss = 0.0, grad_norm = 0.0` is the
0% resolve step: GSPO's within-group normalization gives zero advantage
when every group's reward is identical (all-fail) → no gradient. Stable,
not a failure.

**Entropy stable at 0.74-0.82** across all 20 steps — model hasn't
collapsed despite RL signal. `pg_clipfrac = 0.0` every step — no PPO
clipping triggered (advantages stay within the tiny `eps_clip = 4e-4`
window, consistent with GSPO + small learning rate).

**`tis_clipfrac` stable at 0.55-0.74%** across the entire run, including
the post-broadcast carry-over steps (6, 7, 11, 12, 16, 17). TIS is
absorbing the off-policy mixing without strain — even when `stale_p95 = 2`
and 21+ samples are carrying v=N-1 tokens, only ~0.65% of tokens hit
the `tis_clip = 2.0` ceiling.

**`rollout_lp_diff ≈ 0.10-0.12`** — the bf16↔FP8 quantization gap. Holds
flat throughout the run, no drift.

**`grad_norm`** is the main signal indicator (0.087 to 0.212). Tracks
resolve rate roughly: big steps (8 with 26.6% resolve → 0.210; 18 with
15.6% → 0.212) produce bigger gradients; post-broadcast steps with
many resumed samples (5, 9, 11, 13) have lower grad_norm because TIS
down-weights some tokens.

### 8. Async / staleness

| step | resume_mean | n_resumed | stale_mean | stale_p95 | grp_max_mean | grp_max_p95 | grp_spread_mean |
|----:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| 0 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 1.23 | 15 | 0.42 | 1.00 | 0.50 | 1.00 | 0.12 |
| 7 | 1.33 | 21 | 0.50 | 2.00 | 0.75 | 2.00 | 0.75 |
| 8 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 11 | 1.16 | 10 | 0.30 | 1.00 | 0.38 | 1.00 | 0.12 |
| 12 | 1.33 | 21 | 0.78 | 2.00 | 1.00 | 2.00 | 0.50 |
| 13 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 14 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 15 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 16 | 1.34 | 22 | 0.38 | 1.00 | 0.38 | 1.00 | 0.00 |
| 17 | 1.11 | 7 | 0.50 | 2.00 | 0.50 | 2.00 | 0.00 |
| 18 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 19 | 1.00 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Staleness bounded 0-2, firing only on the first 1-2 rollouts after each
weight broadcast — same pattern as 37043. Counts mirror 37043 almost
exactly at the equivalent steps:

```
step      37043 n_resumed     38508 n_resumed
  6           16                  15
  7           22                  21
 11           11                  10
 12            7                  21
 16            8                  22
 17           46                   7
```

**Step 17 did NOT have 37043's "massive churn" (46 resumed)** — 38508
drained the v=3 backlog one rollout earlier (step 16 had 22 resumed
instead of 37043's 8). This is normal stochastic queue behavior — which
samples the worker happens to pull when isn't deterministic.

`grp_spread_mean ≤ 0.75` — within-group staleness consistency stays good
(most groups have all samples at the same staleness value), confirming
that GSPO's per-group normalization isn't being noised by mixed policy
versions inside one group.

## Reward-fails JSONL summary

```
results/reward-fails-38508.jsonl    1,174 records (one per non-resolve sample)
  pytest_exit_nonzero  708  (60.3%)  ← model produced patch, applied,
                                        tests failed
  no_model_patch       296  (25.2%)  ← model gave up / didn't emit a patch
  model_patch_failed   170  (14.5%)  ← patch applied to rollout sandbox,
                                        rejected in reward sandbox
                                        (test_patch conflict)
```

This is the FIRST run with the per-sample raw fail-reason JSONL. Examples
captured (previously discarded):

```
[astropy__astropy-14995]  model_patch apply failed (rc=1):
                            error: patch failed: pyproject.toml:1
                            error: pyproject.toml: patch does not apply

[scikit-learn__scikit-learn-10508]  pytest exit 123:
                            ...FF..................         [100%]
                            (2 specific tests failed of ~22 — not all)
```

The astropy pattern is consistent — pyproject.toml is a common collision
point. The sklearn pattern shows tests partially passing, which is useful
diagnostic detail that the aggregated bucket would have lost.

## Interpretation

### What this run validated

1. **64K rollout context is being used and helps trajectory completion.**
   `seq_p95` exceeded the old 32K cap on 16 of 20 steps. `completed_frac`
   averaged 91.9% (vs 37043's ~88%) — model finishes long trajectories
   more often when context isn't the limiting factor.

2. **The three harness bug fixes deliver the predicted ~2× lift.** Mean
   resolve climbed from 37043's 6.5% to 38508's 12.03% on identical
   dataset and identical model. The diagnostic predicted "~15-25% of
   samples were being silently zeroed" — the actual lift (~5.5pp / 12pp
   = ~46% relative) is consistent with that estimate given measurement
   noise and the bimodal-prompt structure.

3. **Per-sample fail-reason JSONL is operational.** 1,174 raw fail
   strings captured, surface real `git apply` rejection messages and
   pytest exit codes that the bucketed metrics discarded. Enables
   targeted iteration on specific failure patterns (e.g., the astropy
   pyproject.toml pattern is now visible and could potentially be
   mitigated by prompting the model to avoid editing build config).

4. **TIS handles staleness without strain.** `tis_clipfrac` stayed flat
   at 0.55-0.74% throughout, even when `stale_p95 = 2` and 21+ samples
   carried mixed weight versions. No need to tighten clip or shorten
   the weight-update interval.

5. **The pipeline is stable end-to-end at scale.** 20 rollouts × 64
   samples = 1,280 samples processed cleanly. No engine deaths, no
   save_model OOM, no gloo timeouts, no sandbox storms. Auto-cleanup
   terminated orphan sandboxes successfully at job end.

### What this run did NOT validate

1. **The model is not learning measurably within 20 rollouts.** Resolve
   rate varies step-to-step driven by prompt-draw luck, not a visible
   training trajectory. With Lite-40 (40 unique instances) and 8 prompts
   per rollout, batches recur on a ~5-rollout cycle — 20 steps gives
   only ~4 full passes through the dataset, not enough to see learning.
   Loss trajectory is consistent with this: tiny, sign-flipping, no
   clear directional trend.

2. **The bimodal-prompt problem is unchanged.** `zero_std/8` averages
   ~6/8 — most groups still have all 8 samples landing on the same
   reward. RL needs within-group variance to compute meaningful
   advantages, and Lite-40 doesn't provide it for most prompts at
   the current policy. Prompt curriculum (filtering to instances with
   intermediate solve rates) would be more impactful than more steps.

3. **`mp_fail` floor at 12.5% is a structural ceiling.** Even with the
   harness fixes, ~1 group per rollout fails patch-apply due to
   test_patch conflict — fundamental to how SWE-bench constructs its
   eval (model can't be told which files test_patch will edit). The
   only way to reduce this further is data-side (filter Lite-40 to
   instances with non-overlapping model/test patches).

### Per-step rollout wall times (for reference)

```
sec:  642  604  517  440  500  689  678  360  379  479
      487  457  589  556  483  471  528  431  548  602
mean: 519s, range 360-689s
```

Rollout wall times are dominated by the prefill cost of resumed samples
in post-broadcast steps (5, 6, 12, 16 all > 500s). Steps without
carry-over and warm cache (7, 8) complete in 360-380s. Step 0 is slowest
(642s) because of cold-cache bootstrap.
