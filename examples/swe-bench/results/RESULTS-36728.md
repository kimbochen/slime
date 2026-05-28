# Run 36728 — PP=2 CP=4 + partial_rollout, NUM_ROLLOUT=20

- **Status**: ✅ COMPLETED (exit 0:0)
- **Wall**: 3h29m24s
- **Date**: 2026-05-28
- **Log**: `mnt/logs/infx-swebench-new-36728.out`
- **Metrics JSONL**: `mnt/logs/metrics-36728.jsonl` (100 records)
- **Sbatch script (frozen at submit)**: `examples/swe-bench/run_swe.sbatch`
- **Inner script (frozen at submit)**: `examples/swe-bench/run_swe.sh`

## Summary

First end-to-end completion of the barebones SWE-bench example. Validates
the PP=2 CP=4 trainer reshape (matching the reference's exp 13 stable
configuration) survives 4 weight updates across 20 rollouts without the
gloo-timeout failure class that killed jobs 36556/36557/36615 on PP=4 CP=2.
Pipeline ran cleanly: 20/20 rollouts, 20/20 trainer steps, 20/20 swebench
metric emits, clean `Training command finished` teardown.

Reward signal is flat as expected (mean resolve 14.6%, oscillating 0-23%
across 20 steps). 20 rollouts is too few + 10-prompt pool is too small +
8-sample groups is too narrow for visible learning. This run is a
**stability / throughput experiment**, not a learning experiment.

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
Data:               10 SWE-bench-Verified instances (mnt/data/swebench_verified/sample.jsonl)
                    Each prompt used ~16 times across the 20-step run (high reuse)
Batching:           rollout_batch_size=8, n_samples_per_prompt=8, global_batch_size=64
                    → 64 trajectories per rollout (8 prompts × 8 samples)
Weight updates:     every 5 rollouts (interval=5) → 4 updates (initial + 3 regular)
Custom hooks:       generate_with_openhands.generate, reward.compute_reward,
                    metrics.log_rollout_data
```

## Tables

### 1. Trainer step time breakdown (sec/step)

| step | step_t | train_t | actor_t | logp_t | wait_t | upd_w | wait_r |
|-----:|-------:|--------:|--------:|-------:|-------:|------:|-------:|
| 0 | 995 | 319 | 260 | 58 | 676 | 34.3 | 0.680 |
| 1 | 455 | 204 | 181 | 21 | 252 | — | 0.553 |
| 2 | 593 | 250 | 221 | 29 | 343 | — | 0.578 |
| 3 | 497 | 206 | 183 | 22 | 291 | — | 0.585 |
| 4 | 455 | 212 | 188 | 23 | 243 | — | 0.534 |
| 5 | 543 | 206 | 183 | 21 | 337 | 35.7 | 0.621 |
| 6 | 499 | 196 | 172 | 22 | 303 | — | 0.608 |
| 7 | 462 | 206 | 181 | 24 | 256 | — | 0.553 |
| 8 | 467 | 204 | 180 | 23 | 263 | — | 0.563 |
| 9 | 562 | 204 | 180 | 23 | 358 | — | 0.637 |
| 10 | 489 | 198 | 174 | 23 | 291 | 33.7 | 0.595 |
| 11 | 601 | 211 | 188 | 21 | 390 | — | 0.649 |
| 12 | 464 | 193 | 170 | 22 | 271 | — | 0.584 |
| 13 | 531 | 216 | 194 | 22 | 315 | — | 0.592 |
| 14 | 585 | 193 | 171 | 21 | 392 | — | 0.670 |
| 15 | 584 | 231 | 199 | 31 | 353 | 37.5 | 0.605 |
| 16 | 443 | 186 | 164 | 22 | 257 | — | 0.580 |
| 17 | 455 | 191 | 169 | 20 | 265 | — | 0.581 |
| 18 | 505 | 192 | 170 | 21 | 313 | — | 0.620 |
| 19 | 574 | 207 | 181 | 24 | 368 | — | 0.640 |

**Steady-state means** (steps 1-19, excluding warmup): step_time ≈ 512s,
train_time ≈ 207s, actor_train ≈ 181s, log_probs ≈ 23s, train_wait ≈ 307s,
wait_ratio ≈ 0.60. Four weight updates at steps 0/5/10/15 took 34.3/35.7/33.7/37.5s
— consistent and predictable.

### 2. Generator per-sample breakdown (sec, mean across 64 samples)

| step | inf_mean | inf_p95 | sb_mean | sb_p95 | sb_frac | roll_t |
|-----:|---------:|--------:|--------:|-------:|--------:|-------:|
| 0 | 350 | 529 | 20.9 | 64.0 | 0.056 | 640 |
| 1 | 344 | 538 | 19.0 | 49.7 | 0.052 | 571 |
| 2 | 374 | 568 | 20.5 | 70.8 | 0.052 | 546 |
| 3 | 327 | 492 | 20.0 | 53.8 | 0.058 | 541 |
| 4 | 327 | 439 | 20.0 | 50.1 | 0.058 | 449 |
| 5 | 315 | 425 | 17.4 | 37.5 | 0.052 | 514 |
| 6 | 354 | 493 | 28.2 | 81.9 | 0.074 | 509 |
| 7 | 391 | 738 | 14.9 | 39.4 | 0.037 | 452 |
| 8 | 315 | 493 | 22.2 | 85.6 | 0.066 | 469 |
| 9 | 340 | 511 | 19.9 | 46.2 | 0.055 | 562 |
| 10 | 356 | 518 | 19.3 | 61.5 | 0.051 | 461 |
| 11 | 371 | 633 | 18.3 | 47.8 | 0.047 | 588 |
| 12 | 320 | 513 | 28.0 | 85.4 | 0.080 | 482 |
| 13 | 341 | 520 | 17.2 | 61.7 | 0.048 | 507 |
| 14 | 341 | 526 | 21.4 | 73.6 | 0.059 | 608 |
| 15 | 341 | 506 | 23.9 | 105.6 | 0.066 | 509 |
| 16 | 337 | 507 | 17.9 | 74.9 | 0.051 | 488 |
| 17 | 312 | 520 | 19.3 | 58.5 | 0.058 | 451 |
| 18 | 330 | 543 | 22.5 | 77.2 | 0.064 | 503 |
| 19 | 339 | 502 | 24.0 | 77.9 | 0.066 | 560 |

**`inf_mean ≈ 340s, sb_mean ≈ 21s, sb_frac ≈ 0.06`** — inference dominates
sandbox time by ~16×. Optimizing the Modal sandbox round-trip would not
move the needle on rollout wall time. The single 738s `inf_p95` at step 7
was a straggler sample (one trajectory took ~12 min of pure model
inference) — not catastrophic.

### 3. Throughput (tokens/GPU/sec)

| step | train_tok/s | train_tflops | gen_tok/s | gen_eff/s | longest/s |
|-----:|------------:|-------------:|----------:|----------:|----------:|
| 0 | 5797 | 47.8 | 35.10 | 20.76 | 54.28 |
| 1 | 7572 | 59.1 | 35.77 | 25.13 | 55.67 |
| 2 | 7260 | 61.4 | 43.62 | 28.60 | 62.40 |
| 3 | 7513 | 58.5 | 37.47 | 24.85 | 58.46 |
| 4 | 7606 | 60.5 | 47.54 | 29.41 | 74.69 |
| 5 | 7550 | 58.6 | 39.96 | 25.34 | 60.62 |
| 6 | 8228 | 65.0 | 41.26 | 26.69 | 64.79 |
| 7 | 7564 | 59.3 | 44.97 | 32.88 | 73.53 |
| 8 | 8228 | 67.0 | 46.96 | 27.82 | 79.46 |
| 9 | 8273 | 67.2 | 39.54 | 25.01 | 60.40 |
| 10 | 8025 | 63.0 | 44.85 | 31.37 | 69.54 |
| 11 | 7341 | 57.2 | 34.97 | 22.37 | 57.22 |
| 12 | 7911 | 61.4 | 41.43 | 25.70 | 68.12 |
| 13 | 7276 | 57.5 | 41.05 | 27.75 | 65.56 |
| 14 | 8023 | 63.0 | 33.69 | 21.41 | 54.39 |
| 15 | 7398 | 60.2 | 42.99 | 27.63 | 65.62 |
| 16 | 8336 | 65.2 | 41.69 | 25.24 | 66.14 |
| 17 | 8004 | 62.3 | 44.24 | 28.65 | 71.57 |
| 18 | 8531 | 68.4 | 42.92 | 25.78 | 65.63 |
| 19 | 7784 | 62.1 | 37.37 | 25.23 | 63.57 |

**Trainer**: steady ~7800 tok/gpu/s, ~62 TFLOPs (MFU ~6.3%).
**Generator**: ~40 tok/gpu/s (mean), longest-sample ~65 tok/gpu/s — well
under SGLang's peak because rollout wall time includes Modal sandbox
roundtrips (the agent loop's tool execution between generations).

### 4. Rollout length stats (tokens)

| step | resp_mean | resp_max | resp_min | seq_mean | turns | tool_calls |
|-----:|----------:|---------:|---------:|---------:|------:|-----------:|
| 0 | 13284 | 24649 | 5700 | 23540 | 15.6 | 15.4 |
| 1 | 14346 | 26293 | 5107 | 21471 | 13.6 | 13.5 |
| 2 | 15628 | 24966 | 6991 | 25028 | 12.0 | 11.7 |
| 3 | 13440 | 23159 | 2397 | 21446 | 13.1 | 13.0 |
| 4 | 13204 | 25562 | 4961 | 22355 | 13.8 | 13.6 |
| 5 | 13012 | 19109 | 5713 | 21624 | 14.6 | 14.4 |
| 6 | 13576 | 19920 | 4592 | 22175 | 12.9 | 12.6 |
| 7 | 14846 | 26137 | 3933 | 21372 | 9.9 | 9.6 |
| 8 | 13048 | 28595 | 4855 | 23084 | 13.3 | 13.1 |
| 9 | 14053 | 22304 | 5187 | 23222 | 14.2 | 13.9 |
| 10 | 14455 | 23410 | 5953 | 21858 | 12.5 | 12.0 |
| 11 | 13163 | 22920 | 5507 | 21612 | 14.4 | 14.1 |
| 12 | 12377 | 21361 | 4112 | 21060 | 14.1 | 13.8 |
| 13 | 14082 | 25147 | 6519 | 22006 | 12.6 | 12.4 |
| 14 | 13023 | 24697 | 6499 | 21476 | 14.2 | 14.0 |
| 15 | 14050 | 22399 | 3170 | 23051 | 13.4 | 13.1 |
| 16 | 12303 | 22308 | 4511 | 21347 | 13.5 | 13.2 |
| 17 | 12925 | 22393 | 6565 | 21186 | 11.4 | 11.2 |
| 18 | 12974 | 24561 | 4975 | 22624 | 13.7 | 13.6 |
| 19 | 14116 | 22817 | 6175 | 22054 | 11.4 | 11.2 |

**resp_mean ≈ 13.6K, seq_mean ≈ 22K** — well under the 32K trainer
context cap. Turn count stable at 12-15 — model isn't blowing through the
50-turn budget. Tool calls ≈ turns - 1 (every turn except the last makes
a tool call; the last either calls `finish` or runs out of budget).

### 5. Task performance

| step | resolve | complete | trunc | zero_std/8 | raw_rwd |
|-----:|--------:|---------:|------:|-----------:|--------:|
| 0 | 18.8% | 84.4% | 15.6% | 6/8 | 0.1875 |
| 1 | 18.8% | 95.3% | 4.7% | 6/8 | 0.1875 |
| 2 | 0.0% | 75.0% | 25.0% | 8/8 | 0.0000 |
| 3 | 15.6% | 95.3% | 4.7% | 6/8 | 0.1562 |
| 4 | 14.1% | 89.1% | 10.9% | 6/8 | 0.1406 |
| 5 | 23.4% | 92.2% | 7.8% | 6/8 | 0.2344 |
| 6 | 10.9% | 92.2% | 7.8% | 7/8 | 0.1094 |
| 7 | 18.8% | 81.2% | 18.8% | 6/8 | 0.1875 |
| 8 | 15.6% | 82.8% | 17.2% | 6/8 | 0.1562 |
| 9 | 21.9% | 84.4% | 15.6% | 6/8 | 0.2188 |
| 10 | 7.8% | 92.2% | 7.8% | 7/8 | 0.0781 |
| 11 | 20.3% | 89.1% | 10.9% | 6/8 | 0.2031 |
| 12 | 17.2% | 95.3% | 4.7% | 6/8 | 0.1719 |
| 13 | 0.0% | 90.6% | 9.4% | 8/8 | 0.0000 |
| 14 | 20.3% | 84.4% | 15.6% | 5/8 | 0.2031 |
| 15 | 14.1% | 85.9% | 14.1% | 6/8 | 0.1406 |
| 16 | 14.1% | 90.6% | 9.4% | 6/8 | 0.1406 |
| 17 | 9.4% | 90.6% | 9.4% | 7/8 | 0.0938 |
| 18 | 21.9% | 81.2% | 18.8% | 6/8 | 0.2188 |
| 19 | 6.2% | 90.6% | 9.4% | 7/8 | 0.0625 |

**20-step mean resolve: 14.6%, range 0-23.4%**. Two zero-resolve batches
(steps 2 and 13) — likely a hard-prompt set landed in those rollouts.
`zero_std/8 = 5-8` every step: **only 0-2 of 8 groups contributed
gradient signal per step**. Truncation rate 5-25%.

### 6. Reward fail reasons (fraction of all 64 samples)

| step | no_patch | mp_fail | pytest_fail |
|-----:|---------:|--------:|------------:|
| 0 | 35.9% | 1.6% | 43.8% |
| 1 | 17.2% | 37.5% | 26.6% |
| 2 | 21.9% | 25.0% | 53.1% |
| 3 | 31.2% | 25.0% | 28.1% |
| 4 | 25.0% | 12.5% | 48.4% |
| 5 | 25.0% | 12.5% | 39.1% |
| 6 | 23.4% | 25.0% | 40.6% |
| 7 | 32.8% | 25.0% | 23.4% |
| 8 | 26.6% | 12.5% | 45.3% |
| 9 | 25.0% | 12.5% | 40.6% |
| 10 | 15.6% | 37.5% | 39.1% |
| 11 | 20.3% | 25.0% | 34.4% |
| 12 | 32.8% | 12.5% | 37.5% |
| 13 | 31.2% | 25.0% | 43.8% |
| 14 | 17.2% | 12.5% | 50.0% |
| 15 | 20.3% | 25.0% | 40.6% |
| 16 | 23.4% | 0.0% | 62.5% |
| 17 | 26.6% | 37.5% | 26.6% |
| 18 | 32.8% | 12.5% | 32.8% |
| 19 | 34.4% | 25.0% | 34.4% |

**20-step means**: no_patch ≈ 26%, mp_fail ≈ 20%, pytest_fail ≈ 39%.
The model is producing patches ~74% of the time, but ~50% of produced
patches don't pass pytest. The dominant failure mode is "produces a fix
that's wrong" rather than "produces no fix" or "produces malformed diff."

### 7. Training signal

| step | loss | entropy | grad_norm | rollout_lp_diff |
|-----:|---------:|--------:|----------:|----------------:|
| 0 | +4.6e-05 | 0.836 | 0.099 | 0.1055 |
| 1 | -4.5e-05 | 0.771 | 0.143 | 0.1038 |
| 2 | +0.0000 | 0.854 | 0.000 | 0.1127 |
| 3 | +9.1e-05 | 0.804 | 0.103 | 0.1060 |
| 4 | -2.7e-04 | 0.820 | 0.154 | 0.1044 |
| 5 | +9.4e-05 | 0.809 | 0.094 | 0.1060 |
| 6 | +9.7e-05 | 0.820 | 0.095 | 0.1063 |
| 7 | -2.2e-05 | 0.814 | 0.103 | 0.1066 |
| 8 | -9.3e-05 | 0.814 | 0.175 | 0.1059 |
| 9 | +2.3e-05 | 0.805 | 0.151 | 0.1044 |
| 10 | +8.6e-05 | 0.800 | 0.102 | 0.1064 |
| 11 | -1.3e-04 | 0.789 | 0.170 | 0.1036 |
| 12 | +1.0e-04 | 0.807 | 0.139 | 0.1051 |
| 13 | +0.0000 | 0.824 | 0.000 | 0.1036 |
| 14 | +1.4e-04 | 0.806 | 0.184 | 0.1036 |
| 15 | -1.3e-04 | 0.833 | 0.079 | 0.1078 |
| 16 | +8.2e-05 | 0.805 | 0.159 | 0.1005 |
| 17 | -7.4e-06 | 0.826 | 0.116 | 0.1053 |
| 18 | +2.8e-05 | 0.808 | 0.123 | 0.1049 |
| 19 | -1.1e-06 | 0.836 | 0.106 | 0.1050 |

- `loss == pg_loss` everywhere (no KL or entropy loss applied — both coefs are 0)
- `entropy_loss` stable ~0.81 — model isn't collapsing or saturating
- `grad_norm` bounded 0-0.18 — no explosions; zero at steps 2 & 13 where
  all groups had zero variance (no gradient signal that step)
- `rollout_lp_diff` ≈ 0.105 throughout — consistent bf16↔FP8 quantization
  gap between trainer-side and rollout-side logprobs
- `pg_clipfrac = 0` throughout — small updates well within `eps_clip=4e-4`,
  no PPO clipping fires
- `tis_clipfrac` ≈ 0.006 — almost no TIS clipping, off-policy ratio stays
  well-behaved despite the async pipeline

### 8. Async / staleness

| step | resume_mean | n_resumed | stale_mean | stale_p95 |
|-----:|------------:|----------:|-----------:|----------:|
| 0 | 1.00 | 0 | 0.00 | 0.00 |
| 1 | 1.00 | 0 | 0.00 | 0.00 |
| 2 | 1.00 | 0 | 1.00 | 1.00 |
| 3 | 1.00 | 0 | 2.00 | 2.00 |
| 4 | 1.00 | 0 | 3.00 | 3.00 |
| 5 | 1.00 | 0 | 4.00 | 4.00 |
| 6 | 1.09 | 6 | 4.25 | 5.00 |
| 7 | 1.45 | 29 | 5.42 | 6.00 |
| 8 | 1.00 | 0 | 6.00 | 6.00 |
| 9 | 1.00 | 0 | 7.00 | 7.00 |
| 10 | 1.00 | 0 | 8.00 | 8.00 |
| 11 | 1.30 | 19 | 8.38 | 9.00 |
| 12 | 1.31 | 20 | 9.20 | 10.00 |
| 13 | 1.00 | 0 | 10.00 | 10.00 |
| 14 | 1.00 | 0 | 11.00 | 11.00 |
| 15 | 1.00 | 0 | 12.00 | 12.00 |
| 16 | 1.25 | 16 | 12.61 | 13.00 |
| 17 | 1.19 | 12 | 13.14 | 14.00 |
| 18 | 1.00 | 0 | 14.00 | 14.00 |
| 19 | 1.00 | 0 | 15.00 | 15.00 |

⚠️ **Caveat — these numbers use a buggy metric definition that was fixed
post-run.** The old formula was `rollout_id - sample.weight_versions[0]`
where `weight_version` ticks per *update_weights call* (4 events total)
and `rollout_id` ticks per *rollout* (20 events) — different scales. The
monotonic growth to 15 is a unit mismatch, not an actual sample age.

The new metric (in 36842 onwards) uses
`rollout_id - sample.metadata["start_rollout_id"]`, where both terms
count in rollouts. Expected values are ~0-5 (bounded by update interval),
which actually reflects sample age.

`resume_mean` and `n_resumed` are correct as-shown: resumes cluster
right after weight broadcasts (steps 6, 7, 11, 12, 16, 17 — all near
the steps where updates fired). Peak ~29 of 64 samples (45%) resumed in
a single rollout — within normal partial-rollout behavior, no runaway
cycling (vs exp 09's 400-700 resume-cycles-per-sample pathology).

## Interpretation

### What this run validated (infra)

1. **PP=2 CP=4 is stable across multiple weight updates.** Three jobs on
   PP=4 CP=2 (36556/36557/36615) hit the gloo-recv-timeout crash class at
   2-7 hours, all near a weight-broadcast event. Switching to the
   reference's PP=2 CP=4 split survived all 4 broadcasts cleanly. The
   user's prior conclusion from exp 15 — *"Trainer reshape is SLOWER
   than 13 AND less reliable. 13's config remains the best"* — held up
   under our independent test.

2. **The full custom-hook pipeline works end-to-end.**
   `generate_with_openhands.generate` (the agent loop), `reward.compute_reward`
   (fresh-sandbox pytest eval), and `metrics.log_rollout_data` (our 40
   swebench/* keys + slime native metrics) all functioned correctly across
   20 rollouts with no errors. All 100 expected JSONL records were emitted
   (5 per rollout: swebench, rollout native, rollout perf, train, perf-summary).

3. **Modal sandbox lifecycle is healthy.** `n_resumed` capped at 29 of 64
   per rollout (no runaway cycling). Zero sandbox-failure events
   (`fail_reasons/sandbox/*` columns were all 0% throughout). No leaked
   sandboxes that bypassed generate.py's finally blocks until job exit
   (17 orphans at end — normal teardown race, cleaned manually).

4. **Throughput is comparable to the reference**: ~62 TFLOPs MFU 6.3%
   matches exp 13's ~62 TFLOPs / MFU 6.2%. The trainer reshape doesn't
   regress per-GPU efficiency — it just trades for reliability.

### What this run did NOT validate (learning)

1. **Reward signal is flat.** Mean 14.6% resolve rate, range 0-23.4%, no
   trend. This is expected given the configuration:
   - 10-prompt pool (each prompt sampled ~16 times across the 20 steps)
     → high prompt reuse, model overfits to prompt-specific quirks
   - 8-sample groups + binary 0/1 reward → with ~15% solve rate, expected
     fraction of zero-std groups is ~27% in theory, but we observe 75-100%.
     The gap means **per-prompt solve rates are bimodal** (some prompts
     never solved, some always solved), so most groups end up zero-std
     even when overall solve rate is non-trivial.
   - 20 rollouts × ~2 informative groups/step = ~40 actually-useful
     gradient samples total — far below what 235B RL needs to move the
     needle.

2. **Bottleneck is rollout, not training.** `wait_ratio ≈ 0.60` means
   trainer sits idle ~60% of step time waiting for rollouts. This is
   consistent across all 20 steps. To improve throughput, the lever is
   "increase rollouts per step" (bigger global batch) or "speed up agent
   rollouts" (which means speeding up model inference — sandbox is only
   ~6% of time per the `sb_frac` column).

3. **Inference dominates per-sample time by 16×.** Optimizing the Modal
   sandbox path (parallel tool calls, better caching, etc.) would not
   meaningfully change throughput. The bottleneck is SGLang generation
   time on long multi-turn agent trajectories (~14 turns × ~1.5K
   response tokens per turn).

### Bugs surfaced

1. **Staleness metric had a unit mismatch.** Used `weight_versions[0]`
   (slime's SGLang weight version counter that only ticks at
   `update_weights` events) instead of `start_rollout_id`
   (slime's per-sample rollout-id stamp). Fixed for 36842 onwards.

2. **17 orphan Modal sandboxes leaked at job teardown** even on clean
   completion. The generate.py finally blocks call `close()` correctly,
   but the rollout/trainer race tears down the actor before all
   sandboxes finish termination. Now standing-instruction: run
   `cleanup_sandboxes.py` after ANY job end, not just scancel.

3. **`save_model` would have OOMed** (~1.4 TB CPU gather of sharded
   weights for 235B FP8 → bf16 dequant). Mitigated by omitting both
   `--save` and `--save-interval` (slime's `should_run_periodic_action`
   hardcodes "save at last rollout" if save_interval is set — must omit
   both flags to fully disable).

### What we'd change for the next run (informs 36842)

- **n_samples_per_prompt 8 → 16** + **global_batch 64 → 128**: at ~15%
  solve rate, 16-sample groups have P(mixed outcome) ≈ 92% vs 73% with 8.
  Cuts zero-std groups dramatically → more gradient per step.
- **Prompt pool 10 (Verified) → 40 (Lite)**: matches reference exp's
  proven pool. Each prompt used ~4 times across 20 steps (vs current ~16).
- **Per-group max staleness + spread metrics**: added so the next runs
  can detect intra-group inconsistency that affects GSPO advantage
  normalization.
- **Corrected staleness formula**: use `start_rollout_id` directly so the
  numbers actually mean rollouts-since-dispatch.

These changes were applied to job 36842 (running). 36728 + 36842 together
test whether the bigger group + more prompts produce visible learning
signal at the same 20-step budget.

## Files preserved

- `mnt/logs/infx-swebench-new-36728.out` — full sbatch stdout
- `mnt/logs/infx-swebench-new-36728.err` — sbatch stderr
- `mnt/logs/infx-swebench-new-36728-head.out` — Ray head log
- `mnt/logs/infx-swebench-new-36728-worker-*.out` — per-worker Ray logs
- `mnt/logs/metrics-36728.jsonl` — 100 records, ~67 KB, the source of all
  the tables above
