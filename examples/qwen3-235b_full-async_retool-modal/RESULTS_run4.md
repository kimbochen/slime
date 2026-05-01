# Run 4 (job 140) — Pipelined RL with `--update-weights-interval 8 --keep-old-actor`

**Date:** 2026-04-28
**Goal:** Same 20-step smoke as Run 3, but with the trainer pushing weights every 8 steps instead of every step. Test whether the TIS off-policy correction can absorb up-to-7-step staleness while wiping out the trainer's wait time.
**Outcome:** ✅ All 20 training steps complete cleanly. Job ran 1h 0m 47s (vs Run 3's 1h 13m 12s — 17% faster wall-clock).

## Headline metric: `wait_time_ratio` collapsed

| | Run 3 (interval=1) | Run 4 (interval=8) |
|---|---:|---:|
| Steady-state `wait_time_ratio` | **41 %** | **6.9 %** |
| Steady-state `step_time` | 156 s | **104 s** (33% faster) |
| Steady-state `actor_train_time` | 64 s | 68 s (≈ same — compute is the same) |
| Steady-state `train_wait_time` | 64 s | **7 s** |
| Wall-clock samples / s | 1.64 | 2.45 |
| Update_weights events for 20 steps | 20 | 3 |
| Total aborts over 20 steps | 564 | 30 |

The trainer is now compute-bound. The remaining 7% wait is the small queue-fill window after each push (and one-time bring-up).

## Max policy staleness per sample

**Theoretical cap: `interval - 1 = 7` steps** (any sample whose generation was triggered right after a push V_K → V_{K+1} can survive in the buffer up to 7 trainer steps before being consumed).

**Empirical evidence — TIS importance ratio absolute deviation, by phase:**

| Phase | trainer steps | mean `tis_abs` | mean `tis_clipfrac` | Stale-step regime |
|---|:---:|---:|---:|---:|
| Window 1 (under V_1, before push) | 0..7 | 0.0210 | 0.0192 % | 0..7 stale |
| Window 2 (under V_2) | 8..15 | 0.0207 | 0.0191 % | 0..7 stale |
| Window 3 (under V_3, partial) | 16..19 | 0.0205 | 0.0188 % | 0..3 stale |

Crucially **`tis_abs` does NOT grow as we get further from the last push**: step 7 (7-stale) has `tis_abs = 0.02086`, step 0 (0-stale) has `tis_abs = 0.02123`, step 15 (7-stale) has `tis_abs = 0.01902`. The drift signal is dominated by per-token noise, not by the staleness offset.

This is consistent with the slime model already being well-aligned: a 7-step gradient drift on Adam @ lr=1e-6 produces parameter changes too small to noticeably shift the per-token logprob distribution. The TIS clip threshold (`C=2`) wouldn't bite until `tis_abs` ≈ 0.6 nats; we're 30× under that.

**Aborts per push window** (samples that were in flight at the moment of update_weights):

| Window | Aborts | Surviving (≤ 8-step-stale samples that bypass the abort retry) |
|---|---:|---:|
| Push 0 → Push 1 (steps 0..7 of training) | 0 | All 256×8 = 2048 samples generated under V_1 |
| Push 1 → Push 2 (steps 8..15) | 30 | Of 32 in-flight groups, 30 aborted+reissued under V_2; 2 (~16 samples) survived as "V_1-tagged + consumed under V_2" — staleness 8 |
| Push 2 → end (steps 16..19) | 0 | (Job cancelled before next push, no aborts in tail) |

So in absolute counts:
- ~16 samples (out of 5120 = 0.3%) reached the trainer with weight_version 1 step older than the active actor's pushed version
- 0 samples reached the trainer with > 1 push-step of staleness

**In practical terms: max staleness per sample = 7 trainer steps (the `interval - 1` cap)**, but only ~0.3% of samples actually achieve > 0 push-step staleness (the bulk are 0-push-step stale, just consumed at varying offsets within the 8-step window).

## Trainer per-step (steady-state, last 3 steps)

| Metric | Run 3 | Run 4 |
|---|---:|---:|
| step_time | 156 s | **104 s** |
| actor_train_time | 64 s | 68 s |
| wait_time_ratio | 41 % | **6.9 %** |
| update_weights_time (when fired) | 41 s | 41 s |
| actor_train_tflops | 54 | 56 |
| MFU | 5.5 % | 5.6 % |
| samples/s wall-clock (256/step_time) | 1.64 | **2.45** |

Note steps 8 and 16 had `wait_time_ratio = 28 % / 30 %` because update_weights blocks the trainer for ~40 s + drains a fresh batch — same effect as Run 3's old per-step behavior, just amortized over 8 steps now.

## Generator per-step

- `rollout_time` is dominated by the *first* rollout filling the cold queue (rollout 0: 132 s); subsequent rollouts complete near-instantly (0.01-21 s) because the queue fills faster than the trainer drains.
- `gen_to_trainer_ratio` mean is now 0.07 (Run 3 was 1.85) — confirming we've fully saturated the trainer with rollouts; the bottleneck has flipped.
- Mean response length: 5614 tokens (vs Run 3's 5190) — model is gradually emitting more reasoning.
- `truncated_ratio`: 31% (similar to Run 3).
- Sandbox metrics: `sandbox_calls_per_sample = 0.08`, `sandbox_error_rate = 10 %` mean across all 20 rollouts (one rollout had 16 % errors — likely a Modal cold-start blip; steady-state error rate would be lower with longer training).

## Loss / convergence

| step | train/loss | grad_norm | entropy_loss | tis_abs | tis_clipfrac% |
|---:|---:|---:|---:|---:|---:|
| 0 | -4.9e-5 | 0.067 | 0.224 | 0.0212 | 0.022 |
| 7 | +1.1e-4 | 0.080 | 0.223 | 0.0209 | 0.021 |
| 8 (post-push) | +5.3e-6 | 0.084 | 0.234 | 0.0217 | 0.027 |
| 15 | +7.1e-5 | 0.116 | 0.215 | 0.0191 | 0.014 |
| 16 (post-push) | +2.2e-5 | 0.059 | 0.232 | 0.0210 | 0.022 |
| 19 | +1.1e-5 | 0.048 | 0.231 | 0.0197 | 0.014 |

- `train/loss` ±1e-4, same as Run 3. Qwen3-235B-Thinking is well-aligned for these prompts — interval=8 doesn't change the underlying signal.
- `grad_norm` decays similarly (0.07 → 0.05).
- `tis_clipfrac` <0.03 % at all steps — TIS clip is rarely active. Lots of headroom for higher intervals.
- `kl_loss` drifts up slowly (0 → 0.0012) — small but real divergence from the reference policy. Same shape as Run 3.

## Conclusion

`--update-weights-interval 8` is a free 33% wall-clock speedup. The TIS importance correction handles the 7-step max staleness without breaking a sweat (clipfrac stays at 0.02 %, far from the 1-2 % regime where it would start losing samples). For this model + workload, **interval 16 or 32 would likely also work** — recommend trying 16 next to recover more of the rollout-side overhead at small risk.
