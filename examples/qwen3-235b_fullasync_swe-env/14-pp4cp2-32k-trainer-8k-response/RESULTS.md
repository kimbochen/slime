# 32K trainer (PP=4 CP=2) + 8K per-turn response — job 31887

- Log: `mnt/logs/infx-swe-31887.out`
- Steps captured: 10
- Rollouts captured: 11

## Per-step performance

| step | wait (s) | train (s) | step (s) | train_tflops | MFU(in-step) | wallclock_MFU | wait_ratio |
|-----:|---------:|----------:|---------:|-------------:|-------------:|--------------:|-----------:|
|    0 |      568 |       248 |      816 |         33.6 |        3.39% |         0.78% |       0.70 |
|    1 |      148 |       101 |      249 |         66.4 |        6.71% |         2.42% |       0.59 |
|    2 |      248 |       105 |      353 |         78.0 |        7.88% |         2.00% |       0.70 |
|    3 |      402 |        91 |      493 |         75.5 |        7.62% |         1.22% |       0.82 |
|    4 |      207 |        86 |      293 |         77.8 |        7.86% |         2.05% |       0.71 |
|    5 |      353 |        95 |      447 |         77.0 |        7.78% |         1.46% |       0.79 |
|    6 |      380 |        96 |      476 |         68.9 |        6.96% |         1.19% |       0.80 |
|    7 |    15726 |        81 |    15807 |         72.6 |        7.33% |         0.03% |       0.99 |
|    8 |     1765 |       101 |     1866 |         75.3 |        7.61% |         0.36% |       0.95 |
|    9 |      893 |        92 |      985 |         76.0 |        7.67% |         0.61% |       0.91 |

### Steady-state aggregate (step 1+)

- mean train_tflops: **74.2** (MFU 7.49%)
- mean wait_time: 2236s
- mean train_time: 94s
- mean wait_ratio: 0.81

## Per-rollout metrics

| rollout | rt (s) | tok/gpu/s | sandbox_acq (s) | resp_len_mean | resp_len_max | zero_std | tool_calls | policy_staleness_max |
|--------:|-------:|----------:|----------------:|--------------:|-------------:|---------:|-----------:|---------------------:|
|       0 |    532 |      25.2 |             3.8 |         10798 |        17359 |     1/8 |       15.0 |                    0 |
|       1 |    395 |      33.0 |             0.7 |         10566 |        23013 |     2/8 |       16.1 |                    0 |
|       2 |    349 |      38.6 |             0.7 |         10963 |        20235 |     2/8 |       14.5 |                    0 |
|       3 |    507 |      26.0 |             0.8 |         11048 |        21968 |     2/8 |       15.5 |                    0 |
|       4 |    297 |      42.9 |             0.6 |         10619 |        24310 |     2/8 |       13.0 |                    0 |
|       5 |    401 |      34.1 |             0.7 |         10961 |        21751 |     0/8 |       14.5 |                    1 |
|       6 |    475 |      25.9 |             1.7 |         10110 |        20623 |     1/8 |       11.9 |                    0 |
|       7 |  15822 |       0.8 |             0.7 |         10136 |        18126 |     2/8 |       13.2 |                    0 |
|       8 |   1846 |       7.1 |             0.6 |         10785 |        21493 |     1/8 |       16.2 |                    0 |
|       9 |    993 |      13.0 |             0.6 |         10685 |        19174 |     2/8 |       14.2 |                    0 |
|      10 |   1593 |       9.2 |             0.6 |         11332 |        20837 |     2/8 |       18.4 |                    1 |

## Trainer-side reward signal

| rollout | raw_reward | rewards (post-norm) | log_probs | rollout_log_probs | kl |
|--------:|-----------:|--------------------:|----------:|------------------:|---:|
|       0 |      2.66% |             -0.0005 |   -0.7929 |           -0.7695 | +0.0000 |
|       1 |      3.20% |             -0.0009 |   -0.7858 |           -0.7645 | +0.0000 |
|       2 |      3.28% |             -0.0009 |   -0.7298 |           -0.7082 | +0.0000 |
|       3 |      2.81% |             -0.0009 |   -0.7946 |           -0.7709 | +0.0000 |
|       4 |      2.42% |             -0.0009 |   -0.8025 |           -0.7792 | +0.0000 |
|       5 |      2.34% |             -0.0000 |   -0.8039 |           -0.7819 | +0.0000 |
|       6 |      2.27% |             -0.0005 |   -0.8437 |           -0.8213 | +0.0000 |
|       7 |      2.66% |             -0.0009 |   -0.8040 |           -0.7813 | +0.0000 |
|       8 |      3.20% |             -0.0005 |   -0.7277 |           -0.7075 | +0.0000 |
|       9 |      2.89% |             -0.0009 |   -0.7895 |           -0.7671 | +0.0000 |

## Re-launch attempt (job 31887) — extended trajectory, then CRASHED

The original 31816 run was cancelled at 7 perfs after seeing the early
reward drop. To get more data points, this experiment was re-launched as
**31887** with a 12h Slurm wallclock budget. Result: ran for **7h 24m,
10 perfs**, then crashed with the same gloo collective timeout that
killed 09/31011.

### Per-step trajectory

```
              31887            31816 (original)
  p0:  step= 816s  (init)      775s
  p1:  step= 249s               371s
  p2:  step= 353s               282s
  p3:  step= 493s               347s
  p4:  step= 293s               381s
  p5:  step= 447s  upd=38s      353s  upd=35s
  p6:  step= 476s              1372s  ← first run had post-push spike, this one didn't
  p7:  step=15807s   ← CATASTROPHIC, equivalent to 10's p8=14751
  p8:  step= 1866s
  p9:  step=  985s
  p10: (trainer crashed during update_weights — never landed)
```

### Crash signature

Identical class to 09/31011:

```
RuntimeError: [.../gloo/transport/tcp/unbound_buffer.cc:78]
  Timed out waiting 1800000ms for recv operation to complete
ray.exceptions.RayTaskError(RuntimeError): MegatronTrainRayActor.update_weights()
```

The trainer was attempting the second interval-5 weight push (p10) when
gloo's collective recv timed out (30 min). Memory wasn't the issue
(GPU free ~70 GB, host free ~250 GB at crash time) — pure collective
starvation. One rollout-side rank fell so far behind that the trainer's
broadcast couldn't complete.

### What 14 actually told us

1. **The 8K per-turn cap drops reward ~40% (5%→3%)** — confirmed across
   both 31816 and 31887, not a fluke.

2. **The 32K trainer + PP=4/CP=2 reshape gives a real steady-state speedup**
   (~30% faster p1-p4 vs 13). Robust across runs.

3. **The straggler tail problem isn't eliminated by 8K cap — just relocated.**
   31816 saw the spike at p6 (1372s). 31887 saw a much bigger spike at p7
   (15807s). The "smooth, no tail" early reading from 31887's p6=476s
   was misleading.

4. **The catastrophic tail eventually starves the trainer collectives.**
   Same gloo recv timeout class as 09 — when a sample grinds long enough
   that the rollout queue desynchronizes from trainer expectations,
   `update_weights` collectives can't complete, and the run dies.

   ```
                cumulative wallclock when run ended    failure mode
     09/31011:  7h 25m   gloo recv timeout (collective starvation)
     14/31887:  7h 24m   gloo recv timeout (collective starvation — same!)
   ```

   Notable coincidence: the gloo timeout fires at almost exactly the same
   wallclock in both runs (~7h 25m). That suggests the **timeout itself**
   is what's deterministic, not the trajectory. The straggler tail
   eventually accumulates enough collective-wait time that gloo gives up.

5. **JSONL metric capture worked end-to-end** in 31887 — 41 lines, both
   rollout/step and train/step entries, the patch_log_utils path verified
   for the trainer process.

## Final verdict on 14

The 32K trainer + 8K response cap config has **two distinct downsides**:

- **Accuracy:** 8K per-turn cap drops raw_reward by ~40% (5%→3%).
- **Reliability:** the long-context decode-bound straggler tail still
  hits, and when it does, it cascades into trainer crashes via gloo
  collective timeout. Same crash mode as 09/31011.

The steady-state speedup is real (~30%) but doesn't compensate for
either the accuracy regression or the crash risk.

## Next experiment (15) tests the isolated trainer reshape

The right cleanup test is **15** (32K trainer + **16K** response cap),
which uses 14's trainer reshape but reverts the response cap to 13's
value. If 15 recovers ~5% reward at 14's wallclock shape, we have the
new best config. If it ALSO crashes via gloo timeout, then the tail
is fundamental and the trainer reshape doesn't fix anything.
