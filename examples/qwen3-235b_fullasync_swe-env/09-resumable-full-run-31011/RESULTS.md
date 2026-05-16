# Resumable full run — job 31011

- Log: `/home/sa-shared/kimbo/slime/mnt/logs/infx-swe-rsm-31011.out`
- Steps captured: 15
- Rollouts captured: 16

## Per-step performance

| step | wait (s) | train (s) | step (s) | train_tflops | MFU(in-step) | wallclock_MFU | wait_ratio |
|-----:|---------:|----------:|---------:|-------------:|-------------:|--------------:|-----------:|
|    0 |      667 |       247 |      914 |         54.7 |        5.52% |         1.11% |       0.73 |
|    1 |      231 |       128 |      359 |         67.9 |        6.86% |         2.12% |       0.64 |
|    2 |      266 |       179 |      445 |         55.6 |        5.62% |         2.00% |       0.60 |
|    3 |      434 |       182 |      616 |         63.9 |        6.45% |         1.63% |       0.70 |
|    4 |      285 |       172 |      457 |         63.7 |        6.44% |         2.10% |       0.62 |
|    5 |      477 |       182 |      660 |         73.3 |        7.40% |         1.76% |       0.72 |
|    6 |     1349 |       171 |     1520 |         62.1 |        6.27% |         0.62% |       0.89 |
|    7 |     2493 |       156 |     2648 |         61.4 |        6.20% |         0.32% |       0.94 |
|    8 |     1701 |       193 |     1894 |         77.7 |        7.85% |         0.72% |       0.90 |
|    9 |     1237 |       243 |     1479 |        114.1 |       11.53% |         1.65% |       0.84 |
|   10 |      497 |       263 |      760 |        100.8 |       10.18% |         3.02% |       0.65 |
|   11 |      763 |       207 |      971 |         90.4 |        9.14% |         1.71% |       0.79 |
|   12 |     1075 |       224 |     1299 |         89.9 |        9.08% |         1.30% |       0.83 |
|   13 |     1846 |       143 |     1989 |         71.8 |        7.25% |         0.45% |       0.93 |
|   14 |     4775 |       194 |     4969 |        100.6 |       10.16% |         0.33% |       0.96 |

### Steady-state aggregate (step 1+)

- mean train_tflops: **78.1** (MFU 7.89%)
- mean wait_time: 1245s
- mean train_time: 188s
- mean wait_ratio: 0.79

## Per-rollout metrics

| rollout | rt (s) | tok/gpu/s | sandbox_acq (s) | resp_len_mean | resp_len_max | zero_std | tool_calls | policy_staleness_max |
|--------:|-------:|----------:|----------------:|--------------:|-------------:|---------:|-----------:|---------------------:|
|       0 |    630 |      30.5 |             3.8 |         15159 |        27425 |     8/8 |       24.2 |                    0 |
|       1 |    478 |      33.0 |             0.6 |         12125 |        21670 |     7/8 |       22.0 |                    0 |
|       2 |    393 |      41.8 |             0.6 |         13123 |        22766 |     7/8 |       20.5 |                    0 |
|       3 |    612 |      31.5 |             0.6 |         15264 |        25471 |     7/8 |       22.9 |                    0 |
|       4 |    467 |      39.1 |             0.6 |         15026 |        31549 |     7/8 |       21.9 |                    0 |
|       5 |    613 |      34.4 |             0.6 |         16967 |        30450 |     7/8 |       22.1 |                    1 |
|       6 |   1531 |      12.0 |             1.7 |         14869 |        27627 |     7/8 |       21.5 |                    0 |
|       7 |   2663 |       6.4 |             1.4 |         14076 |        28139 |     7/8 |       21.2 |                    0 |
|       8 |   1856 |      10.9 |             1.1 |         14990 |        43929 |     7/8 |       18.7 |                    0 |
|       9 |   1429 |      21.6 |             0.8 |         22587 |        55283 |     4/8 |       17.6 |                    0 |
|      10 |    704 |      41.3 |             0.7 |         21789 |        48669 |     3/8 |       17.9 |                    1 |
|      11 |   1026 |      23.9 |             1.6 |         18135 |        51658 |     6/8 |       23.1 |                    1 |
|      12 |   1282 |      18.7 |             1.3 |         17203 |        40064 |     5/8 |       19.2 |                    1 |
|      13 |   2070 |       8.5 |             0.8 |         14136 |        23317 |     8/8 |       21.9 |                    0 |
|      14 |   4918 |       4.7 |             0.6 |         16900 |        51360 |     7/8 |       18.8 |                    0 |
|      15 |   2711 |       6.5 |             0.6 |         14343 |        25717 |     6/8 |       21.1 |                    0 |

## Trainer-side reward signal

| rollout | raw_reward | rewards (post-norm) | log_probs | rollout_log_probs | kl |
|--------:|-----------:|--------------------:|----------:|------------------:|---:|
|       0 |      5.00% |             -0.0037 |   -0.7391 |           -0.7157 | +0.0000 |
|       1 |      4.92% |             -0.0033 |   -0.7127 |           -0.6928 | +0.0000 |
|       2 |      4.92% |             -0.0033 |   -0.6984 |           -0.6780 | +0.0000 |
|       3 |      4.92% |             -0.0033 |   -0.7265 |           -0.7048 | +0.0000 |
|       4 |      4.92% |             -0.0033 |   -0.7325 |           -0.7089 | +0.0000 |
|       5 |      4.92% |             -0.0033 |   -0.7649 |           -0.7418 | +0.0000 |
|       6 |      4.92% |             -0.0033 |   -0.7636 |           -0.7422 | +0.0000 |
|       7 |      4.92% |             -0.0033 |   -0.7381 |           -0.7158 | +0.0000 |
|       8 |      4.84% |             -0.0033 |   -0.6520 |           -0.6318 | +0.0000 |
|       9 |      3.98% |             -0.0019 |   -0.6170 |           -0.5980 | +0.0000 |
|      10 |      3.52% |             -0.0014 |   -0.6662 |           -0.6461 | +0.0000 |
|      11 |      4.38% |             -0.0028 |   -0.6058 |           -0.5862 | +0.0000 |
|      12 |      4.69% |             -0.0023 |   -0.7038 |           -0.6827 | +0.0000 |
|      13 |      5.00% |             -0.0037 |   -0.7551 |           -0.7325 | +0.0000 |
|      14 |      4.38% |             -0.0033 |   -0.6451 |           -0.6254 | +0.0000 |

## Park / resume sandbox events

| metric | value |
|---|---:|
| PARKED total | 63 |
| RESUMED total | 63 (all parked sandboxes successfully drained) |
| Returned-aborted-group events | 6772 (≈ 484 group cycles per finalized rollout) |

## Crash signature

```
RuntimeError: [.../gloo/transport/tcp/unbound_buffer.cc:78]
  Timed out waiting 1800000ms for recv operation to complete
```

Step 14 (`p14`) took **4969s (83 min)** — exceeded gloo's 30-minute collective
recv timeout. One trainer rank waited for a peer that was idle (gated on the
slow rollout tail) and gave up. Memory at crash: 31 GB GPU free, 247 GB host
free — not OOM, just collective starvation.

Root cause: the 37 samples PARKED at the first weight push entered an
abort-cycling loop. Each cycle produced ~70 new tokens before SGLang
re-aborted, accumulating 400–700 cycles per sample. As cycling continued,
the rollout-pool tail grew until rollout_time exceeded the gloo timeout.

## Verdict

✅ The metric pipeline is **physically verified** end-to-end: `policy_staleness`,
`resume_count/n_resumed`, sandbox PARKED/RESUMED all line up.

❌ Partial-rollout training dynamics under our settings are pathological:
runaway cycling, no reward signal benefit (raw_reward stays at the same
~5% baseline as the no-partial-rollout control in experiment 10), and the
trainer eventually crashes via collective timeout.

The pathology lives in slime's partial-rollout + SGLang abort interaction,
not in our example code. See experiment 10 for the no-partial-rollout
comparison.
