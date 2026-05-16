# DeepEP auto mode (low_latency) + EP=8 + 16K response — job 31395

- Log: `mnt/logs/infx-swe-31395.out`
- Steps captured: 11
- Rollouts captured: 0

## Per-step performance

| step | wait (s) | train (s) | step (s) | train_tflops | MFU(in-step) | wallclock_MFU | wait_ratio |
|-----:|---------:|----------:|---------:|-------------:|-------------:|--------------:|-----------:|
|    0 |      667 |       235 |      902 |         50.8 |        5.13% |         1.03% |       0.74 |
|    1 |      354 |       161 |      515 |         61.5 |        6.22% |         1.73% |       0.69 |
|    2 |      317 |       175 |      492 |         55.5 |        5.61% |         1.80% |       0.64 |
|    3 |      289 |       160 |      449 |         62.7 |        6.33% |         2.00% |       0.64 |
|    4 |      404 |       155 |      560 |         68.0 |        6.87% |         1.66% |       0.72 |
|    5 |      351 |       164 |      515 |         71.3 |        7.20% |         2.04% |       0.68 |
|    6 |      982 |       171 |     1153 |         67.6 |        6.83% |         0.90% |       0.85 |
|    7 |     3313 |       146 |     3459 |         64.6 |        6.53% |         0.25% |       0.96 |
|    8 |     3102 |       148 |     3250 |         57.5 |        5.81% |         0.23% |       0.95 |
|    9 |     3860 |       160 |     4021 |         61.1 |        6.18% |         0.22% |       0.96 |
|   10 |     4016 |       158 |     4174 |         68.2 |        6.89% |         0.23% |       0.96 |

### Steady-state aggregate (step 1+)

- mean train_tflops: **63.8** (MFU 6.45%)
- mean wait_time: 1699s
- mean train_time: 160s
- mean wait_ratio: 0.81

## Trainer-side reward signal

| rollout | raw_reward | rewards (post-norm) | log_probs | rollout_log_probs | kl |
|--------:|-----------:|--------------------:|----------:|------------------:|---:|
|       0 |      4.38% |             -0.0014 |   -0.7417 |           -0.7191 | +0.0000 |
|       1 |      4.84% |             -0.0028 |   -0.7445 |           -0.7240 | +0.0000 |
|       2 |      5.00% |             -0.0037 |   -0.6737 |           -0.6536 | +0.0000 |
|       3 |      4.53% |             -0.0023 |   -0.7550 |           -0.7329 | +0.0000 |
|       4 |      4.92% |             -0.0033 |   -0.7324 |           -0.7107 | +0.0000 |
|       5 |      4.30% |             -0.0023 |   -0.7688 |           -0.7462 | +0.0000 |
|       6 |      4.45% |             -0.0023 |   -0.7676 |           -0.7465 | +0.0000 |
|       7 |      4.61% |             -0.0028 |   -0.7174 |           -0.6961 | +0.0000 |
|       8 |      5.00% |             -0.0037 |   -0.7261 |           -0.7050 | +0.0000 |
|       9 |      4.45% |             -0.0023 |   -0.7101 |           -0.6889 | +0.0000 |
|      10 |      4.77% |             -0.0023 |   -0.7072 |           -0.6850 | +0.0000 |

## Inference throughput collapse — post-push regime

The headline pattern: **rollout_time exploded from ~500s pre-push to ~3500-4000s post-push** while response lengths, tool-call counts, and per-batch decode throughput stayed roughly constant. The single straggler sample's effective throughput (`longest_tok/s`) dropped from ~60 tok/s pre-push to ~7-12 tok/s post-push — an 8× slowdown specifically on the slowest sample in each rollout.

```
       rt    resp_mean  longest_tok/s   cuda graph hit-rate (per batch)
  r5:  471s   16304       69            49%
  r6: 1145s   14447       34            48%   ← cliff starts at 1st post-push
  r7: 3484s   13798        8.6          48%
  r8: 3248s   12135        8.9          48%
  r9: 4008s   14240        7.2          49%
  r10:4142s   14000+      12.0          48%
```

**Per-batch decode throughput stayed normal** (60-100 tok/s in cuda-graph mode, 22-35 tok/s eager, hit rate stable at 48%). What changed: when the rollout enters the tail and only 1 sample is still running on an engine, DeepEP's `low_latency` mode pays its full RDMA setup cost for a single-token dispatch instead of amortizing across 8 concurrent DP ranks. The 20-SM allocation for DeepEP communication (warned by SGLang at startup) compounds this.

In effect: **DeepEP `low_latency` mode optimizes the buffer-full case at the cost of the buffer-sparse case**. Our agentic-RL workload spends most of each rollout in the sparse tail, so we pay the cost without getting the benefit.

## Verdict

✅ **Validated:** DeepEP genuinely speeds up weight broadcast (`update_weights_time`: 35s vs 10's 43s, ~19% faster) and the first post-push step (p6: 1153s vs 10's 1418s).

❌ **Failed:** the long-tail regime (p7+) doesn't recover — every step takes 3000-4000s as DeepEP `low_latency` mode is gated by single-sample stragglers. Net wallclock for 11 perfs ≈ 19000s ≈ 5h 17m — same per-step pace as 10 after accounting for 10's catastrophic p8.

❌ **No reward signal:** raw_reward stayed at ~0.045-0.05 across all 11 rollouts. Same baseline as 09 / 10.

**Next experiments:**
- **12 (queued):** DeepEP `normal` mode — should avoid the low-latency-mode SM allocation pitfall. Hypothesis: keeps the weight-broadcast win, fixes the straggler tail.
- **13 (prepped, not launched):** plain `alltoall` + EP=8 + 16K — pure isolation of "is EP=8 alone the problem, or is DeepEP itself the problem?"

If 12 also has the tail explosion, the issue isn't DeepEP-the-mode — it's DeepEP-the-backend or EP=8-the-shape. 13's alltoall result will distinguish those.
