# Plain alltoall + EP=8 + 16K response — job 31508

- Log: `mnt/logs/infx-swe-31508.out`
- Steps captured: 9
- Rollouts captured: 9

## Per-step performance

| step | wait (s) | train (s) | step (s) | train_tflops | MFU(in-step) | wallclock_MFU | wait_ratio |
|-----:|---------:|----------:|---------:|-------------:|-------------:|--------------:|-----------:|
|    0 |      660 |       298 |      958 |         41.7 |        4.22% |         1.08% |       0.69 |
|    1 |      167 |       156 |      323 |         58.2 |        5.88% |         2.51% |       0.52 |
|    2 |      300 |       158 |      458 |         59.2 |        5.98% |         1.82% |       0.66 |
|    3 |      492 |       188 |      680 |         61.2 |        6.18% |         1.53% |       0.72 |
|    4 |      217 |       180 |      398 |         56.3 |        5.69% |         2.29% |       0.55 |
|    5 |      417 |       182 |      599 |         67.4 |        6.80% |         1.74% |       0.70 |
|    6 |     1719 |       179 |     1898 |         61.8 |        6.24% |         0.53% |       0.91 |
|    7 |     3450 |       178 |     3627 |         53.2 |        5.37% |         0.24% |       0.95 |
|    8 |     2952 |       170 |     3122 |         54.3 |        5.49% |         0.25% |       0.95 |

### Steady-state aggregate (step 1+)

- mean train_tflops: **58.9** (MFU 5.95%)
- mean wait_time: 1214s
- mean train_time: 174s
- mean wait_ratio: 0.74

## Per-rollout metrics

| rollout | rt (s) | tok/gpu/s | sandbox_acq (s) | resp_len_mean | resp_len_max | zero_std | tool_calls | policy_staleness_max |
|--------:|-------:|----------:|----------------:|--------------:|-------------:|---------:|-----------:|---------------------:|
|       0 |    624 |      31.5 |             3.7 |         15210 |        22916 |     4/8 |       24.2 |                    0 |
|       1 |    465 |      35.9 |             0.6 |         13261 |        25106 |     6/8 |       20.3 |                    0 |
|       2 |    456 |      34.1 |             0.6 |         12095 |        22492 |     7/8 |       21.9 |                    0 |
|       3 |    649 |      30.5 |             0.6 |         16970 |        27854 |     2/8 |       19.6 |                    0 |
|       4 |    405 |      43.9 |             0.6 |         14045 |        30995 |     4/8 |       24.2 |                    0 |
|       5 |    560 |      35.3 |             0.7 |         16208 |        26501 |     4/8 |       22.7 |                    1 |
|       6 |   1901 |      10.2 |             0.8 |         15352 |        25059 |     7/8 |       22.3 |                    0 |
|       7 |   3628 |       4.7 |             0.6 |         14023 |        28102 |     5/8 |       20.7 |                    0 |
|       8 |   3129 |       4.7 |             0.6 |         11910 |        27515 |     6/8 |       19.8 |                    0 |

## Trainer-side reward signal

| rollout | raw_reward | rewards (post-norm) | log_probs | rollout_log_probs | kl |
|--------:|-----------:|--------------------:|----------:|------------------:|---:|
|       0 |      4.22% |             -0.0019 |   -0.7419 |           -0.7194 | +0.0000 |
|       1 |      4.84% |             -0.0028 |   -0.7248 |           -0.7039 | +0.0000 |
|       2 |      4.84% |             -0.0033 |   -0.6689 |           -0.6495 | +0.0000 |
|       3 |      4.38% |             -0.0009 |   -0.7783 |           -0.7555 | +0.0000 |
|       4 |      4.61% |             -0.0019 |   -0.6941 |           -0.6730 | +0.0000 |
|       5 |      3.98% |             -0.0019 |   -0.7654 |           -0.7431 | +0.0000 |
|       6 |      4.84% |             -0.0033 |   -0.7542 |           -0.7339 | +0.0000 |
|       7 |      4.53% |             -0.0023 |   -0.7279 |           -0.7058 | +0.0000 |
|       8 |      4.84% |             -0.0028 |   -0.6666 |           -0.6469 | +0.0000 |

## Cross-experiment comparison

The point of running 13 was to **isolate whether the straggler-tail collapse
seen in 11 was DeepEP-specific or shape-driven**. The answer is now clear:

```
              p1-p4 mean   p5 (push)   p6 (post-push)   p7-p8 (tail)
  10 (alltoall, EP=4, 32K):  499s     625s              1418s            4387+14751=19138s
  11 (deepep,   EP=8, 16K):  504s     515s              1153s            3459+ 3250= 6709s
  13 (alltoall, EP=8, 16K):  465s     599s              1898s            3627+ 3122= 6749s   ← 13
```

Critical observations:

1. **The tail (p7-p8) is essentially identical between 11 and 13** —
   13's tail steps sum to 6749s vs 11's 6709s, a difference of <1%.
   Despite using completely different SGLang MoE backends (alltoall vs
   DeepEP), the per-step wallclock in the long-context decode regime
   matches almost exactly. So the tail is NOT DeepEP-specific.

2. **The biggest variance reducer is `--rollout-max-response-len`**
   (32K → 16K). 10 had one 4-hour catastrophic step at p8; both 11 and
   13 (with 16K cap) capped the worst step at ~3.5K seconds. That's the
   single most important flag in the matrix.

3. **EP=8 vs EP=4 is marginally helpful on weight push** — 13's
   upd_weights (37s) is closer to 11's deepep (35s) than to 10's
   alltoall + EP=4 (43s). So most of DeepEP's weight-broadcast advantage
   was actually from the EP=8 sharding, not from DeepEP itself.

4. **EP=8 + alltoall is worst at post-push (p6=1898s)** — DeepEP's
   NVSHMEM-RDMA path does help in the immediate post-push step (11's
   p6=1153s), but the win is lost again by p7+ where everything converges.

## Verdict

The MoE backend / EP-size choice doesn't materially change the headline:
**per-rollout wallclock is gated by the agentic-RL straggler at long
context**, regardless of dispatch backend.

The configuration matrix collapses to:

| variable | impact |
|---|---|
| `--rollout-max-response-len 16K` (vs 32K) | huge — prevents catastrophic single-step grinds |
| `--update-weights-interval 5` | moderate — amortizes broadcast cost ~2×/wallclock |
| EP=8 (vs EP=4) | marginal — small win on weight broadcast |
| DeepEP `low_latency` (vs alltoall) | wash — small wins at p5/p6, no benefit at p7+ |
| DeepEP `normal` mode | broken — disables cuda graphs, ~5× slower |

**No reward signal in any of 09-13** (raw_reward stuck at ~5% across all
of them). The straggler problem isn't an obstacle to learning per se —
it's just slowing iteration. To see actual reward movement we'd need
either reward shaping, a smaller / faster model, or more curated easy
instances.

For practical use of this scaffold, the cleanest config is:

```
--rollout-max-response-len 16384    # cap response per turn
--update-weights-interval 5         # amortize broadcasts
--sglang-ep-size 8                  # marginal win
(default alltoall, no --sglang-moe-a2a-backend deepep)
```

i.e. essentially **experiment 13's setup**. DeepEP isn't worth the
config complexity in our regime.
