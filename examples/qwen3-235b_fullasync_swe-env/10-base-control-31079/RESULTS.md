# Base SWE-env (no partial-rollout) — job 31079

- Log: `/home/sa-shared/kimbo/slime/mnt/logs/infx-swe-31079.out`
- Steps captured: 10
- Rollouts captured: 10

## Per-step performance

| step | wait (s) | train (s) | step (s) | train_tflops | MFU(in-step) | wallclock_MFU | wait_ratio |
|-----:|---------:|----------:|---------:|-------------:|-------------:|--------------:|-----------:|
|    0 |      705 |       238 |      943 |         58.8 |        5.94% |         1.15% |       0.75 |
|    1 |      330 |       144 |      474 |         67.5 |        6.82% |         1.82% |       0.70 |
|    2 |      306 |       137 |      443 |         64.0 |        6.46% |         1.77% |       0.69 |
|    3 |      329 |       153 |      482 |         76.4 |        7.72% |         2.09% |       0.68 |
|    4 |      441 |       157 |      598 |         67.1 |        6.78% |         1.56% |       0.74 |
|    5 |      435 |       190 |      625 |         73.4 |        7.42% |         1.94% |       0.70 |
|    6 |     1247 |       171 |     1418 |         59.2 |        5.98% |         0.65% |       0.88 |
|    7 |     4223 |       164 |     4387 |         64.1 |        6.47% |         0.21% |       0.96 |
|    8 |    14590 |       161 |    14751 |         61.9 |        6.25% |         0.06% |       0.99 |
|    9 |     1675 |       171 |     1846 |         59.5 |        6.01% |         0.50% |       0.91 |

### Steady-state aggregate (step 1+)

- mean train_tflops: **65.9** (MFU 6.66%)
- mean wait_time: 2620s
- mean train_time: 161s
- mean wait_ratio: 0.80

## Per-rollout metrics

| rollout | rt (s) | tok/gpu/s | sandbox_acq (s) | resp_len_mean | resp_len_max | zero_std | tool_calls | policy_staleness_max |
|--------:|-------:|----------:|----------------:|--------------:|-------------:|---------:|-----------:|---------------------:|
|       0 |    670 |      30.0 |             3.7 |         16068 |        24370 |     6/8 |       23.9 |                    0 |
|       1 |    568 |      30.6 |             0.6 |         13600 |        19804 |     7/8 |       22.1 |                    0 |
|       2 |    450 |      33.3 |             0.6 |         11902 |        20693 |     8/8 |       19.5 |                    0 |
|       3 |    466 |      41.5 |             0.6 |         15909 |        27155 |     7/8 |       20.0 |                    0 |
|       4 |    594 |      30.7 |             0.6 |         14854 |        27669 |     8/8 |       23.1 |                    0 |
|       5 |    548 |      40.1 |             0.6 |         16847 |        31558 |     7/8 |       26.0 |                    1 |
|       6 |   1437 |      12.6 |             1.5 |         14403 |        22128 |     8/8 |       22.5 |                    0 |
|       7 |   4394 |       4.1 |             0.6 |         14428 |        28581 |     6/8 |       22.7 |                    0 |
|       8 |  14754 |       1.1 |             0.6 |         12972 |        42538 |     7/8 |       20.4 |                    0 |
|       9 |   1836 |       9.8 |             0.6 |         15251 |        25675 |     8/8 |       21.0 |                    0 |

## Trainer-side reward signal

| rollout | raw_reward | rewards (post-norm) | log_probs | rollout_log_probs | kl |
|--------:|-----------:|--------------------:|----------:|------------------:|---:|
|       0 |      4.84% |             -0.0028 |   -0.7301 |           -0.7076 | +0.0000 |
|       1 |      4.92% |             -0.0033 |   -0.6785 |           -0.6591 | +0.0000 |
|       2 |      5.00% |             -0.0037 |   -0.7377 |           -0.7170 | +0.0000 |
|       3 |      4.92% |             -0.0033 |   -0.7451 |           -0.7234 | +0.0000 |
|       4 |      5.00% |             -0.0037 |   -0.7208 |           -0.6983 | +0.0000 |
|       5 |      4.84% |             -0.0033 |   -0.7252 |           -0.7036 | +0.0000 |
|       6 |      5.00% |             -0.0037 |   -0.7644 |           -0.7434 | +0.0000 |
|       7 |      4.84% |             -0.0028 |   -0.7279 |           -0.7055 | +0.0000 |
|       8 |      4.92% |             -0.0033 |   -0.6534 |           -0.6335 | +0.0000 |
|       9 |      5.00% |             -0.0037 |   -0.7283 |           -0.7064 | +0.0000 |

## Park / resume sandbox events

Not applicable — `--partial-rollout` is OFF, so no sandbox parking. Weight
updates still abort in-flight requests (SGLang internal), but those samples
restart from scratch instead of resuming.

| metric | value |
|---|---:|
| Returned-aborted-group events | 99 |

## Why TIMEOUT?

Hit Slurm wallclock at 08:00:17. Step 8 alone ate half the wallclock —
one or two Qwen3-Thinking trajectories produced ~40-50K tokens of agentic
+ chain-of-thought output and gated the entire rollout finalization.

## Verdict

The partial-rollout vs no-partial-rollout comparison (09 vs 10) is now clean:

```
                   09 (partial-rollout ON)         10 (this — partial-rollout OFF)
  result:          FAILED — gloo recv timeout      TIMEOUT — 8h wall
  rollouts done:   14 in 7h25m                     10 in 8h
  pathology:       resume_count cycling (400+×)    monolithic slow tails (1 sample / 4h)
  raw_reward:      0.035–0.050 (noisy)             0.048–0.050 (flat)
  learning:        none visible                    none visible
```

Neither variant produced reward movement. The bottleneck is **per-sample
agentic wall-clock cost being unbounded** under our settings
(max_turns=30, max_response=32K, Qwen3-Thinking's 8-15K thinking tokens
per hard turn). The bottleneck is the workload, not slime's machinery.

The right lever to bound rollout time is `--rollout-max-response-len`
(per-turn cap), not max_turns — see experiment 11 for that test.
