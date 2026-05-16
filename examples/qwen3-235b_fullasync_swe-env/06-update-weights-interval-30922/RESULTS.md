# Base + --update-weights-interval=5 — job 30922

- Log: `/home/sa-shared/kimbo/slime/mnt/logs/infx-swe-30922.out`
- Steps captured: 4
- Rollouts captured: 4

## Per-step performance

| step | wait (s) | train (s) | step (s) | train_tflops | MFU(in-step) | wallclock_MFU | wait_ratio |
|-----:|---------:|----------:|---------:|-------------:|-------------:|--------------:|-----------:|
|    0 |      883 |       234 |     1117 |         52.8 |        5.33% |         0.85% |       0.79 |
|    1 |      211 |       179 |      390 |         66.9 |        6.76% |         2.73% |       0.54 |
|    2 |      374 |       192 |      566 |         64.7 |        6.53% |         1.97% |       0.66 |
|    3 |      400 |       179 |      579 |         66.5 |        6.72% |         1.84% |       0.69 |

### Steady-state aggregate (step 1+)

- mean train_tflops: **66.0** (MFU 6.67%)
- mean wait_time: 328s
- mean train_time: 183s
- mean wait_ratio: 0.63

## Per-rollout metrics

| rollout | rt (s) | tok/gpu/s | sandbox_acq (s) | resp_len_mean | resp_len_max | zero_std | tool_calls | policy_staleness_max |
|--------:|-------:|----------:|----------------:|--------------:|-------------:|---------:|-----------:|---------------------:|
|       0 |    840 |      21.9 |             3.5 |         14413 |        28055 |     5/8 |       24.0 |                    0 |
|       1 |    445 |      45.2 |             0.6 |         16652 |        32672 |     7/8 |       24.3 |                    1 |
|       2 |    552 |      37.4 |             0.6 |         17024 |        27287 |     6/8 |       21.9 |                    1 |
|       3 |    591 |      33.9 |             0.6 |         16945 |        37520 |     8/8 |       22.6 |                    2 |

## Trainer-side reward signal

| rollout | raw_reward | rewards (post-norm) | log_probs | rollout_log_probs | kl |
|--------:|-----------:|--------------------:|----------:|------------------:|---:|
|       0 |      4.69% |             -0.0023 |   -0.7604 |           -0.7390 | +0.0000 |
|       1 |      4.92% |             -0.0033 |   -0.7507 |           -0.7297 | +0.0000 |
|       2 |      4.84% |             -0.0028 |   -0.8130 |           -0.7903 | +0.0000 |
|       3 |      5.00% |             -0.0037 |   -0.7602 |           -0.7380 | +0.0000 |
