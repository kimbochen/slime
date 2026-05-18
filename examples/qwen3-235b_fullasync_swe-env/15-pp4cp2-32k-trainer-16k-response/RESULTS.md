# 32K trainer (PP=4 CP=2) + 16K per-turn response — job 31888

- Log: `mnt/logs/infx-swe-31888.out`
- Steps captured: 5
- Rollouts captured: 6

## Per-step performance

| step | wait (s) | train (s) | step (s) | train_tflops | MFU(in-step) | wallclock_MFU | wait_ratio |
|-----:|---------:|----------:|---------:|-------------:|-------------:|--------------:|-----------:|
|    0 |      615 |       276 |      891 |         48.2 |        4.87% |         1.11% |       0.69 |
|    1 |      341 |       111 |      453 |         86.4 |        8.73% |         1.88% |       0.75 |
|    2 |      244 |       111 |      355 |         91.3 |        9.22% |         2.43% |       0.69 |
|    3 |      507 |       121 |      628 |         88.5 |        8.94% |         1.50% |       0.81 |
|    4 |      675 |       122 |      798 |         81.5 |        8.23% |         1.10% |       0.85 |

### Steady-state aggregate (step 1+)

- mean train_tflops: **86.9** (MFU 8.78%)
- mean wait_time: 442s
- mean train_time: 116s
- mean wait_ratio: 0.77

## Per-rollout metrics

| rollout | rt (s) | tok/gpu/s | sandbox_acq (s) | resp_len_mean | resp_len_max | zero_std | tool_calls | policy_staleness_max |
|--------:|-------:|----------:|----------------:|--------------:|-------------:|---------:|-----------:|---------------------:|
|       0 |    579 |      32.6 |             3.8 |         14555 |        23941 |     3/8 |       22.2 |                    0 |
|       1 |    617 |      27.9 |             0.6 |         13578 |        27693 |     6/8 |       22.3 |                    0 |
|       2 |    355 |      45.1 |             0.6 |         12810 |        21715 |     7/8 |       20.2 |                    0 |
|       3 |    617 |      29.7 |             0.6 |         15382 |        27453 |     3/8 |       19.1 |                    0 |
|       4 |    796 |      22.1 |             0.6 |         14336 |        24136 |     5/8 |       20.7 |                    0 |
|       5 |    883 |      21.2 |             0.6 |         14400 |        22860 |     3/8 |       24.8 |                    1 |

## Trainer-side reward signal

| rollout | raw_reward | rewards (post-norm) | log_probs | rollout_log_probs | kl |
|--------:|-----------:|--------------------:|----------:|------------------:|---:|
|       0 |      4.14% |             -0.0014 |   -0.7438 |           -0.7212 | +0.0000 |
|       1 |      4.84% |             -0.0028 |   -0.7214 |           -0.7008 | +0.0000 |
|       2 |      4.84% |             -0.0033 |   -0.6991 |           -0.6777 | +0.0000 |
|       3 |      4.45% |             -0.0014 |   -0.7397 |           -0.7184 | +0.0000 |
|       4 |      4.45% |             -0.0023 |   -0.7438 |           -0.7206 | +0.0000 |

## Result — trainer reshape doesn't speed up rollouts, AND crashed early

5 perfs landed (p0-p4) before the trainer crashed during the first
weight push at p5. Run: 2h 00m, FAILED with gloo recv timeout — same
class as 09/31011 and 14/31887, but earlier in the run.

### Per-step trajectory

```
              15/31888      13/31508 (the "best" baseline)
  p0:    891s              958s
  p1:    453s              323s
  p2:    355s              458s
  p3:    628s              680s
  p4:    798s              398s
  ─────────────────────────────
  p1-p4 mean: 558s          465s
  
  p5: trainer crashed during update_weights (gloo timeout, 30 min recv wait)
```

### Reward signal

```
  r0..r4 raw_reward:   0.041   0.048   0.048   0.045   0.045  → mean 0.045
  13's reference:                                              ~0.046
  14's broken value:                                           ~0.027
```

**16K cap fully recovered the reward** (matches 13's baseline). Confirms
14's reward drop was entirely from the 8K cap.

### Crash signature

```
RuntimeError: [.../gloo/transport/tcp/unbound_buffer.cc:78]
  Timed out waiting 1800000ms for recv operation to complete
ray.exceptions.RayTaskError(RuntimeError): MegatronTrainRayActor.update_weights()
```

Trainer was attempting the **first** interval-5 weight push at p5
(~10 min into rollout r5) and one rollout-side rank failed to respond
within 30 min. Memory was fine (76 GB GPU free, 370 GB host free at
crash). Pure collective starvation from a straggler in flight.

## Final verdict — the trainer reshape doesn't help

Three knobs tested across 13/14/15, isolated:

```
                              p1-p4 mean   reward    reliability
  13 (PP=2 CP=4, 16K resp):       465s     0.046    ran 8h to TIMEOUT (9 perfs, no crash)
  14 (PP=4 CP=2,  8K resp):       347s     0.027    crashed 7h24m (gloo timeout)
  15 (PP=4 CP=2, 16K resp):       558s     0.045    crashed 2h00m (gloo timeout, earlier!)
```

The PP=4/CP=2 trainer reshape from experiment 02 (which worked fine for
bring-up at 32K context) **doesn't translate to a useful speedup**
in our agentic-RL setup, and may even make the gloo-timeout crash more
likely (more PP stages = more collective participants = more brittle
under straggler stress).

### What this rules out

- **8K response cap (14):** drops reward 40%. Not usable.
- **PP=4/CP=2 trainer reshape (15):** slower than 13, AND crashes faster.
  Not usable.

### What stays as the recommended config

**Experiment 13's setup is still the best we've found:**

```
--rollout-max-response-len 16384        # 16K per-turn cap
--update-weights-interval 5
--sglang-ep-size 8
--pipeline-model-parallel-size 2
--context-parallel-size 4
# (no --decoder-last-pipeline-num-layers; PP=2 splits evenly)
# alltoall SGLang backend, NOT --sglang-moe-a2a-backend deepep
```

13 also eventually hits the 8h Slurm wallclock without producing reward
movement, but at least it doesn't crash mid-run.

### Why the trainer reshape might be slower

Hypothesis: PP=4 means 4 pipeline stages instead of 2. The per-step
training latency includes pipeline bubble overhead, which scales with
PP count. Even though CP=2 instead of CP=4 reduces per-rank
context-parallel comm, the PP bubble cost dominates for our short
rollouts. Net: slower per train step → slower per rollout step
(rollout-bound but trainer-bound contributes too at the margin).

The 02 context-scaling experiment showed PP=4 was needed for the 32K
config to fit memory; it's not a perf optimization, it's a memory
shape requirement. For 64K (PP=2/CP=4), PP=2 happens to be both
memory-feasible AND faster — best of both. That's what 13 uses.
