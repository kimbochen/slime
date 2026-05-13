# Qwen3-235B SWE-env — 64K context run with `add_slime_metrics` (job 30045)

Re-run of the 64K config (job 29922, [`RESULTS_64K.md`](RESULTS_64K.md)) now
routed through `add_slime_metrics.py` instead of the default slime metrics.
Surfaces per-tool / per-sandbox / per-SWE-bench-outcome / policy-staleness
data that the previous run couldn't see.

Run wall-clock: **~3h35m** from sbatch to perf 19. Cluster: 4 actor + 8
rollout H200 nodes (12 total).

## Headline

| metric (20-rollout mean) | 29922 (orig 64K) | **30045 (this, +metrics)** |
|---|---:|---:|
| raw_reward | 0.0491 | **0.0481** |
| truncated_ratio | 0.000 | 0.003 (one batch had 6.2%) |
| response_lengths (batch mean) | 19,670 | 20,166 |
| **max response in any sample** | 22,987 | **51,133** ← one sample nearly hit 64K cap |
| log_probs_time (s) | 7.32 | 7.02 |
| actor_train_time (s) | 66.7 | 69.6 |
| update_weights_time (s) | 35.1 | 35.8 |
| log_probs_tflops | 121.6 | 132.4 |
| actor_train_tflops | 38.4 | 39.4 |
| rollout_time (s) | 503.9 | 609 (max 1413s) |

The two runs are within noise on every trainer-side number. The interesting
delta is **`max response in any sample 22,987 → 51,133`** — 30045 happened
to see longer trajectories on a few samples (notably rollouts 5 and 16 had
samples breaking 49 K and 51 K tokens). 32 K trajectories also appeared
several times (rollouts 1, 3, 14, 19). 64 K is genuinely doing work on
this workload; 32 K would have truncated multiple times.

## What the new metrics reveal

### Per-tool call patterns (mean across 320 trajectories = 20 rollouts × 16)

| tool | calls/sample | time/call | error_rate |
|---|---:|---:|---:|
| `run_command` | 10.96 | 1.24 s | **65.4%** |
| `read_file` | 2.12 | 0.62 s | 0.0% |
| `write_file` | 4.58 | 0.34 s | 0.0% |
| `apply_patch` | 3.63 | 0.76 s | **99.3%** |
| `run_tests` | 1.63 | 0.42 s | **100.0%** |
| **total** | **22.93** | — | — |

Per trajectory, the agent spends **20.3 s of wall-clock inside the sandbox**
across 22.9 tool calls. The sandbox itself is fast — the bottleneck is
trajectory length, not tool latency.

### SWE-bench outcomes

- **`swebench/submitted_rate = 0.963`** — the agent completed its agent
  loop (called the `submit` tool) on ~30 of every 31 trajectories. Almost
  no trajectories truncate-out without submitting.
- **`swebench/tests_pass_rate = 0.000`** — *no* SWE-bench instance passed
  `/eval.sh` on any rollout. The 0.0481 raw_reward seen by the trainer is
  the "you submitted but didn't pass" partial credit (0.05) being averaged
  with the rare (3.7%) ones that didn't even submit.

### Policy staleness

```
mean = 0.04   max = 1
```

Fully-async pipeline is working as intended — almost zero drift between
rollout-time and trainer-time policy. (Each rollout was generated at most
1 trainer step before the trainer consumed it.)

## The real bottleneck: patch generation

The new metrics make the failure mode obvious:

```
apply_patch  count/sample=3.63  error_rate=99.3%
run_tests    count/sample=1.63  error_rate=100%
```

The agent attempts to apply diffs ~3.6 times per trajectory; **99.3%** of
those `git apply` calls fail (malformed diff, wrong file path, context
mismatch). Test execution then fails 100% — `/eval.sh` can't run if the
patch never landed. The agent calls `submit` anyway (`submitted_rate=0.96`),
collects the 0.05 partial-credit reward, and the cycle repeats.

The high `run_command` error rate (65.4%) is consistent with the same
behaviour — the agent runs commands that exit non-zero (failing greps,
missing files, etc.) while exploring.

**Implication for training:** the gradient signal on this slice is
essentially "always 0.05" — group-relative advantages are nearly zero
(`rollout/advantages` was 0.0 or ~10⁻⁸ in the trainer log for every batch).
There's no learning signal until the agent occasionally produces a passing
patch. To unblock learning we'd need:

1. **Better patch-generation prompting / few-shot examples in the system
   prompt** — the `apply_patch` failure rate at 99% is well below what
   Qwen3-235B should be able to do given even modest scaffolding.
2. **Partial-credit reward** — parse `/eval.sh` to count passing tests
   even when not all do. Currently a single failing test = 0.0 reward.
3. **Easier slice for cold-start** — `sample_10.jsonl` is a balanced
   slice from 5 popular repos; some include heavy-dependency tests. A
   "warm-up" slice of instances with simpler fix patterns would give
   the policy non-zero rewards to bootstrap from.

## Per-rollout panel

| N | resp_mean | resp_max | trunc | raw_reward | submit | tool_calls | apply_patch_err | rollout_time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 14200 | 19464 | 0.000 | 0.0500 | 1.00 | 18.9 | 1.00 | 553s |
| 1 | 16891 | **33161** | 0.000 | 0.0469 | 0.94 | 20.6 | 1.00 | 438s |
| 2 | 16263 | 24125 | 0.000 | 0.0500 | 1.00 | 22.8 | 1.00 | 455s |
| 3 | 17758 | **40427** | 0.000 | 0.0438 | 0.88 | 21.6 | 1.00 | 786s |
| 4 | 18815 | 28823 | 0.000 | 0.0469 | 0.94 | 24.2 | 0.98 | 1413s |
| 5 | 18682 | **51133** | **0.062** | 0.0469 | 0.94 | 22.7 | 1.00 | 995s |
| 6 | 14558 | 19845 | 0.000 | 0.0500 | 1.00 | 22.1 | 0.98 | 394s |
| 7 | 13753 | 23275 | 0.000 | 0.0469 | 0.94 | 23.1 | 1.00 | 449s |
| 8 | 17897 | 30720 | 0.000 | 0.0469 | 0.94 | 23.4 | 1.00 | 640s |
| 9 | 16168 | 23478 | 0.000 | 0.0500 | 1.00 | 23.4 | 1.00 | 531s |
| 10 | 13890 | 23464 | 0.000 | 0.0469 | 0.94 | 29.2 | 0.97 | 434s |
| 11 | 19229 | 26134 | 0.000 | 0.0500 | 1.00 | 21.5 | 0.98 | 492s |
| 12 | 14053 | 25584 | 0.000 | 0.0500 | 1.00 | 18.1 | 1.00 | 475s |
| 13 | 14532 | 18730 | 0.000 | 0.0500 | 1.00 | 24.4 | 0.97 | 346s |
| 14 | 17915 | **32963** | 0.000 | 0.0500 | 1.00 | 22.0 | 1.00 | 575s |
| 15 | 15713 | 20442 | 0.000 | 0.0500 | 1.00 | 27.8 | 0.99 | 382s |
| 16 | 15996 | **49199** | 0.000 | 0.0469 | 0.94 | 21.1 | 1.00 | 814s |
| 17 | 14800 | 18955 | 0.000 | 0.0500 | 1.00 | 26.9 | 1.00 | 340s |
| 18 | 17135 | 27339 | 0.000 | 0.0500 | 1.00 | 17.9 | 0.98 | 809s |
| 19 | 16855 | **33028** | 0.000 | 0.0406 | 0.81 | 26.8 | 1.00 | 855s |

**Bold `resp_max` = sample exceeded 32 K** (6 out of 20 rollouts had at
least one such sample, max 51 K). The 32 K config would have truncated
several of these.

## Reproducibility

- Job: `infx-swe-64k-30045`
- Log: `mnt/logs/infx-swe-64k-30045.out`
- Launcher: `run_swe_64k.sh` / `run_swe_64k.sbatch` on commit
  `09f159dd` of `feat/swe-env-example`
- Custom-generate / RM / log functions: `add_slime_metrics.{generate,reward_func,custom_rollout_log_function}`
- Prompt-data: `mnt/data/swebench_verified/sample_10.jsonl` (10 instances ×
  4 samples × 16 prompts/batch = 64 trajectories per rollout — wait, 16
  prompts/batch and 4 samples/prompt with `--rollout-batch-size 4`
  means 16 trajectories per rollout, hence the `0.063 = 1/16` truncation
  resolution)
- Image: `slimerl/slime:nightly-dev-20260425a` (SGLang 0.5.9)
- Trainer parallelism: TP=4 PP=2 CP=4 EP=8 ETP=1 + CPU offload
- Max context: CP × max_tokens_per_gpu = 4 × 16 384 = **65 536 (64 K)**
- Algorithm: GSPO, `--eps-clip 4e-4`, KL coef 0
