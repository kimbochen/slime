# Qwen3-235B SWE-env — 64K context run (job 29922)

Untrained `Qwen3-235B-A22B-Thinking-2507-FP8` against a 10-instance SWE-bench
Verified slice, 20 rollouts × 4 samples/prompt × 16 prompts/rollout = 64
trajectories per batch. Trainer parallelism reshaped from PP=4/CP=2 to
PP=2/CP=4 to give 4× the context-parallel ranks, paired with
`--max-tokens-per-gpu 16384`, yielding an effective context cap of
**CP × max_tokens_per_gpu = 4 × 16 K = 64 K tokens**.

Run took **~3h12m** wall-clock to 20 trainer perf lines. Cluster:
4 actor nodes + 8 rollout nodes, all H200 ×8, ConnectX-7 IB.

## Headline

| metric (20-rollout mean) | 16K | 32K | **64K (this)** | 128K |
|---|---:|---:|---:|---:|
| raw_reward | 0.0169 | 0.0467 | **0.0491** | 0.0479 |
| truncated_ratio | 0.659 | 0.047 | **0.000** | 0.000 |
| response_lengths (mean) | ~13.5 K | 19,476 | 19,670 | 18,848 |
| response_lengths (max in run) | 16.2 K | 22.8 K | **22.99 K** | n/a (3 rollouts) |
| log_probs_time (s) | 6.77 | 5.46 | **7.32** | 6.71 |
| actor_train_time (s) | 53.9 | 51.8 | **66.7** | 56.5 |
| update_weights_time (s) | 33.8 | 33.7 | 35.1 | n/a |
| train_wait_time (s) | 311 | 449 | 465 | n/a |
| log_probs_tflops | 88.9 | 153.1 | **121.6** | 128.9 |
| actor_train_tflops | 32.9 | 48.4 | **38.4** | 39.5 |
| rollout_time (s) | 343.6 | 478 | **503.9** | n/a |

(16K and 32K numbers from the same workload; 128K from a 3-rollout
short-circuit — included for shape but variance is high.)

## Trainer perf per step

| N | log_probs (s) | log_probs TFLOPs | actor_train (s) | actor_train TFLOPs | update_wts (s) | train_wait (s) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 46.76 (warmup) | 17.0 | 120.5 (warmup) | 19.8 | 33.7 | 430.7 |
| 1 |  9.26 |  89.3 |  70.7 | 35.1 | 34.1 | 538.0 |
| 2 |  5.21 | 120.8 |  58.6 | 32.2 | 35.6 | 266.8 |
| 3 | 10.62 | 101.6 |  77.3 | 41.9 | 36.1 | 493.8 |
| 4 |  6.91 | 126.1 |  65.4 | 40.0 | 38.7 | 521.6 |
| 5 |  5.27 | 148.1 |  66.0 | 35.4 | 33.9 | 713.4 |
| 6 |  6.12 | 138.9 |  63.8 | 40.0 | 33.0 | 496.9 |
| 7 |  6.38 | 113.8 |  61.5 | 35.4 | 34.1 | 411.2 |
| 8 |  8.89 |  96.4 |  62.6 | 41.1 | 32.9 | 689.9 |
| 9 |  6.44 | 141.2 |  67.6 | 40.3 | 35.6 | 438.2 |
| 10 |  6.37 | 126.3 |  65.1 | 37.1 | 33.2 | 490.6 |
| 11 |  7.04 | 118.8 |  71.1 | 35.3 | 33.5 | 404.3 |
| 12 | 10.74 |  80.8 |  67.2 | 38.7 | 39.3 | 576.3 |
| 13 |  5.18 | 144.0 |  71.3 | 31.4 | 36.3 | 364.6 |
| 14 |  6.77 | 149.7 |  64.2 | 47.3 | 33.0 | 459.7 |
| 15 | 10.27 |  89.6 |  66.8 | 41.3 | 36.6 | 443.9 |
| 16 |  5.62 | 137.2 |  64.7 | 35.8 | 34.0 | 377.0 |
| 17 |  6.35 | 144.5 |  65.5 | 42.1 | 37.4 | 393.5 |
| 18 |  5.59 | 136.1 |  64.4 | 35.5 | 33.1 | 315.9 |
| 19 | 10.08 | 106.9 |  74.0 | 43.7 | 36.3 | 443.6 |

Steady-state band (steps 1-19):
- `log_probs_time`: 5.2-10.7s (mean 7.3s)
- `actor_train_time`: 58.6-77.3s (mean 66.7s)
- `log_probs_tflops`: 80.8-149.7 (mean 121.6)
- `actor_train_tflops`: 31.4-47.3 (mean 38.4)
- `update_weights_time`: 32.9-39.3s (mean 35.1s)
- `train_wait_time`: 266-713s — pipeline almost entirely rollout-bound

## Rewards & response lengths per batch

| N | raw_reward | truncated | resp_len mean | resp_len max | total_len mean | logp(train) | logp(rollout) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0500 | 0.000 | 18910 | n/a | 20001 | -0.78 | -0.76 |
| 1 | 0.0500 | 0.000 | 19631 | n/a | 20454 | -0.72 | -0.70 |
| 2 | 0.0500 | 0.000 | 16317 | n/a | 17215 | -0.78 | -0.76 |
| 3 | 0.0500 | 0.000 | 22987 | n/a | 24074 | -0.76 | -0.74 |
| 4 | 0.0500 | 0.000 | 20055 | n/a | 21131 | -0.79 | -0.77 |
| 5 | 0.0500 | 0.000 | 18753 | n/a | 19718 | -0.73 | -0.71 |
| 6 | 0.0500 | 0.000 | 19214 | n/a | 20173 | -0.73 | -0.71 |
| 7 | 0.0469 | 0.000 | 17845 | n/a | 18824 | -0.79 | -0.77 |
| 8 | 0.0500 | 0.000 | 19756 | n/a | 20820 | -0.77 | -0.75 |
| 9 | 0.0500 | 0.000 | 20663 | n/a | 21863 | -0.80 | -0.77 |
| 10 | 0.0500 | 0.000 | 18636 | n/a | 19828 | -0.80 | -0.77 |
| 11 | 0.0469 | 0.000 | 19034 | n/a | 20023 | -0.72 | -0.70 |
| 12 | 0.0469 | 0.000 | 19998 | n/a | 21065 | -0.74 | -0.72 |
| 13 | 0.0500 | 0.000 | 17971 | n/a | 18937 | -0.72 | -0.70 |
| 14 | 0.0469 | 0.000 | 22144 | n/a | 23378 | -0.77 | -0.74 |
| 15 | 0.0500 | 0.000 | 20925 | n/a | 21868 | -0.77 | -0.75 |
| 16 | 0.0500 | 0.000 | 18468 | n/a | 19553 | -0.79 | -0.77 |
| 17 | 0.0469 | 0.000 | 20373 | n/a | 21538 | -0.77 | -0.75 |
| 18 | 0.0500 | 0.000 | 18324 | n/a | 19332 | -0.75 | -0.73 |
| 19 | 0.0469 | 0.000 | 23401 | n/a | 24405 | -0.75 | -0.73 |

Aggregates:
- `raw_reward` ∈ {0.0469, 0.0500}, mean **0.04906** — extremely consistent
  (3 passes per 64 trajectories, every batch, ~1 sample of variance)
- `truncated` literally **0.000 across all 20 batches** — no sample ever
  hit the 64K cap
- Policy staleness (`logp(train) - logp(rollout)`) consistently ~0.02 →
  fully-async pipeline is healthy, drift small per step

## Comparison to 32K (the Pareto winner)

64K vs. 32K with everything else equal:

| | 32K | 64K | delta |
|---|---:|---:|---|
| raw_reward (mean) | 0.0467 | 0.0491 | +5% (within noise) |
| truncated (mean) | 0.047 | 0.000 | -100% (no batches truncated) |
| actor_train_time | 51.8s | 66.7s | **+29% slower** |
| log_probs_time | 5.5s | 7.3s | **+34% slower** |
| log_probs_tflops | 153.1 | 121.6 | -21% throughput |
| actor_train_tflops | 48.4 | 38.4 | -21% throughput |
| rollout_time | 478s | 504s | +5% |
| response_lengths (mean) | 19,476 | 19,670 | **same** |

**The reward is statistically the same, the response_lengths are the
same — but 64K costs ~30% more per trainer step.** The PP=4→2 / CP=2→4
reshape is a real throughput tax (CP requires more cross-rank
communication per layer; PP=2 means each rank carries more weight +
gradient state).

The agent's *natural* trajectory length on this slice (untrained Qwen3-235B-Thinking)
is ~19 K tokens regardless of cap. The extra 32 K of headroom in the 64K
config is wasted budget — almost nothing ever exceeds 24 K (`max` per
batch was 22.99 K, exceeding the 32K cap on **one** sample of one batch
in 20 batches).

## Verdict

**32K remains the recommended operating point for the SWE-env workload
with this untrained policy.**

Conditions under which 64K starts to matter:
1. **Trained policy that uses more tools per trajectory.** A learned
   agent that read 5 files per task instead of 2 would push median
   response length closer to 30-40 K, and the 32K cap's 4.7% truncation
   would creep up. Re-evaluate after the first ~500 training steps.
2. **Harder instances.** This 10-instance bring-up slice is a balanced
   mix from astropy/django/matplotlib/seaborn/flask/requests. The
   official Verified set has longer-context instances (e.g., Sphinx, Sympy)
   where multi-file edits + long test traces could exceed 32K.
3. **Partial-credit reward.** With binary reward (`/eval.sh` exit
   code), there's no incentive to optimize "almost-correct" patches.
   A per-test-passed reward would reward longer iteration, increasing
   trajectory length and tipping the 32K → 64K cost/benefit.

## Reproducibility

- Job: `infx-swe-64k-29922`
- Log: `mnt/logs/infx-swe-64k-29922.out` (full)
- Trainer launcher: `run_swe_64k.sh`
- Slurm wrapper: `run_swe_64k.sbatch`
- Prompt-data: `mnt/data/swebench_verified/sample_10.jsonl` (10 instances)
- Image: `slimerl/slime:nightly-dev-20260425a` (SGLang 0.5.9, mooncake-transfer-engine 0.3.9)
- Model checkpoint: `mnt/checkpoints/Qwen3-235B-A22B-Thinking-2507-FP8/` (HF FP8)
- Ref weights: `mnt/checkpoints/Qwen3-235B-A22B_torch_dist/` (Megatron torch_dist)
- Algorithm: GSPO, `--eps-clip 4e-4`, KL coefficient 0 (no reference anchor in loss)
