# 14 — 32K trainer (PP=4 CP=2) + 8K per-turn response cap

**Status:** ❌ FAILED — 31887 re-launch crashed at 7h24m via gloo collective timeout (same class as 09/31011) after a catastrophic p7=15807s spike. 8K cap drops reward ~40% AND doesn't prevent the long tail. See RESULTS.md.
**Lib SHA:** current (post-`a5878973`, with JSONL metric dump active for rollout + trainer)
**Dataset:** Lite sample_40
**Headline:** Same as 13 (alltoall + EP=8 + interval=5, no SGLang DeepEP)
but with **smaller trainer context** (32K instead of 64K, via the 02
context-scaling 32K-tier parallelism: PP=4 + CP=2) and **half the response
cap** (8K per turn instead of 16K). Tests whether tightening both the
trainer cap and the per-turn response cap further reduces the
straggler-tail problem we saw in 09-13.

## What changed vs 13

| Knob | 13 | 14 (this) |
|---|---|---|
| `--pipeline-model-parallel-size` | 2 | **4** |
| `--context-parallel-size` | 4 | **2** |
| `--decoder-last-pipeline-num-layers` | (not set) | **22** (PP=4 split: 24/24/24/22) |
| Effective trainer context cap | CP × max_tok/gpu = 4 × 16K = **64K** | CP × max_tok/gpu = 2 × 16K = **32K** |
| `--sglang-context-length` | 65536 | **32768** (matches new trainer cap) |
| `--rollout-max-response-len` | 16384 | **8192** (halved) |
| Trainer MoE backend | DeepEP (`--moe-flex-dispatcher-backend deepep`) | DeepEP (unchanged — always on) |
| SGLang MoE backend | alltoall (default) | alltoall (default — unchanged) |
| `--update-weights-interval` | 5 | 5 |
| `--sglang-ep-size` | 8 | 8 |
| `--partial-rollout` | OFF | OFF |

## Hypothesis

13 confirmed the straggler tail is workload-fundamental — no SGLang
backend choice rescues it. The remaining lever is the workload itself.
Two complementary moves here:

1. **Smaller per-turn response cap (16K → 8K).** Forces the model to
   emit tool calls sooner; can't burn 15K tokens of thinking in one
   shot. Should bound the worst-sample wallclock more tightly.

2. **Smaller trainer context window (64K → 32K).** Halves the trainer's
   activation memory budget; could allow larger `max-tokens-per-gpu`
   later if we need it, and reduces the size of the train step's gradient
   accumulation. The 8K per-turn cap means total response lengths
   should stay well within 32K minus prompt, so we don't lose useful work.

If both changes work in concert, we expect:
- Per-step time more uniform across the rollout (less long tail)
- Trainer MFU improves slightly (smaller activations to crunch)
- Potentially more steps in the 8h budget vs 13's 9 perfs
- Reward signal still flat (this is a throughput experiment, not a learning one)

If the agent doesn't have enough thinking budget at 8K per turn and
truncates often, we'll see `rollout/truncated_ratio > 0` and may need
to back off to 12K or revert to 16K.

## Why "32K trainer config from 02"

The 02 context-scaling sweep showed 32K was the smallest cap with
truncated_ratio=0.000 on Verified data (16K and 32K both hit
truncated_ratio=0.047 on the longest-thinking samples). For Lite-40
with an 8K per-turn cap, multi-turn responses should rarely exceed 32K
total — but worth watching.

The PP=4 + CP=2 shape from 02's 32K tier reuses the
`--decoder-last-pipeline-num-layers 22` setting which splits Qwen3-235B's
94 layers as 24/24/24/22 across PP stages.

## How to launch

```bash
sbatch 14-pp4cp2-32k-trainer-8k-response/run_swe.sbatch
```

8h wallclock will be set via `scontrol update job=${JOB} TimeLimit=08:00:00`
after submission (default sbatch is 24h).
