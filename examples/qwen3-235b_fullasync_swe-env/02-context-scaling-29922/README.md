# 02 — Trainer context-scaling sweep (16K / 32K / 64K / 128K — job 29922 cohort)

**Status:** ✅ knee found at 64K
**Lib SHA:** `09f159dd` (per-tool sandbox metrics + context-scaling variants)
**Dataset:** Verified sample_10
**Headline:** **64K is the smallest cap that produces 0 truncations** on
Qwen3-235B-Thinking's natural trajectory length distribution. See `COMPARISON.md`
for the full per-tier numbers.

A sweep of trainer-context budgets for the swe-env workload on Qwen3-235B-A22B-Thinking-2507. Tiers explore how much room the agent's trajectory needs and where the cost/benefit knee sits.

## What's here

| file | tier | trainer parallelism | effective cap |
|---|---:|---|---:|
| `run_swe_16k.{sh,sbatch}` | **16K** | TP=4 PP=4 CP=2, max_tok/gpu=8K | 16,384 tokens |
| `run_swe_32k.{sh,sbatch}` | **32K** | TP=4 PP=4 CP=2, max_tok/gpu=16K | 32,768 tokens |
| `run_swe_64k.{sh,sbatch}` | **64K** | **TP=4 PP=2 CP=4**, max_tok/gpu=16K | 65,536 tokens |
| `run_swe_128k.{sh,sbatch}` | **128K** | TP=4 PP=2 CP=4, max_tok/gpu=32K | 131,072 tokens |
| `COMPARISON.md` | — | cross-tier comparison + headline numbers | — |

The 32K → 64K transition is the only point where parallelism reshapes (PP=4 → 2, CP=2 → 4) — needed to give CP more ranks to spread activations once max_tok/gpu × CP exceeds the PP=4 shape's headroom.

## Submitting

```bash
sbatch examples/qwen3-235b_fullasync_swe-env/context-scaling/run_swe_<tier>.sbatch
```

Each tier's `.sh` knows where it sits in the tree (one extra `../` for `SLIME_ROOT`) and PYTHONPATH includes the parent example dir so `rollout_metrics` / `coding_sandbox` / `generate_with_codingagent` are importable from a subdir.

## Which tier should I use?

**64K is the canonical pick.** It's promoted to the parent dir as `run_swe.{sh,sbatch}` because it's the smallest cap that doesn't truncate the long-tail of swe-env trajectories. See `COMPARISON.md` for the per-tier numbers and the reasoning.

The 16K and 32K tiers are kept around for the smaller-cluster footprint case (TP=4 PP=4 CP=2 fits in fewer GPUs if PP=2 is unavailable). The 128K tier is the stretch shape; it works but doesn't help on the current workload because untrained Qwen3-235B's natural trajectory length is ~20K with rare excursions to 50K — almost never above 64K.
