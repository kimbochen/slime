# 15 — 32K trainer (PP=4 CP=2) + 16K per-turn response cap

**Status:** ❌ FAILED at 2h00m — gloo recv timeout during 1st weight push (same crash class as 09/14). Trainer reshape is SLOWER than 13 (558s vs 465s) AND less reliable. 13's config remains the best. See RESULTS.md.
**Lib SHA:** current (post-`a5878973`, with JSONL metric dump for rollout + trainer)
**Dataset:** Lite sample_40
**Headline:** Same trainer reshape as 14 (PP=4 CP=2, 32K effective context)
but restores the 16K per-turn response cap (vs 14's 8K). Isolates the
trainer-reshape effect from the response-cap effect. If 15 keeps 14's
~30% steady-state speedup AND recovers 13's ~5% reward signal, we have
the best-of-both config.

## What changed vs 13 and 14

| Knob | 13 | 14 | 15 (this) |
|---|---|---|---|
| Trainer `--pipeline-model-parallel-size` | 2 | **4** | **4** |
| Trainer `--context-parallel-size` | 4 | **2** | **2** |
| Trainer `--decoder-last-pipeline-num-layers` | (not set) | **22** | **22** |
| Effective trainer context cap | **64K** | **32K** | **32K** |
| `--sglang-context-length` | 65536 | 32768 | **65536** (restored — 16K × multi-turn > 32K total) |
| `--rollout-max-response-len` | **16384** | **8192** | **16384** (restored) |
| MoE backends, EP=8, interval=5, partial-rollout, dataset | unchanged | unchanged | unchanged |

## Hypothesis

14 conflated two changes (trainer reshape + smaller response cap) so we
can't isolate which contributed how much to the 30% speedup. The reward
dropped from ~5% to ~3% — most likely from the 8K response cap forcing
shallower thinking.

15 keeps the trainer reshape but reverts the response cap to 16K. Three
possible outcomes:

1. **Best case:** 15 has 14's ~345s steady-state AND 13's ~5% reward.
   The trainer reshape alone gives the speedup; the 8K cap was the
   accuracy regression. **This becomes the new "effective best config."**

2. **Mixed case:** 15 falls between 13 and 14 on both axes. The 8K cap
   contributed to the speedup AND to the reward drop; the trainer
   reshape gave some independent speedup but less than 14's full 30%.

3. **Null case:** 15 has 13's ~465s steady-state AND ~5% reward. The
   trainer reshape contributed nothing — 14's speedup was entirely
   from the 8K response cap, and the reward drop was also from that.

If outcome 1 lands, we adopt 15's config for any future learning runs.
If 2 or 3, we're back to 13's config for accuracy + the existing
"agentic-RL is rollout-bound" verdict.

## Why bump SGLang context back to 65536

With 16K per-turn response cap and ~25-30 turns max, multi-turn responses
can accumulate to 50K+ tokens. Trainer's effective cap is 32K (CP × max_tok
= 2 × 16K) — but slime would just truncate / abort samples that exceed
that on the trainer side. For SGLang generation, we need enough headroom
so the model can finish trajectories without prompt+response exceeding
SGLang's context window.

13 used 65536 with 16K response cap and saw `resp_max=27K-30K` with
`truncated_ratio=0.00`. So 65536 is appropriate for 16K per-turn cap.

## How to launch

```bash
sbatch 15-pp4cp2-32k-trainer-16k-response/run_swe.sbatch
scontrol update job=${JOB} TimeLimit=12:00:00
```

Will queue behind 14 (31887). Picks up nodes when 14 finishes or is
cancelled.
