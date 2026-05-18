# Experiments — chronological log

Linear log of what we tried, what changed, and what we learned. Read top
to bottom for the journey.

Each experiment dir is **self-contained** for what was *unique* to that
run (its `run_swe.sh` / `.sbatch` / metrics / artifacts), and uses
**`../lib/`** for the shared agent runtime (sandbox, agent loop,
metric wrapper, data prep). The README at the top of each dir links to
the relevant `lib/` SHA so anyone wanting an exact replay can
`git checkout <sha> -- ../lib` and re-run.

## Status table

| # | Experiment | Job | Status | One-line outcome |
|---|---|---|---|---|
| 01 | initial-bringup | 29747 | ✅ | 12-node fully-async + Modal sandbox bring-up green |
| 02 | context-scaling | 29922 cohort | ✅ | 64K is the knee — smallest cap with 0 truncations |
| 03 | slime-metrics-instrumentation | 30045 | ✅ | per-tool + per-sandbox + per-outcome telemetry working |
| 04 | pd-disaggregation-abandoned | (multiple) | ❌ | Mooncake KVTransferError, abandoned upstream |
| 05 | resumable-smoke | 30707 | ✅ | first end-to-end --partial-rollout validation, 20 steps |
| 06 | update-weights-interval | 30922 | ✅ | interval=5 cut step time ~52% (1078s → 511s) |
| 07 | switch-to-lite-40 | — | ✅ | denser per-group reward variance vs Verified |
| 08 | metadata-metric-fix | 30939 | ✅ | found Ray-attr-strip bug, fixed via sample.metadata, 10 hermetic tests |
| 09 | resumable-full-run | 31011 | ❌ | FAILED at 7h25m — gloo recv timeout, runaway abort cycling |
| 10 | base-control | 31079 | ❌ | TIMEOUT at 8h — catastrophic single step took 4h (32K response cap) |
| 11 | sglang-deepep-16k-response | 31395 | ⚠️ | DeepEP `auto` (low_latency): straggler-tail collapse, 8× per-token slowdown post-push |
| 12 | deepep-normal-16k-response | 31507 | ❌ | DeepEP `normal` mode DOA: cuda graphs broken, 0% hit rate, 5-11 tok/s eager |
| 13 | ep8-no-deepep-16k-response | 31508 | ✅ | TIMEOUT 8h, 9 perfs. Confirms straggler tail is workload-fundamental, not DeepEP-specific |
| 14 | pp4cp2-32k-trainer-8k-response | 31816 / 31887 | ❌ | 32K trainer + 8K response: 30% faster steady-state but 40% reward drop. Re-launch crashed at 7h24m with same gloo recv timeout as 09. 8K cap doesn't prevent the long tail |
| 15 | pp4cp2-32k-trainer-16k-response | 31888 | ❌ | Same trainer reshape with 16K response: reward recovered (~5%) but steady-state SLOWER than 13 (558s vs 465s), crashed at 2h00m on 1st weight push. **PP=4/CP=2 reshape doesn't help.** |

## Top-level findings

**The plumbing works.** Modal sandboxes, the 6-tool agent loop, fully-async
rollout, partial-rollout machinery, weight-interval pushing, and the metric
pipeline are all verified end-to-end (experiments 01, 03, 05, 06, 08).

**Agentic RL on Qwen3-235B-Thinking under our settings is rollout-bound,
not trainer-bound.** Both 09 and 10 saw rollouts dominated by a small
number of long-thinking trajectories (~50K tokens, 4h+ per sample). The
trainer sat idle 70%+ of wallclock.

**Partial-rollout doesn't help on long-context agentic tasks.** It moves
the bottleneck (from monolithic slow tails to cycling re-dispatches) but
doesn't eliminate it. Under heavy cycling, the trainer's gloo collectives
eventually time out and crash the run (09).

**`--rollout-max-response-len 16K` is the single most impactful flag.**
10 (with 32K cap) had a catastrophic single 4-hour step (`p8=14751s`).
Both 11 and 13 (with 16K cap) keep the worst step under 4000s. This
flag alone delivers ~2× cumulative wallclock improvement.

**The straggler tail is workload-fundamental, not backend-specific.**
Experiments 11 (DeepEP `low_latency`) and 13 (plain alltoall), with
identical EP=8 + 16K otherwise, have nearly identical p7-p8 wallclock
(6709s vs 6749s) despite using completely different MoE backends. The
~8× per-token slowdown on the straggler sample in the post-push regime
appears regardless of DeepEP. **Conclusion: there is no inference-side
config lever that fixes this.**

**The bottleneck driver is per-turn thinking-token volume**, not
`max_turns`. Mean tool-calls-per-sample was 20-22, well under the
`max_turns=30` cap. The cost comes from Qwen3-Thinking emitting 8-15K
chain-of-thought tokens per hard turn. The right lever is
`--rollout-max-response-len` (per-turn cap), not max_turns.

**EP=8 vs EP=4** has only marginal impact. The biggest measurable
difference is on the weight broadcast step (`update_weights_time` drops
from ~43s to ~37s with EP=8). DeepEP-the-backend was the wrong attribution
for most of that win — the EP=8 sharding itself is responsible.

**`--rollout-max-response-len 8K` (exp 14) is too aggressive.** Halving
the per-turn cap from 16K to 8K gave 30% steady-state speedup but
dropped raw_reward from ~5% to ~3% (40% relative drop). `truncated_ratio
= 0.00` confirms it's not hard truncation — the model just *adapts* to
the smaller budget and emits shallower thinking, so patch quality
drops. **16K is the sweet spot** for both speed and accuracy.

**Effective best config** (matches experiment 13, *not* 14):

```
--rollout-max-response-len 16384      # cap response per turn (most important)
                                      # NOT 8K — 8K hurts accuracy too much (exp 14)
--update-weights-interval 5           # amortize weight broadcasts
--sglang-ep-size 8                    # marginal win on broadcast
# (default alltoall; do NOT use --sglang-moe-a2a-backend deepep)
```

**Trainer reshape (PP=4/CP=2, 32K context) doesn't help (exp 15 resolved
this).** Tested directly with 16K response cap: per-step time goes from
13's 465s → 15's 558s (20% **slower**), AND the run crashed at 2h00m on
the first weight push (gloo recv timeout — same class as 09/14, just
earlier). PP=4 adds pipeline-bubble overhead and more collective
participants under stress, with no compensating benefit at our scale.

**14's apparent 30% speedup was 100% from the 8K response cap, which
cost ~40% of reward.** Not a usable optimization.

**Reward signal is still flat at ~5% across all runs (09-13).** The
straggler problem isn't blocking learning per se — it's just slowing
iteration. To see real reward movement we'd need either reward shaping,
a smaller / faster model, or more curated easy instances. The slime
plumbing and example are not the bottleneck.

## How to add a new experiment

1. Copy the most recent comparable experiment's `run_swe.sh` and
   `run_swe.sbatch` into a new `NN-name/` dir at the example root.
2. Edit flags.
3. Submit: `sbatch NN-name/run_swe.sbatch`.
4. Once the run lands, write `README.md`, `metrics.txt`, drop a trimmed
   error tail in `artifacts/` if it failed, and append a row to this table.
