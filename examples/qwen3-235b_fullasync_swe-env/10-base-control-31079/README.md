# 10 — Base + Lite + interval=5, NO partial-rollout (job 31079 — TIMEOUT)

**Status:** ❌ TIMEOUT at 8h wall-clock, 10 training steps completed
**Lib SHA:** `a5878973` (canonical, with metadata-based metric fix)
**Dataset:** Lite sample_40
**Headline:** No-partial-rollout control to compare against experiment 09.
Same setup, same dataset, same trainer config — just no `--partial-rollout`
flag, so weight-update aborts discard the partial work and re-dispatch
from scratch. Avoided 09's cycling pathology, but hit a different one:
**monolithic slow tails** — single trajectories grinding for hours.
Step 8 alone took 4h 5m.

## What changed vs experiment 09

| Knob | 09 (resumable, FAILED) | 10 (this) |
|---|---|---|
| `--partial-rollout` | ON | **OFF** |
| Everything else | identical | identical |

## How it ran

- 10 rollouts (r0–r9) in 8h wallclock before TIMEOUT
- p0 (init), p5 (1st interval push), p10 not reached
- raw_reward: 0.048–0.050 — flat, no drift
- `truncated_ratio = 0.00` across all rollouts
- `n_resumed = 0` everywhere (no partial-rollout, so no resume cycle)
- Single-sample resp_max: 42538 in r8 (total length ~51K including 8K prompt)

## Step time profile

```
p0: step= 943s  (init)
p1: step= 474s
p2: step= 443s
p3: step= 482s
p4: step= 598s
p5: step= 625s  (1st weight push, upd=43s)
p6: step=1418s  (post-push)
p7: step=4387s  ← 73 min anomaly
p8: step=14751s ← 4h 5m !!! single step
p9: step=1846s  (recovery)
```

Step 8 ate half the wallclock. One or two Qwen3-Thinking trajectories in
that rollout produced ~40-50K tokens of agentic + chain-of-thought
output, and the rollout couldn't finalize until they finished.

## Why no partial-rollout doesn't help either

Without `--partial-rollout`:
- ✅ No abort cycling (each aborted sample restarts from scratch — cleaner)
- ❌ Each "slow tail" sample blocks its rollout for hours
- ❌ Wallclock-per-rollout is gated by max-sample-trajectory-time, which
  for Qwen3-Thinking + max_turns=30 + max_response=32K can be hours

So **the bottleneck migrates** but doesn't disappear:
- With partial-rollout: many fast re-dispatches → cycling → trainer-collective starvation
- Without partial-rollout: few slow monoliths → wallclock saturation

Both are manifestations of the same root cause: **per-sample agentic-RL
cost is unbounded under our current settings** (max_turns=30, max_response=32K,
Qwen3-235B-Thinking emits 8-15K thinking tokens per hard turn).

## Files of interest

- `artifacts/timeout-tail-50.log` — Slurm timeout output
- `metrics.txt` — per-rollout numbers
- `run_swe.sh` — exact launcher (no `--partial-rollout`)

## Verdict

The partial-rollout vs no-partial-rollout comparison is now clean:

```
                   09 (partial-rollout)          10 (no partial-rollout)
  result:          FAILED (gloo recv timeout)    TIMEOUT (8h wall)
  rollouts done:   14                            10
  pathology:       resume_count cycling (400×)   monolithic slow tails (1 sample / 4h)
  reward:          0.035-0.050 (noisy)           0.048-0.050 (flat)
  learning:        none visible                  none visible
```

Neither variant produces useful reward movement in <10 steps. The bottleneck
is **not** slime's machinery — it's the agentic-RL premise on this scaffold.
Real progress would need lower `--rollout-max-response-len` (cap per-turn
thinking) and/or a faster model.
