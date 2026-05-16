# 09 — Resumable + Lite + interval=5 + fixed metrics (job 31011 — FAILED)

**Status:** ❌ FAILED at 7h25m — gloo recv timeout
**Lib SHA:** `a5878973` (metadata-based metric fix in place)
**Dataset:** Lite sample_40
**Headline:** Partial-rollout cycling pathology under long-context agentic
settings. Resumed samples accumulated 400-700 re-dispatches each, with
~70 tokens of new generation per cycle. By step 14, rollout time hit 4969s
(83 minutes) — exceeding gloo's 30-minute collective recv timeout — and
the trainer crashed.

## What changed vs experiment 08

| Knob | 08 (metric-fix discovery) | 09 (this) |
|---|---|---|
| `lib/` | pre-fix (`cac0d666`) | post-fix (`a5878973`) |
| Hermetic test coverage | none | 10 tests + 18 sandbox tests |
| Goal | find the bug | run with fixed metrics, see real numbers |
| Otherwise | identical | identical |

## How it ran

- 14 rollouts (r0–r13) before crash
- p5 and p10 had the expected `update_weights_time ≈ 35s` (interval=5 pushes)
- PARKED=63, RESUMED=63 (all parked sandboxes drained)
- 6772 "Returned aborted group" events — slime's group-level retry firing
  for groups with any aborted sample
- raw_reward: 0.048 → 0.035 (dip at r10) → 0.044 → 0.047 (recovery at r13)
- First non-zero `n_resumed` landed at r8 (resume_mean=55.8, n_resumed=8)
- resume_mean grew to 91.5 by r12 (→ ~730 cycles per resumed sample)

## Why it crashed

```
RuntimeError: [.../gloo/transport/tcp/unbound_buffer.cc:78]
  Timed out waiting 1800000ms for recv operation to complete
```

Step 14 (`p14`) took 4969s — exceeded the 30-minute gloo collective timeout.
One trainer rank waited 30 min for a peer that was idle (rollout-gated)
and gave up.

**Root cause** (from log analysis):
1. 37 samples got parked at p5's first weight push.
2. Their groups had ≥1 ABORTED sample → slime's
   `fully_async_rollout.py:198-211` returns the *whole group* to the buffer.
3. On re-dispatch, the aborted sample's SGLang call **kept aborting again**
   (likely propagation aftermath / scheduler state) and only produced
   ~50-100 tokens before each new abort.
4. Each cycle: ~70 new tokens, status=ABORTED, return group, re-dispatch.
   Took seconds per cycle.
5. Over hours, samples accumulated 400-700 cycles. Each grew toward the
   64K context cap. Slow tail samples gated the rollout, gating the
   trainer, blowing the gloo timeout.

This is **not a fix-this-in-our-example** bug — the cycling lives in
slime's partial-rollout + SGLang interaction. But it shows that naive
partial-rollout doesn't work on long-context agentic tasks under these
settings. See **experiment 10** for the no-partial-rollout control.

## Files of interest

- `artifacts/crash-tail-100.log` — gloo timeout traceback
- `metrics.txt` — per-rollout numbers + raw_reward trajectory
- `run_swe.sh` — exact launcher (has `--partial-rollout` + `--update-weights-interval=5`)

## Verdict

✅ The metric pipeline is **physically verified** (PARKED=63=RESUMED=63;
`policy_staleness` and `resume_count` reporting non-zero real values).

❌ The partial-rollout *training dynamics* under our settings are
pathological — runaway cycling, eventually crashes the trainer.

❌ No reward-side benefit visible: raw_reward stays at the same ~5% baseline
as the no-partial-rollout control (experiment 10) for the same number of
rollouts.
