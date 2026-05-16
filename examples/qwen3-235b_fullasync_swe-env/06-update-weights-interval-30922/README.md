# 06 — `--update-weights-interval=5` (job 30922)

**Status:** ✅ headline win (52% step-time reduction)
**Lib SHA:** `cac0d666` (interval flag added; pre-metadata-fix)
**Dataset:** Lite sample_40 (base example, no partial-rollout)
**Headline:** Pushing weights every 5 steps instead of every 1 reduced
mean step time from **~1078s → ~511s (52% reduction)**, with no observed
reward regression. Weight-broadcast overhead dropped from ~7% to ~2.1% of
wallclock.

## Hypothesis

Under fully-async + every-step-push, ~7% of wallclock is the 35s weight
broadcast. Amortising the broadcast across 5 steps should reclaim most of
that, at the cost of slight policy-staleness (samples dispatched in the
same 5-step window are 0 weight-versions stale instead of N rollouts
stale).

## What changed vs experiment 05

| Knob | 05 (smoke) | 06 (this) |
|---|---|---|
| `--update-weights-interval` | 1 (default) | 5 |
| Other flags | identical | identical |

## Result

```
step_time mean: 1078s → 511s   (-52%)
update_weights_time as % of wallclock: ~7% → ~2.1%
raw_reward: ~5% baseline, unchanged
```

`policy_staleness/mean` reported sensible non-zero values here (0.0,
0.875, 0.75, 1.0 across r0–r3). Note: this is the **pre-fix raw-delta
formula** (rollout-id-delta, not weight-version-delta), and base-example
samples mostly *do* preserve their stamp because there's no partial-rollout
abort+re-dispatch cycle to trip the Ray-attribute-strip bug. The bug only
manifested visibly in the **resumable** variant; see experiment 08.

## How to reproduce

The flag change is one line in `run_swe.sh`:
```bash
   --update-weights-interval 5 \
```

The current canonical `lib/` already includes the metadata-based metric
fix, so re-running this experiment with the present `lib/` would produce
*correct* staleness numbers (unlike the original job 30922).
