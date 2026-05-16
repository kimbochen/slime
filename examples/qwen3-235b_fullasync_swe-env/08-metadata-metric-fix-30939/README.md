# 08 — Metadata-based metric fix (job 30939 exposed the Ray-strip bug)

**Status:** ✅ bug found, fix landed, hermetic tests added
**Lib SHA (pre-fix, ran job 30939):** `cac0d666` — attribute-based stamps
**Lib SHA (post-fix, canonical):** `a5878973` — metadata-based stamps + interval-aware staleness
**Dataset:** Lite sample_40
**Headline:** `resume_count/n_resumed` was stuck at 0 across all rollouts
even when sandbox-level PARKED/RESUMED logs showed 36/36 events (this run
had a full park/resume cycle, but the metric saw none of it). Same root
cause for `policy_version_at_dispatch`: both were stored on plain
attributes (`sample.foo = ...`), which Ray's cloudpickle strips when
sending the sample back through the partial-rollout buffer.

**Why staleness *partly* worked while resume_count totally failed**:
the wrapper re-stamps `policy_version_at_dispatch` if it's missing — so
on a re-dispatch, the strip just causes the stamp to be reset to the
current consumer_version. Stale = 0 for re-dispatched samples, but a
fresh sample completed in the same rollout would correctly stamp & log,
giving non-zero values for *some* batches. `resume_count` doesn't have
this fallback — every re-stamp restarts the counter at 0, so the metric
never accumulates. Hence 30939's `stale_mean` shows 0.25–0.875 across
r0–r6 (mostly correct-shaped but undercounted), while `n_resumed` was
0 across all 7 rollouts despite 36 sandbox-level resumes.

**Fix:** move both into `sample.metadata` (a real dataclass field,
survives serialization). Also made staleness *weight-version-aware*:
`weight_version = rollout_id // update_weights_interval`, so samples in
the same N-rollout window are 0 stale, not N stale.

## How it was found

In experiment 05 (job 30707) and the subsequent interval=5 runs, every
rollout reported `policy_staleness/mean = 0` despite the sandbox logs
clearly showing PARKED/RESUMED events from partial-rollout. That was
inconsistent with the design ("staleness should be > 0 whenever a sample
spans a weight push"). Tracing the data flow showed:

1. `rollout_metrics.generate()` stamped `sample.policy_version_at_dispatch =
   _consumer_version` as a *plain attribute*.
2. On abort + re-dispatch, slime routed the sample through Ray's
   cloudpickle-based data buffer.
3. Cloudpickle preserves *declared dataclass fields*; arbitrary attributes
   on `__dict__` get stripped. The stamp was gone by the time the consumer
   aggregator read it.
4. Same bug for `resume_count`.

## The fix

```python
# OLD (broken)
sample.policy_version_at_dispatch = _consumer_version
sample.resume_count = getattr(sample, "resume_count", -1) + 1

# NEW (canonical, post-a5878973)
if sample.metadata is None:
    sample.metadata = {}
if "policy_version_at_dispatch" not in sample.metadata:
    sample.metadata["policy_version_at_dispatch"] = _consumer_version
sample.metadata["resume_count"] = sample.metadata.get("resume_count", -1) + 1
```

Plus interval-aware staleness in the aggregator:

```python
interval = max(1, getattr(args, "update_weights_interval", 1) or 1)
versions = [(s.metadata or {}).get("policy_version_at_dispatch", rollout_id) for s in samples]
stalenesses = [max(0, (rollout_id // interval) - (v // interval)) for v in versions]
```

## Hermetic test coverage (tests/test_metric_fix.py)

10 tests covering:
- Sticky-on-first stamping (first call sets, second call preserves)
- resume_count increment across multiple dispatches
- Metadata survives pickle round-trip (cloudpickle proxy)
- Resume after pickle keeps the original stamp
- Interval-aware staleness math (`14 // 5 - 0 // 5 = 2`)
- Regression guard vs naive `(rollout_id - v)` formula
- `interval=1` reduces to raw delta
- `n_resumed` aggregator counts samples with `resume_count > 0`
- Partial-rollout preserves state; no-partial-rollout wipes state

All 10 pass against `lib/`.

## Result

Re-running job 31011 (next experiment) confirmed:
- `policy_staleness/mean` now reports non-zero values when samples span
  weight pushes (e.g., r5: 0.875 with interval=5).
- `resume_count/n_resumed` reports actual resumed-sample counts (e.g., r8: 8).

## Files of interest

- `lib/rollout_metrics.py` — the canonical wrapper with the fix
- `tests/test_metric_fix.py` — the 10 hermetic regression tests

The run script for job 30939 itself was effectively identical to 09's
(`run_swe.sh` in this dir), just with the pre-fix lib version. See SHA
above for the exact pre-fix state.
