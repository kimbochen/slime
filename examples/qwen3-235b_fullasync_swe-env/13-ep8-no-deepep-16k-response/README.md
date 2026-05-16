# 13 — EP=8 with default alltoall (no DeepEP), 16K response cap

**Status:** ✅ COMPLETED — TIMEOUT at 8h, 9 perfs (p0-p8). Confirms tail is workload-fundamental, not DeepEP-specific. See RESULTS.md for cross-experiment verdict.
**Lib SHA:** current (post-`a5878973`, with JSONL metric dump)
**Dataset:** Lite sample_40
**Headline:** Isolates the EP=8-alone variable. Same as 11/12 (EP=8 +
16K response cap) but with the default SGLang `alltoall` MoE backend
instead of DeepEP. Run if 12 (DeepEP normal mode) still shows the
straggler tail — confirms whether DeepEP-the-backend is the problem
or EP=8-the-shape.

## Matrix view of the relevant experiments

| # | ep_size | moe backend | deepep_mode | max_response | result |
|---:|---:|---|---|---:|---|
| 10 | 4 | alltoall (default) | — | 32K | TIMEOUT 8h, 10 perfs, monolithic slow tails |
| 11 | 8 | **DeepEP** | `auto` (low_latency) | **16K** | CANCELLED at 11 perfs — straggler tail collapse (8× slowdown on longest sample) |
| 12 | 8 | **DeepEP** | **`normal`** | **16K** | (running — does normal mode avoid the low_latency tail?) |
| **13** | **8** | alltoall (default) | — | **16K** | (this — to be run if 12 also tails) |

If 12 fixes the tail → DeepEP backend is fine, `low_latency` mode is the
problem. 13 may not be needed.

If 12 also tails → DeepEP backend itself doesn't suit our straggler-heavy
workload. 13 isolates: does plain `alltoall` work at EP=8, or is EP=8
inherently the problem regardless of backend?

## Configuration deltas vs 10 and 11

| Knob | 10 (baseline) | 11 (DeepEP) | 12 (this) |
|---|---|---|---|
| `--sglang-ep-size` | 4 | 8 (DeepEP forces) | **8 (explicit)** |
| `--sglang-moe-a2a-backend` | (alltoall, default) | `deepep` | (alltoall, default) |
| `--sglang-deepep-mode` | — | `auto` | — |
| `--sglang-cuda-graph-max-bs` | (512, default) | 64 (needed for DeepEP buffer) | (512, default) |
| `--rollout-max-response-len` | 32768 | 16384 | **16384** |
| `--partial-rollout` | OFF | OFF | OFF |
| `--update-weights-interval` | 5 | 5 | 5 |

## Hypothesis

11's early data showed ~45% of decode batches falling back to eager mode
(no cuda graph), at 8–19 tok/s vs 32–66 tok/s under cuda graph. That
fallback is what's killing per-rollout time. Two possible causes:
1. **DeepEP-specific**: DeepEP's runtime conditions disable cuda graphs
   for some batch shapes, and we'd recover throughput by going back to
   alltoall.
2. **EP=8-specific**: with EP=8, the MoE dispatch hits more all-to-all
   communication overhead than EP=4, regardless of backend.

12 distinguishes these: if alltoall-EP=8 runs at 10's speed, cause (1)
is right. If it runs at 11's speed, cause (2) is right.

## How to launch

```bash
sbatch 12-ep8-no-deepep-16k-response/run_swe.sbatch
```

Same path-finder + `LIB_DIR` resolution as all other experiments.
Inherits the JSONL metric dump via `SLIME_METRICS_JSONL`.

## Notes

- We're picking 16K (not 32K) for the response cap because 10's tail-latency
  problem at 32K is the original motivation for the cap reduction. Holding
  16K constant across 11 and 12 isolates the EP/backend variables.
- If 12 *also* shows slow rollouts, the conclusion is that **per-sample
  agentic wall-clock cost dominates regardless of inference-backend tuning**
  — same conclusion as 09/10 but more rigorously isolated.
