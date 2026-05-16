# 11 — SGLang DeepEP `auto` (low_latency) backend, EP=8, 16K response cap

**Status:** ⚠️ COMPLETED — DeepEP `low_latency` mode validated as ineffective
for our agentic-RL straggler tail. See `RESULTS.md` for details and the
verdict section.
**Lib SHA:** current (post-`a5878973`, plus JSONL metric dump)
**Dataset:** Lite sample_40
**Job:** 31395 (cancelled after 11 perfs, ~6h elapsed, no reward signal)
**Headline:** First test of SGLang's DeepEP MoE backend on the rollout side.
DeepEP genuinely sped up weight broadcast (35s vs 10's 43s), but the
`low_latency` dispatch mode pays a fixed RDMA setup cost per-call that's
amortized when the buffer is full (8 concurrent DP requests) and wasted
when the buffer is sparse (1 straggler sample). Our agentic rollouts
spend most of their time in the sparse tail → 8× per-token slowdown
specifically on the straggler → rollout_time exploded from ~500s to
~4000s per step after the first weight push.

## Hypothesis

The SWE-bench rollouts are gated on Qwen3-235B-Thinking's per-turn
generation rate (~30-40 tok/gpu/sec measured in 09/10). DeepEP's
intranode-NVLink + internode-IB hybrid all-to-all should beat SGLang's
default `alltoall` backend on the 128-expert × top-k 8 routing pattern.
The trainer side already saw a measurable win switching to DeepEP — the
rollout side should too.

If this helps, per-rollout wallclock drops and we get more steps in the
same budget. If it doesn't, we're confirming the bottleneck is elsewhere
(per-sample tail latency, not per-token throughput).

## What changed vs experiment 10

| Knob | 10 (base control) | 11 (this) |
|---|---|---|
| `--sglang-moe-a2a-backend` | (default: `alltoall`) | **`deepep`** |
| `--sglang-deepep-mode` | n/a | **`auto`** |
| `--sglang-ep-size` | 4 (effective; SGLang honored it under `alltoall`) | **8** (explicit; DeepEP forces this anyway) |
| `--sglang-cuda-graph-max-bs` | (default: 512) | **64** (DeepEP buffer doesn't fit larger captures) |
| `--rollout-max-response-len` | 32768 | **16384** (cap per-turn thinking) |

## Crash history

**First attempt (job 31352, CANCELLED after 1:26):** Hit DeepEP buffer overflow
during CUDA graph capture at `bs=512`:

```
RuntimeError: Failed: Assertion error /sgl-workspace/DeepEP/csrc/deep_ep.cpp:1105
  'x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank'
...
Exception: Capture cuda graph failed
```

deep_ep.cpp:1105 checks that the number of dispatched tokens per rank fits
in the pre-allocated low-latency buffer. SGLang's default `cuda_graph_max_bs=512`
tries to capture a CUDA graph at batch size 512, which exceeds DeepEP's
buffer cap. SGLang's own error message suggests `--cuda-graph-max-bs 16`.

Notably this is **NOT** a hidden-dim divisibility issue (line 1103 — the
adjacent check `x.size(1) % 128 == 0` — passes because the dispatch sees
the full `hidden_size=4096`, not the per-expert `moe_ffn_hidden_size=1536`,
and 4096 % 128 == 0). With `--sglang-enable-dp-attention` enabled, TP=8
shards only the attention layers; the FFN keeps each expert's 1536 intact
per rank, so EP=8 + Qwen3-235B's shapes line up cleanly.

**Fix applied for resubmit:** `--sglang-cuda-graph-max-bs 64` — matches our
total rollout batch size of 64 samples and fits well within DeepEP's
low-latency buffer cap. Also set `--sglang-ep-size 8` explicitly to match
what DeepEP would have used anyway (it overrides our 4 → 8 when `deepep`
backend is enabled, since DeepEP requires `ep_size` aligned with the
engine's `tp_size`).

## Hypothesis (preserved)
| JSONL metrics dump | not enabled | **enabled** via `SLIME_METRICS_JSONL` |
| Other flags | — | identical |

## New: structured metric dump

This is the first experiment to use the JSONL dump. `run_swe.sh` exports:

```bash
SLIME_METRICS_JSONL=${SLIME_ROOT}/mnt/logs/metrics-${SLURM_JOB_ID}.jsonl
```

The wrapper in `lib/rollout_metrics.py` monkey-patches
`slime.utils.logging_utils.log` so every call to it (5 chokepoints across
rollout + train + eval) appends one JSON line. Each line has:

- `step_key` — `"rollout/step"`, `"train/step"`, or `"eval/step"`
- `step` — int index along that axis
- `timestamp` — Unix epoch float
- all the metric key/value pairs from that call

After the run, instead of grepping the stdout log:

```bash
# raw_reward trajectory (rollout-side)
jq -r 'select(.step_key=="rollout/step") | "\(.step) \(."rollout/raw_reward")"' \
  mnt/logs/metrics-31???.jsonl

# train loss curve
jq -r 'select(.step_key=="train/step") | "\(.step) \(."train/loss")"' \
  mnt/logs/metrics-31???.jsonl
```

## Files

- `run_swe.sh` / `run_swe.sbatch` — frozen launcher snapshot
- `metrics.txt` — populated after the run lands (synthesized from JSONL)
- `artifacts/full-log-location.txt` — pointer to `mnt/logs/`
- `artifacts/metrics.jsonl` — symlink (or copy) of the JSONL once the run completes

## How to reproduce

```bash
sbatch 11-sglang-deepep-generator/run_swe.sbatch
```

Same path-finder + `LIB_DIR` resolution as all other experiments.
