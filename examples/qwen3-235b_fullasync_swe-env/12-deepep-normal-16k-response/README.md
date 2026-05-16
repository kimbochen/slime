# 12 — SGLang DeepEP `normal` mode, EP=8, 16K response cap

**Status:** ❌ DOA — cuda graph capture incompatible with `deepep normal` mode. 0 perfs in 59 min. See RESULTS.md.
**Lib SHA:** current (post-`a5878973`, plus JSONL metric dump via lib/patch_log_utils.py)
**Dataset:** Lite sample_40
**Headline:** Same as 11 but with `--sglang-deepep-mode normal` instead of `auto`.
Tests whether DeepEP `normal` mode avoids the straggler-tail collapse that
killed 11 while still keeping DeepEP's faster weight broadcast.

## What's different vs 11

| Knob | 11 (DeepEP auto/low_latency) | 12 (this) |
|---|---|---|
| `--sglang-deepep-mode` | `auto` (== `low_latency` for our decode batch sizes) | **`normal`** |
| everything else | identical | identical |

## Why this might work

In 11, DeepEP's `low_latency` mode caused the straggler problem:

- `low_latency` mode pre-allocates fixed-size NVSHMEM buffers and uses
  lock-free fast paths optimized for *many concurrent requests across
  many EP ranks*. Each dispatch is cheap **when the buffer is full**.
- In our agentic-RL workload, most of each rollout is the buffer-sparse
  tail: 1 straggler sample on an engine with 7 idle DP ranks. The
  per-dispatch RDMA setup cost is paid for a 1-token dispatch instead of
  amortized across 8 tokens.
- The 20-SM allocation for DeepEP communication (warned by SGLang at
  startup) compounds the problem — small amount of compute reserved
  for setup overhead that doesn't scale down when the workload is sparse.

`normal` mode:

- Uses dynamic buffer allocation (no fixed pre-allocation)
- Doesn't have the `num_max_dispatch_tokens_per_rank` constraint that
  caused 11's first crash (so we *might* be able to drop the
  `cuda-graph-max-bs=64` cap, but we keep it for safety)
- Trades peak throughput at large batches for lower per-call overhead
  at small batches — exactly the asymmetry the straggler tail needs

## Hypothesis

If `normal` mode fixes the straggler tail, we expect post-push steps
to look more like 10's p1-p4 (~500s mean) instead of 11's p7+ (~3500s mean).
Weight broadcast time should stay ~35s (same DeepEP infrastructure).

If `normal` mode also has the straggler tail → DeepEP itself isn't the
right backend for our workload, and we fall back to `alltoall` (experiment 13).

## JSONL metrics (now full coverage)

This is the first experiment to run with `lib/patch_log_utils.py` actively
intercepting the trainer's `logging_utils.log` calls (via sitecustomize.py).
The JSONL at `mnt/logs/metrics-${JOB}.jsonl` will contain:

- `step_key=rollout/step` from the rollout-side custom logger (sandbox metrics, tool calls, prompt len, etc.)
- `step_key=rollout/step` from the trainer-side `data.py:213` (raw_reward, advantages, kl, log_probs)
- `step_key=train/step` from the trainer-side `model.py:655` (train_loss, grad_norm, pg_loss, etc.)

Query examples after the run:

```bash
# raw_reward trajectory (from trainer)
jq -r 'select(.step_key=="rollout/step" and ."rollout/raw_reward" != null) | "\(.step) \(."rollout/raw_reward")"' \
   mnt/logs/metrics-${JOB}.jsonl

# train loss curve
jq -r 'select(.step_key=="train/step") | "\(.step) \(."train/loss")"' \
   mnt/logs/metrics-${JOB}.jsonl
```
