# DeepEP normal mode + EP=8 + 16K — job 31507 (DOA)

- Log: `/home/sa-shared/kimbo/slime/mnt/logs/infx-swe-31507.out`
- Wallclock: 59:12 elapsed before cancel
- Steps captured: 0
- Rollouts captured: 0

## What happened

`deepep_mode='normal'` confirmed in SGLang server_args. Everything else
identical to 11 (`auto`/`low_latency`). Result: **0 perfs in 59 minutes**.

Direct from the SGLang decode logs:

```
=== last 10 decode batches (no rollouts had completed) ===
  Decode batch, #running-req: 1, #token:  9288, cuda graph: False, gen throughput: 5.45 tok/s
  Decode batch, #running-req: 2, #token: 25042, cuda graph: False, gen throughput: 11.00 tok/s
  Decode batch, #running-req: 1, #token: 13165, cuda graph: False, gen throughput: 5.33 tok/s
  Decode batch, #running-req: 1, #token:  9442, cuda graph: False, gen throughput: 5.51 tok/s
  Decode batch, #running-req: 2, #token: 25202, cuda graph: False, gen throughput: 10.91 tok/s
  ...

=== cuda graph hit rate across all 306 decode batches: 0 / 306 = 0% ===
```

### Findings

1. **CUDA graph capture never happened** — `Capturing batches: 0` in the
   log. With `--sglang-moe-a2a-backend deepep --sglang-deepep-mode normal`,
   SGLang's MoE forward path through DeepEP's `normal` dispatcher is
   incompatible with the cuda graph capture pipeline. SGLang silently
   falls back to pure eager.

2. **Eager mode under DeepEP normal is much slower than eager mode under
   DeepEP low_latency:**
   ```
                            decode tok/s
     11 (low_latency, eager fallback):  22-35
     12 (normal, no graphs at all):      5-11
   ```
   The `low_latency` eager path still uses NVSHMEM RDMA for the dispatch
   (just without graph capture); `normal` mode's dispatcher has higher
   per-call overhead and no fast path equivalent.

3. **No samples completed in 59 minutes.** At ~5 tok/s on the common
   single-request decode case, a 14K-token response would need ~47 min
   per sample. 64 concurrent samples ≠ 64× speedup because the engines
   already serialize through single-request decode for most of the
   rollout tail. We projected 1-2 perfs in the 8h budget at best.

4. **Confirmed in SGLang server_args dump:**
   ```
   moe_a2a_backend='deepep'
   deepep_mode='normal'
   cuda_graph_max_bs=64
   cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]
   disable_cuda_graph=False
   ```
   So cuda graph was *enabled*, but never *captured*. That's the failure mode.

## Verdict

❌ **DeepEP `normal` mode is not viable for SGLang inference in this stack.**
The combination of normal-mode dispatcher + no cuda graph capture +
eager-only forward = 4-7× slowdown vs already-slow `low_latency` mode.
Not a usable production config for our workload.

## Implication for the experimental matrix

The DeepEP backend has effectively two operating modes, both flawed for
our agentic-RL workload:

```
deepep + auto/low_latency  (11)   → cuda graphs work, but straggler tail
                                    pathology (8× slowdown on lone samples)
                                    in the post-push phase
deepep + normal            (12)   → cuda graphs broken, pure eager,
                                    4-7× slower everywhere. DOA.
```

The right next experiment is therefore the original plan: plain SGLang
`alltoall` at EP=8 (experiment 13). If 13 works cleanly, the conclusion
is that DeepEP-the-backend is the wrong choice for our workload, and
EP=8 alone is fine. If 13 also has problems, EP=8-the-shape itself is
the issue and we should fall back to EP=4 (= experiment 10's setup).

## Aside: lib/patch_log_utils.py validation

Even though the run produced no trainer metrics (no training steps ran),
this experiment did validate that `lib/patch_log_utils.py` loads via
`lib/sitecustomize.py` in every Ray worker. The JSONL file was created
but stayed empty since no `logging_utils.log` calls fired before cancel.
