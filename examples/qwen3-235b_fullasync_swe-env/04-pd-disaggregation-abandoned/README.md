# 04 — PD (Prefill/Decode) disaggregation — ABANDONED

**Status:** ❌ abandoned — upstream Mooncake/SGLang KVTransferError blocker
**Lib SHA:** `cac0d666` (PD smoke ladder + 12-node attempts; both failed)
**Datasets tried:** Verified sample_10 (12-node), debian_slim (2-node smokes)
**Headline:** Three configuration-level fixes were tried (drop dp-attention,
add `--sglang-disaggregation-ib-device`, per-GPU IB JSON map). All failed
identically with `KVTransferError: Failed to get kvcache from prefill
instance`. Source-code review localised the failure to Mooncake's
`batch_transfer_sync` C++ binding returning non-zero on every per-request
KV chunk — below the SGLang Python layer, not reachable via flags.

After 12-node attempts on two slime images
(`nightly-dev-20260307a` and `nightly-dev-20260425a`) all failed identically,
**user decision: "officially giving up on PD"**.

Prefill/Decode-disaggregated rollout variant for the swe-env workload. **Blocked by an upstream Mooncake/SGLang PD bug** — see `../../../../../mnt/mooncake_bug_report/BUG_REPORT.md` for the full diagnosis.

## What's here

| file | purpose |
|---|---|
| `run_swe_pd.sh` / `.sbatch` | 64K-context swe-env config that layers `--sglang-config sglang_pd.yaml` on top of the canonical launcher — splits the 64-GPU rollout pool into 5 prefill engines × TP=8 + 3 decode engines × TP=8 |
| `sglang_pd.yaml` | the 5:3 prefill/decode layout |
| `mooncake_ib_per_gpu.json` | per-GPU HCA binding for Mooncake's RDMA transport (`{"0":"mlx5_0", …, "7":"mlx5_7"}`) — required to work around the comma-list parser bug in `sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py:get_ib_devices_for_gpu()` |

## Status (as of 2026-05-13)

Three configuration-level fix attempts have all failed identically:

1. ✗ Drop `--sglang-enable-dp-attention` — DP-attention isn't the cause
2. ✗ Add `--sglang-disaggregation-ib-device` — same fallback chain in code, no-op
3. ✗ Per-GPU JSON IB map (this dir's `mooncake_ib_per_gpu.json`) — fixes a real comma-list parser bug, but doesn't unblock the actual KV transfer

Source-code review localised the failure to Mooncake's `batch_transfer_sync` C++ binding returning non-zero on every per-request KV chunk. This is below the SGLang Python layer and not reachable via flags. Tested across two slime images (`nightly-dev-20260307a` and `nightly-dev-20260425a`) — both fail identically, so it's not a recent regression.

## Submitting (if/when upstream PD bug resolves)

```bash
sbatch examples/qwen3-235b_fullasync_swe-env/pd/run_swe_pd.sbatch
```

Until then, use the non-PD launcher in the parent directory.
