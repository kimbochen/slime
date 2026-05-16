#!/bin/bash
#
# 2-node PD smoke test. Validates the upstream-aligned PD config shape on
# our hardware/image WITHOUT occupying the 12-node pool the main
# Qwen3-235B SWE run uses (the spare 2 idle nodes get exercised in parallel).
#
# Uses SYNCHRONOUS train.py + --colocate, matching upstream slime's only
# working PD reference (tests/test_glm4.7_30B_A3B_pd_mooncake.py). PD is
# purely an inference-side property of the rollout engines (SGLang) — the
# trainer doesn't know or care about prefill/decode disaggregation. So a
# validated PD config from train.py transfers unchanged to train_async.py.
# (train_async.py rejects --colocate with "Colocation is not supported for
# async training.", which is why we can't use the async path here.)
#
# Differences from the canonical 12-node SWE PD job:
#   • Driver: train.py (sync) instead of train_async.py
#   • Model: Qwen3-4B (dense, 4B) — small enough that 8 actors × ~10 GB
#     of Adam state on host fits comfortably (vs GLM-4.5-Air 106B which
#     blew out the 1.4 TB host RAM at 8 actors × 190 GB each).
#   • Dataset: dapo-math-17k (single-turn) — fastest path to a real rollout
#     batch, no Modal sandboxes.
#   • --colocate: trainer + rollout share the same 16 GPUs.
#   • PD layout: 1 prefill (TP=8) + 1 decode (TP=8) on 16 GPUs total.
#
# What this smoke validates:
#   ✓ Per-engine `overrides: {disaggregation_transfer_backend: mooncake}`
#     wiring through slime → sglang
#   ✓ Long --sglang-watchdog-timeout / --sglang-router-request-timeout
#   ✓ Explicit --sglang-disaggregation-transfer-backend flag
#   ✓ Mooncake auto-discovery of IB devices (no manual JSON map)
#   ✓ Prefill↔decode KV pairing across nodes via IB
#
# If green here, the same SGLANG_ARGS block transfers verbatim to
# train_async.py for the 12-node 30439 upstream-PD test.

set -ex

HEAD_IP="${1:-${MASTER_ADDR:-127.0.0.1}}"
export MASTER_ADDR="$HEAD_IP"
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Walk up until we find the example root (named directory), then derive
# SLIME_ROOT. Robust to running from any depth under experiments/.
EXAMPLE_DIR=""
_d="$SCRIPT_DIR"
while [ "$_d" != "/" ]; do
   if [ "$(basename "$_d")" = "qwen3-235b_fullasync_swe-env" ]; then
      EXAMPLE_DIR="$_d"; break
   fi
   _d="$(dirname "$_d")"
done
LIB_DIR="${EXAMPLE_DIR}/lib"
SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." &>/dev/null && pwd)"
FULLY_ASYNC_DIR="${SLIME_ROOT}/examples/fully_async"

source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"

CKPT_ROOT="${CKPT_ROOT:-${SLIME_ROOT}/mnt/checkpoints}"
CKPT_ARGS=(
   --hf-checkpoint "${CKPT_ROOT}/Qwen/Qwen3-4B"
   --ref-load "${CKPT_ROOT}/Qwen/Qwen3-4B_torch_dist"
   # no --save: smoke run, results go in logs only
)

DATA_ROOT="${DATA_ROOT:-${SLIME_ROOT}/mnt/data}"
ROLLOUT_ARGS=(
   # No --rollout-function-path: sync train.py uses the default
   # slime.rollout.sglang_rollout.generate_rollout, which is the codepath
   # upstream's PD test exercises.

   --prompt-data "${DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   # No --reward-key here: deepscaler RM returns an int (0/1), not a dict.
   # Setting --reward-key score made slime's metric pipeline try to subscript
   # an int and crash with "TypeError: 'int' object is not subscriptable"
   # (sglang_rollout types.py:147). Killed smoke 30658 right after r0.
   --rm-type deepscaler

   # Small batch — we just want a handful of rollouts through PD to verify
   # the handshake works, not to converge anything.
   --num-rollout "${NUM_ROLLOUT:-5}"
   --rollout-batch-size 4
   --n-samples-per-prompt 2
   --rollout-max-response-len 2048
   --rollout-temperature 1.0

   --global-batch-size 8
   --balance-data
)

# 16-GPU colocated trainer. Qwen3-4B is dense (no MoE), so EP/ETP are 1.
# TP=2 × PP=1 × CP=1 = 2; with 16 GPUs that gives DP=8 (data parallelism
# across 8 model replicas). Plenty of headroom for the 4B model.
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --sequence-parallel
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   # No --optimizer-cpu-offload: Qwen3-4B Adam state is tiny (~64 GB total
   # / 16 ranks = 4 GB/rank). Keeping it on GPU avoids the host-RAM OOM
   # that killed the GLM-4.5-Air attempt (8 actors × 190 GB/actor exceeded
   # the 1.4 TB node capacity).
)

# Upstream-aligned PD knobs — the four load-bearing differences from our
# earlier (failed) PD attempts:
#   1. Per-engine `overrides: {disaggregation_transfer_backend: mooncake}`
#      in sglang_pd.yaml.
#   2. Explicit --sglang-disaggregation-transfer-backend mooncake CLI flag
#      (belt-and-suspenders).
#   3. Long watchdog/router timeouts (default 60s is too short for
#      multi-turn / heavy decode).
#   4. NO --sglang-mooncake-ib-device — let mooncake auto-discover IB HCAs.
SGLANG_ARGS=(
   --sglang-config "${SCRIPT_DIR}/sglang_pd.yaml"
   # 5 prefill + 3 decode, TP=2 each → 8 engines × 2 GPUs = 16 GPUs total.
   # Exact ratio match to the 12-node pd/run_swe_pd_upstream.yaml, just
   # at 4× smaller TP.
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.85
   --sglang-disaggregation-transfer-backend mooncake
   --sglang-watchdog-timeout 1200
   --sglang-router-request-timeout-secs 1200
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

WANDB_ARGS=()
if [ -n "${WANDB_KEY}" ]; then
   WANDB_ARGS+=(
      --use-wandb
      --wandb-project slime-pd-smoke
      --wandb-group qwen3-4b-2node-5p3d
      --wandb-key "${WANDB_KEY}"
   )
fi

MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/root/Megatron-LM}"
# NOTE: NO PYTORCH_*_ALLOC_CONF=expandable_segments:True here. With
# --colocate, sglang uses TorchMemorySaver to swap GPU memory between
# trainer and rollout phases. TorchMemorySaver does NOT yet support
# PyTorch's expandable_segments allocator — setting it produces a
# deterministic startup error on every DP rank ("TorchMemorySaver is
# disabled for the current process because expandable_segments is not
# supported yet."). The main non-colocated run_swe.sh keeps them.
RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${LIB_DIR}:${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${SLIME_ROOT}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",

    "OTEL_SDK_DISABLED": "true",
    "OTEL_TRACES_EXPORTER": "none",
    "OTEL_METRICS_EXPORTER": "none",
    "OTEL_LOGS_EXPORTER": "none"
  }
}
EOF
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 2 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 16 \
   --colocate \
   --update-weight-buffer-size $(( 1024 * 1024 * 1024 * 4 )) \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
