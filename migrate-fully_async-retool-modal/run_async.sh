#!/bin/bash
#
# Inner ray-job submitter for the GLM-4.5 106B-A12B fully-async + Modal
# retool example. Invoked by run_async.sbatch on the Slurm head node, after
# Ray has been brought up across all 12 nodes.
#
# Argument $1 is the Ray head node IP (forwarded from sbatch).

set -ex

HEAD_IP="${1:-${MASTER_ADDR:-127.0.0.1}}"
export MASTER_ADDR="$HEAD_IP"
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
FULLY_ASYNC_DIR="${SLIME_ROOT}/examples/fully_async"

source "${SLIME_ROOT}/scripts/models/glm4.5-106B-A12B.sh"

CKPT_ROOT="${CKPT_ROOT:-${SLIME_ROOT}/mnt/checkpoints}"
CKPT_ARGS=(
   --hf-checkpoint "${CKPT_ROOT}/zai-org/GLM-4.5-Air"
   --ref-load "${CKPT_ROOT}/zai-org/GLM-4.5-Air_torch_dist"
   --save "${CKPT_ROOT}/GLM-4.5_slime/"
   --save-interval 50
)

DATA_ROOT="${DATA_ROOT:-${SLIME_ROOT}/mnt/data}"
ROLLOUT_ARGS=(
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async

   --prompt-data "${DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --reward-key score

   --num-rollout "${NUM_ROLLOUT:-3000}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-32}"
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 50
   --eval-prompt-data aime "${DATA_ROOT}/aime-2024/aime-2024.jsonl"
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

# 32 actor GPUs (4 nodes x 8): TP=4, EP=8, PP=1, CP=1, DP=1.
# 106B BF16 weights ~212 GB; with optimizer state and activation recompute we
# fit comfortably under 32 x 141 GB H200 HBM.
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --sequence-parallel

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096

   # NOTE: --moe-enable-deepep was tried earlier but Megatron rejects it
   # with "DeepEP backend is only supported with flex token dispatcher."
   # The glm4.5-106B-A12B model config sets --moe-token-dispatcher-type
   # alltoall, which is incompatible. Sticking with alltoall (default for
   # this model).
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# 64 rollout GPUs (8 nodes x 8): one TP=8 SGLang engine per rollout node.
# Note: --sglang-moe-a2a-backend deepep --sglang-deepep-mode auto was tried
# first but hit `assert False, "forward_deepgemm_masked is deprecated"` in
# sglang/srt/layers/moe/ep_moe/layer.py:242 during warmup. Falling back to
# the default a2a backend.
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.85
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   # add_slime_metrics wraps generate_with_retool with sandbox-time and
   # prompt-length instrumentation, and emits a richer set of wandb metrics
   # (sandbox/*, prompt_len/*, throughput/*) via custom_rollout_log_function.
   --custom-generate-function-path add_slime_metrics.generate
   --custom-rm-path add_slime_metrics.reward_func
   --custom-rollout-log-function-path add_slime_metrics.custom_rollout_log_function
)

WANDB_ARGS=()
if [ -n "${WANDB_KEY}" ]; then
   WANDB_ARGS+=(
      --use-wandb
      --wandb-project slime-retool-async
      --wandb-group glm45-106b-a12b
      --wandb-key "${WANDB_KEY}"
   )
fi

# Modal credentials live in $SLIME_ROOT/.modal.toml; point the SDK at it
# so the rollout workers can spawn sandboxes.
MODAL_CONFIG_PATH="${MODAL_CONFIG_PATH:-${SLIME_ROOT}/.modal.toml}"

MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/root/Megatron-LM}"
RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${FULLY_ASYNC_DIR}:${SLIME_ROOT}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "MODAL_CONFIG_PATH": "${MODAL_CONFIG_PATH}",
    "SLIME_MODAL_APP": "${SLIME_MODAL_APP:-infx-slime-retool-sandbox}",
    "SLIME_MODAL_POOL_SIZE": "${SLIME_MODAL_POOL_SIZE:-64}",
    "SLIME_MODAL_PER_EXEC_TIMEOUT": "${SLIME_MODAL_PER_EXEC_TIMEOUT:-60}",

    "OTEL_SDK_DISABLED": "true",
    "OTEL_TRACES_EXPORTER": "none",
    "OTEL_METRICS_EXPORTER": "none",
    "OTEL_LOGS_EXPORTER": "none",

    "PYTORCH_ALLOC_CONF": "expandable_segments:True"
  }
}
EOF
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 4 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 64 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"
