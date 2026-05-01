#!/bin/bash
#
# Inner ray-job submitter for the Qwen3-235B-A22B-Thinking-2507 fully-async
# + Modal retool example. Invoked by run_async.sbatch on the Slurm head node
# after Ray is up across all 32 nodes (8 actor + 24 rollout).
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

# Qwen3-Thinking-2507 has rope_theta=5_000_000 (not the 1M default in the
# slime model script). Set BEFORE sourcing.
export MODEL_ARGS_ROTARY_BASE=5000000
source "${SLIME_ROOT}/scripts/models/qwen3-235B-A22B.sh"

# Pre-downloaded HF snapshots on /data (read-only). The FP8 path is what
# SGLang loads (FP8 inference); the BF16 torch_dist holds the Megatron actor
# weights (BF16 training).
HF_FP8=/data/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Thinking-2507-FP8/snapshots/f07f63f2bbd7540917118ebdf3812696ef303b03
CKPT_ROOT="${CKPT_ROOT:-/data/outputs/slime-checkpoints}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_FP8}"
   --ref-load "${CKPT_ROOT}/Qwen3-235B-A22B-Thinking-2507_torch_dist"
   --save "${CKPT_ROOT}/Qwen3-235B-A22B_slime/"
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

# 64 actor GPUs (8 nodes × 8): TP=4 PP=4 CP=2 EP=16 ETP=1 (Megatron's
# validated layout for Qwen3-235B). 4*4*2 = 32, so DP_attn = 64/32 = 2.
# Expert side: ETP*EP*PP = 1*16*4 = 64 — divides world_size cleanly.
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 4
   --context-parallel-size 2
   --expert-model-parallel-size 16
   --expert-tensor-parallel-size 1
   --decoder-last-pipeline-num-layers 22
   --sequence-parallel

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384

   # Qwen3-235B model config sets --moe-token-dispatcher-type alltoall.
   # SETUP.md quirk #9: --moe-enable-deepep would require flex dispatcher.
   # Sticking with alltoall (default for this model).
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

   # Crucial for fitting 235B BF16 + Adam state on 64 H200 GPUs.
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

# Pipelined RL: trainer runs 8 gradient steps between weight pushes. Samples
# consumed during steps K..K+7 all carry weight_version V_K, so max policy
# staleness per sample = 7 steps. The TIS off-policy correction (--use-tis,
# --tis-clip 2.0 above) absorbs the drift. With wait_time_ratio = 41% at
# interval=1, this should drop the trainer's wait window toward 0 by giving
# SGLang 8x the wall-clock to fill the buffer between pushes.
ASYNC_ARGS=(
   --update-weights-interval 8
   --keep-old-actor
)

# 192 rollout GPUs: TP=4 per engine = 48 inference engines (each holds 220 GB
# FP8 model across 4 H200s; ~55 GB per GPU plus KV cache). FP8 weight blocks
# are 128×128, so the per-rank shard of the MoE gate_proj output must be
# divisible by 128. With TP=8: 1536/8=192 → not divisible → SGLang dies with
# `output_size of gate's and up's weight = 192 is not divisible by weight
# quantization block_n = 128`. With TP=4: 1536/4=384 → 3 blocks ✓.
#
# SETUP.md quirk #8 (deepep) is avoided — sticking with the default a2a
# backend that the slime nightly image actually supports.
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
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
   # prompt-length instrumentation, and emits richer wandb metrics
   # (sandbox/*, prompt_len/*, throughput/*).
   --custom-generate-function-path add_slime_metrics.generate
   --custom-rm-path add_slime_metrics.reward_func
   --custom-rollout-log-function-path add_slime_metrics.custom_rollout_log_function
)

WANDB_ARGS=()
if [ -n "${WANDB_KEY}" ]; then
   WANDB_ARGS+=(
      --use-wandb
      --wandb-project slime-retool-async
      --wandb-group qwen3-235b-a22b-thinking-2507
      --wandb-key "${WANDB_KEY}"
   )
fi

MODAL_CONFIG_PATH="${MODAL_CONFIG_PATH:-${SLIME_ROOT}/.modal.toml}"
# The provided modal.toml uses [semianalysis] not [default] — point Modal
# SDK at that profile.
MODAL_PROFILE="${MODAL_PROFILE:-semianalysis}"

MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/root/Megatron-LM}"
RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${FULLY_ASYNC_DIR}:${SLIME_ROOT}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "MODAL_CONFIG_PATH": "${MODAL_CONFIG_PATH}",
    "MODAL_PROFILE": "${MODAL_PROFILE}",
    "SLIME_MODAL_APP": "${SLIME_MODAL_APP:-infx-slime-retool-sandbox}",
    "SLIME_MODAL_POOL_SIZE": "${SLIME_MODAL_POOL_SIZE:-128}",
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
   --actor-num-nodes 8 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 192 \
   --update-weight-buffer-size $(( 1024 * 1024 * 1024 * 4 )) \
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
   "${ASYNC_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"
