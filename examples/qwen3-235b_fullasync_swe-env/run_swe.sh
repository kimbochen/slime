#!/bin/bash
#
# Inner ray-job submitter for the Qwen3-235B-Thinking SWE-bench-Verified
# agentic-coding training run. Same 12-node layout as the retool example
# (4 actor + 8 rollout, fully-async via train_async.py), but:
#
#   • Sandbox: examples/qwen3-235b_fullasync_swe-env/coding_sandbox.py
#     (per-instance Epoch AI Docker images, not a generic debian_slim)
#   • Agent loop: generate_with_codingagent.generate (six tools:
#     run_command/read_file/write_file/apply_patch/run_tests/submit)
#   • Reward: reward_func runs the instance's /eval.sh inside the same
#     sandbox the agent used, returns 1.0 on green, 0.0 on red.
#   • Prompt-data: 10-instance SWE-bench-Verified slice (sample_10.jsonl);
#     swap to the full 500-instance set once bring-up is green.
#
# Argument $1 is the Ray head node IP (forwarded from sbatch).

set -ex

HEAD_IP="${1:-${MASTER_ADDR:-127.0.0.1}}"
export MASTER_ADDR="$HEAD_IP"
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
FULLY_ASYNC_DIR="${SLIME_ROOT}/examples/fully_async"

# Qwen3-235B-A22B-Thinking-2507 uses rope_theta=5000000.
export MODEL_ARGS_ROTARY_BASE=5000000

source "${SLIME_ROOT}/scripts/models/qwen3-235B-A22B.sh"

CKPT_ROOT="${CKPT_ROOT:-${SLIME_ROOT}/mnt/checkpoints}"
CKPT_ARGS=(
   --hf-checkpoint "${CKPT_ROOT}/Qwen3-235B-A22B-Thinking-2507-FP8"
   --ref-load "${CKPT_ROOT}/Qwen3-235B-A22B_torch_dist"
   --save "${CKPT_ROOT}/Qwen3-235B-A22B-swe_slime/"
   --save-interval 25
)

DATA_ROOT="${DATA_ROOT:-${SLIME_ROOT}/mnt/data}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_ROOT}/swebench_verified/sample_10.jsonl}"

# SWE-bench rollouts are heavy (multi-turn, long observations); smaller
# rollout-batch + fewer samples per prompt to keep step time tractable.
ROLLOUT_ARGS=(
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async

   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --rollout-shuffle
   --reward-key score

   --num-rollout "${NUM_ROLLOUT:-200}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-4}"
   --n-samples-per-prompt 4
   --rollout-max-response-len 32768
   --rollout-temperature 1.0

   # global = rollout_batch_size × n_samples_per_prompt
   --global-batch-size 16
   --balance-data
)

# Eval disabled for the bring-up — the 10-instance training set IS the eval.
EVAL_ARGS=()

# Identical trainer parallelism to the retool variant: TP=4 PP=4 CP=2 EP=8
# ETP=1 + CPU offload. Memory math doesn't change with the new env.
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 4
   --context-parallel-size 2
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --sequence-parallel
   --decoder-last-pipeline-num-layers 22
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator gspo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 4e-4
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.85
   --sglang-enable-dp-attention
   --sglang-dp-size 8
   --sglang-ep-size 4
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_codingagent.generate
   --custom-rm-path generate_with_codingagent.reward_func
)

WANDB_ARGS=()
if [ -n "${WANDB_KEY}" ]; then
   WANDB_ARGS+=(
      --use-wandb
      --wandb-project slime-swe-env
      --wandb-group qwen3-235B-thinking-swe
      --wandb-key "${WANDB_KEY}"
   )
fi

MODAL_CONFIG_PATH="${MODAL_CONFIG_PATH:-${SLIME_ROOT}/.modal.toml}"

MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/root/Megatron-LM}"
RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${FULLY_ASYNC_DIR}:${SLIME_ROOT}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "MODAL_CONFIG_PATH": "${MODAL_CONFIG_PATH}",
    "SLIME_SWEBENCH_APP": "${SLIME_SWEBENCH_APP:-infx-slime-swebench-sandbox}",
    "SLIME_SWEBENCH_MAX_CONCURRENT": "${SLIME_SWEBENCH_MAX_CONCURRENT:-32}",
    "SLIME_SWEBENCH_SPAWN_QPS": "${SLIME_SWEBENCH_SPAWN_QPS:-4}",
    "SLIME_SWEBENCH_TIMEOUT": "${SLIME_SWEBENCH_TIMEOUT:-1800}",
    "SLIME_SWEBENCH_PER_CMD_TIMEOUT": "${SLIME_SWEBENCH_PER_CMD_TIMEOUT:-120}",
    "SLIME_SWEBENCH_MAX_TURNS": "${SLIME_SWEBENCH_MAX_TURNS:-30}",

    "OTEL_SDK_DISABLED": "true",
    "OTEL_TRACES_EXPORTER": "none",
    "OTEL_METRICS_EXPORTER": "none",
    "OTEL_LOGS_EXPORTER": "none",

    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
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
   "${CUSTOM_ARGS[@]}"
