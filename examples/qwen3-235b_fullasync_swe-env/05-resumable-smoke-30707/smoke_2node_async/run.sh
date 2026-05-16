#!/bin/bash
#
# 2-node async SWE-bench smoke for VALIDATING the resumable agent path.
#
# Why this exists: the 12-node run (30707) ran the resumable variant
# through 20 perf steps without crashing, but we couldn't tell from logs
# whether the partial-rollout / sandbox-park / sandbox-resume code path
# actually fired (vs slime just dropping aborted samples and the agent
# silently regenerating from a fresh sandbox).
#
# Two unrelated bugs that were silently neutering partial-rollout:
#  1. rollout_metrics.py's generate-wrapper wiped sample.tokens etc.
#     before our agent could see them — so resume detection always saw
#     a "fresh" sample. FIXED: gated the wipe on `not args.partial_rollout`.
#  2. policy_version_at_dispatch got overwritten on every dispatch, so
#     the staleness metric always read 0. FIXED: sticky-on-first stamp.
#
# Plus added instrumentation:
#  • print() at pool.park / pool.acquire_or_resume so we can see what
#    fires in the log.
#  • sample.resume_count counter, surfaced as resume_count/{mean,max,...}
#    in the per-rollout metrics dict.
#
# Smaller-scale than the 12-node run so we can iterate faster.

set -ex

HEAD_IP="${1:-${MASTER_ADDR:-127.0.0.1}}"
export MASTER_ADDR="$HEAD_IP"
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi

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
   # no --save: smoke
)

DATA_ROOT="${DATA_ROOT:-${SLIME_ROOT}/mnt/data}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_ROOT}/swebench_verified/sample_10.jsonl}"

ROLLOUT_ARGS=(
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async

   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --rollout-shuffle
   --reward-key score

   # Small batch — we want resumes to fire, not a long run.
   --num-rollout "${NUM_ROLLOUT:-10}"
   --rollout-batch-size 4
   --n-samples-per-prompt 2
   --rollout-max-response-len 16384
   --rollout-temperature 1.0
   --global-batch-size 8
   --balance-data

   # The whole point of this smoke.
   --partial-rollout
)

# 1 actor node × 8 GPUs. Qwen3-4B is dense + small, so TP=2 DP=4.
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --sequence-parallel
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
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
   # No --optimizer-cpu-offload: Qwen3-4B Adam state is tiny.
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.85
   --sglang-context-length 32768
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path rollout_metrics.generate
   --custom-rm-path rollout_metrics.reward_func
   --custom-rollout-log-function-path rollout_metrics.custom_rollout_log_function
)

WANDB_ARGS=()
if [ -n "${WANDB_KEY}" ]; then
   WANDB_ARGS+=(
      --use-wandb
      --wandb-project slime-resume-smoke
      --wandb-group qwen3-4b-2node-async
      --wandb-key "${WANDB_KEY}"
   )
fi

MODAL_CONFIG_PATH="${MODAL_CONFIG_PATH:-${SLIME_ROOT}/.modal.toml}"

MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/root/Megatron-LM}"
RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${LIB_DIR}:${MEGATRON_LM_PATH}:${EXAMPLE_DIR}:${FULLY_ASYNC_DIR}:${SLIME_ROOT}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "MODAL_CONFIG_PATH": "${MODAL_CONFIG_PATH}",
    "SLIME_SWEBENCH_APP": "${SLIME_SWEBENCH_APP:-infx-slime-swebench-sandbox}",
    "SLIME_SWEBENCH_MAX_CONCURRENT": "${SLIME_SWEBENCH_MAX_CONCURRENT:-32}",
    "SLIME_SWEBENCH_SPAWN_QPS": "${SLIME_SWEBENCH_SPAWN_QPS:-10}",
    "SLIME_SWEBENCH_TIMEOUT": "${SLIME_SWEBENCH_TIMEOUT:-7200}",
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
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   --update-weight-buffer-size $(( 1024 * 1024 * 1024 * 2 )) \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"
