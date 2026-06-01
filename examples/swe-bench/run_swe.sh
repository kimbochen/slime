#!/bin/bash
#
# Inner ray-job submitter for the barebones SWE-bench example.
#
# Differences from the reference (qwen3-235b_fullasync_swe-env, exp 15):
#   • Custom modules at examples/swe-bench/* instead of lib/:
#       --custom-generate-function-path generate_with_openhands.generate
#       --custom-rm-path                 reward.compute_reward
#       --custom-rollout-log-function-path metrics.log_rollout_data
#   • --partial_rollout + --use-tis enabled (our generate.py supports resume).
#     Reference disables both because its agent rejects partial rollout.
#   • Tool specs in tool_specs.py; --apply-chat-template + --tool-key tools
#     handle templating at data-load time. gen_prompt_data.py needs to emit
#     a "tools" field per row (TOOL_SPECS copied in).
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
EXAMPLE_DIR="$SCRIPT_DIR"
SLIME_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." &>/dev/null && pwd)"
FULLY_ASYNC_DIR="${SLIME_ROOT}/examples/fully_async"

# Qwen3-235B-A22B-Thinking-2507 uses rope_theta=5000000.
export MODEL_ARGS_ROTARY_BASE=5000000
source "${SLIME_ROOT}/scripts/models/qwen3-235B-A22B.sh"

# -------------------------------------------------------------------- ckpts --
CKPT_ROOT="${CKPT_ROOT:-${SLIME_ROOT}/mnt/checkpoints}"
CKPT_ARGS=(
   --hf-checkpoint "${CKPT_ROOT}/Qwen3-235B-A22B-Thinking-2507-FP8"
   --ref-load "${CKPT_ROOT}/Qwen3-235B-A22B_torch_dist"
   # No --save / --save-interval intentionally. Slime's save_model gathers
   # all 8×~180GB sharded weights to CPU (~1.4TB) and OOMs even on H200
   # nodes. should_run_periodic_action() in slime/utils/misc.py:73 hardcodes
   # "save at rollout_id == num_rollout - 1" regardless of save_interval —
   # the ONLY way to disable saves is to omit both --save and --save-interval
   # (interval=None short-circuits at the top of the function). Re-add both
   # only if you have ~1.5TB CPU RAM per actor node and want checkpoints.
)

# -------------------------------------------------------------------- data --
DATA_ROOT="${DATA_ROOT:-${SLIME_ROOT}/mnt/data}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_ROOT}/swebench_lite/sample_40.jsonl}"

ROLLOUT_ARGS=(
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async

   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle
   # NOTE: NOT using --reward-key. Slime's Sample.get_reward_value does
   # sample.reward[args.reward_key] when reward_key is set, but our
   # compute_reward returns a float (per slime's documented signature),
   # not a dict. Passing --reward-key would crash with TypeError after
   # the first rollout completes.
   # NOTE: NOT using --apply-chat-template or --tool-key. generate.py
   # renders the chat template itself via tokenizer.apply_chat_template(
   # messages, tools=TOOL_SPECS, ...) so the tool list lives with the
   # generation code, matching the qwen3-235b reference's pattern.

   --num-rollout "${NUM_ROLLOUT:-200}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}"
   --n-samples-per-prompt 8    # Reverted from 16: the 16-sample variant
                                # consistently triggered SGLang engine
                                # instability (one TP worker dying mid-forward,
                                # cascading to gloo "Connection closed" → next
                                # update_weights deadlock). 8-sample groups
                                # are the proven-stable point. Trade-off:
                                # more zero-std groups (~75-100%) so less
                                # gradient signal per step.
   --rollout-max-response-len 16384
   --rollout-max-context-len 65536  # Bumped from 32768 to use the trainer's
                                     # full effective context cap (CP=4 ×
                                     # max_tokens_per_gpu=16384 = 64K). Lets
                                     # samples that were hitting the 32K cap
                                     # generate longer multi-turn trajectories.
                                     # REQUIRED by our generate.py (asserts).
   --rollout-temperature 1.0

   # Partial rollout + TIS: our generate.py supports resume across aborts
   # (sample.metadata["sandbox_id"] persists the Modal container); TIS
   # corrects for the staleness rather than masking the off-policy prefix.
   --partial_rollout
   --use-tis
   --tis-clip 2.0
   --tis-clip-low 0

   --global-batch-size 64    # Must = rollout_batch_size × n_samples_per_prompt
                              # (= 8 × 8) so trainer consumes all rollout data.
   --balance-data
)

EVAL_ARGS=()   # bring-up: training set IS the eval; enable later if needed

# ----------------------------------------------------------- trainer perf --
# 4 actor nodes × 8 GPUs = 32 GPUs. TP4 × PP2 × CP4 × EP8 ETP1.
# Effective trainer context cap = CP × max_tokens_per_gpu = 4 × 16384 = 64 K.
#
# Using exp 13's proven-stable parallelism (PP=2 CP=4) instead of exp 15's
# PP=4 CP=2. Per user's prior exp notes, the PP=4 CP=2 reshape is "SLOWER
# than 13 AND less reliable" — exp 14, 15, and our 36615 all hit gloo
# recv timeouts at the 2-hr mark with PP=4 CP=2.
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 2
   --context-parallel-size 4
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --sequence-parallel
   # No --decoder-last-pipeline-num-layers: PP=2 splits 94 layers evenly
   # (47/47), no manual override needed (unlike PP=4 which needs 24/24/24/22).

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384

   --moe-token-dispatcher-type flex
   --moe-flex-dispatcher-backend deepep
)

# GSPO (slime's recommended recipe for Qwen3-235B).
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

# ------------------------------------------------------------------ rollout --
# 8 rollout nodes × 8 GPUs = 64 GPUs. 8 SGLang engines TP=8, DP attention.
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.85
   --sglang-enable-dp-attention
   --sglang-dp-size 8
   --sglang-ep-size 8
   --sglang-context-length 65536    # comfortable above trainer's 32 K cap
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ------------------------------------------------------------------ custom --
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_openhands.generate
   --custom-rm-path                 reward.compute_reward
   --custom-rollout-log-function-path metrics.log_rollout_data
)

# ------------------------------------------------------------------- wandb --
WANDB_ARGS=()
if [ -n "${WANDB_KEY}" ]; then
   WANDB_ARGS+=(
      --use-wandb
      --wandb-project slime-swebench-barebones
      --wandb-group qwen3-235B-thinking-swe
      --wandb-key "${WANDB_KEY}"
   )
fi

# ----------------------------------------------------------------- env vars --
# Modal credentials for rollout workers to spawn sandboxes.
MODAL_CONFIG_PATH="${MODAL_CONFIG_PATH:-${SLIME_ROOT}/.modal.toml}"

# JSONL sidecar — log_patch auto-loaded via sitecustomize.py appends
# every metric call (rollout, perf, train, eval) to this path. Unset → no-op.
SLIME_METRICS_JSONL="${SLIME_METRICS_JSONL:-${SLIME_ROOT}/examples/swe-bench/results/metrics-${SLURM_JOB_ID:-manual}.jsonl}"
mkdir -p "$(dirname "${SLIME_METRICS_JSONL}")"

# Per-sample reward-fail JSONL — reward.py appends one line per non-1.0
# sample with {instance_id, category, reason}. Lets us see WHY a bucket
# fired (e.g. the actual `git apply` rejection message) which the bucketed
# metrics in SLIME_METRICS_JSONL discard. Unset → no-op.
SLIME_REWARD_FAIL_JSONL="${SLIME_REWARD_FAIL_JSONL:-${SLIME_ROOT}/examples/swe-bench/results/reward-fails-${SLURM_JOB_ID:-manual}.jsonl}"
mkdir -p "$(dirname "${SLIME_REWARD_FAIL_JSONL}")"

MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/root/Megatron-LM}"

# EXAMPLE_DIR must be on PYTHONPATH (a) so Ray actors find our generate/
# reward/metrics/sandbox modules, and (b) so Python's site machinery finds
# sitecustomize.py for the JSONL patch.
RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${EXAMPLE_DIR}/telemetry:${EXAMPLE_DIR}:${MEGATRON_LM_PATH}:${FULLY_ASYNC_DIR}:${SLIME_ROOT}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "MODAL_CONFIG_PATH": "${MODAL_CONFIG_PATH}",
    "SLIME_METRICS_JSONL": "${SLIME_METRICS_JSONL}",
    "SLIME_REWARD_FAIL_JSONL": "${SLIME_REWARD_FAIL_JSONL}",
    "SLIME_SWEBENCH_APP": "${SLIME_SWEBENCH_APP:-slime-swebench-sandbox}",
    "SLIME_SWEBENCH_TIMEOUT": "${SLIME_SWEBENCH_TIMEOUT:-1800}",
    "SLIME_SWEBENCH_PER_CMD_TIMEOUT": "${SLIME_SWEBENCH_PER_CMD_TIMEOUT:-120}",
    "SLIME_SWEBENCH_MAX_OUTPUT_BYTES": "${SLIME_SWEBENCH_MAX_OUTPUT_BYTES:-65536}",
    "SLIME_SWEBENCH_CPU": "${SLIME_SWEBENCH_CPU:-2.0}",
    "SLIME_SWEBENCH_MEMORY_MB": "${SLIME_SWEBENCH_MEMORY_MB:-8192}",
    "SLIME_SWEBENCH_EVAL_TIMEOUT": "${SLIME_SWEBENCH_EVAL_TIMEOUT:-1500}",
    "SLIME_SWEBENCH_WARN_CONCURRENT": "${SLIME_SWEBENCH_WARN_CONCURRENT:-128}",

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
   --update-weights-interval 5 \
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
