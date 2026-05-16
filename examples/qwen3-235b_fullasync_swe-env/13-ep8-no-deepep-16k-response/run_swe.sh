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
#   • Prompt-data: 40-instance SWE-bench-Lite slice (sample_40.jsonl, 4 per
#     repo × 10 repos). Override PROMPT_DATA=... to point at Verified or
#     a larger slice. Lite is easier so per-rollout reward signal is denser
#     (more groups with non-zero variance), which matters for GRPO learning.
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
PROMPT_DATA="${PROMPT_DATA:-${DATA_ROOT}/swebench_lite/sample_40.jsonl}"

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
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}"
   --n-samples-per-prompt 8
   # Halved from 32K to cap per-turn thinking-token blowup. Multi-turn
   # responses still accumulate but each individual SGLang call is bounded.
   --rollout-max-response-len 16384
   --rollout-temperature 1.0

   # global = rollout_batch_size × n_samples_per_prompt
   --global-batch-size 64
   --balance-data

   # NOT setting --partial-rollout: the SWE-bench agent in
   # generate_with_codingagent.py has an explicit assert that rejects it
   # (line 290 — "Partial rollout not supported"). The agent state machine
   # (tool-call loop with sandbox state) can't safely resume mid-trajectory.
   # Pipeline-RL behavior would require modifying the agent to checkpoint
   # and restore mid-call state.
)

# Eval disabled for the bring-up — the 10-instance training set IS the eval.
EVAL_ARGS=()

# Identical trainer parallelism to the retool variant: TP=4 PP=4 CP=2 EP=8
# ETP=1 + CPU offload. Memory math doesn't change with the new env.
PERF_ARGS=(
   --tensor-model-parallel-size 4
   # PP=2 (was 4) + CP=4 (was 2) keeps world=32 unchanged; reshapes the
   # parallelism to give context-parallel more ranks to spread activations
   # across. Effective context cap = CP × max_tokens_per_gpu = 4 × 16384 = 64K.
   # Per-PP-stage layer count goes from ~23 to ~47, so per-rank weight/grad
   # footprint grows ~30 GB before Adam offload — still fits with CPU
   # offload + the ~110 GB H200 headroom.
   --pipeline-model-parallel-size 2
   --context-parallel-size 4
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --sequence-parallel
   # No --decoder-last-pipeline-num-layers: PP=2 splits evenly (47 / 47),
   # the uneven-split tuning from PP=4 doesn't apply.
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384

   # Override the model script's default --moe-token-dispatcher-type alltoall
   # with flex (required by the deepep backend below). DeepEP is Megatron's
   # high-throughput MoE all-to-all backend; faster than alltoall on H200 for
   # the 128-expert + topk=8 Qwen3-235B routing pattern. Argparse later-wins
   # means these override the values in scripts/models/qwen3-235B-A22B.sh.
   # NOTE: --moe-enable-deepep is the OLD spelling, deprecated by Megatron in
   # favour of --moe-flex-dispatcher-backend=deepep (which is what enable-deepep
   # auto-sets under the hood). Using the new name to silence the warning.
   --moe-token-dispatcher-type flex
   --moe-flex-dispatcher-backend deepep
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

   # EP=8 explicit (default SGLang `alltoall` backend, NOT DeepEP). Compared
   # to 10 (--sglang-ep-size 4, alltoall), this isolates whether EP=8 alone
   # is slower than EP=4 — independent of the DeepEP vs alltoall axis that
   # 11 was testing. Compared to 11 (EP=8, deepep), this removes DeepEP to
   # see what fraction of 11's slowdown was DeepEP vs EP=8 itself.
   --sglang-ep-size 8

   # Cap SGLang's per-request context at 64K to match the trainer's
   # effective context window (CP × max_tokens_per_gpu = 4 × 16384). Without
   # this, SGLang defaults to Qwen3-235B's full max-position-embeddings
   # (262K), and a long-thinking sample could exceed what the trainer is
   # sized to process. Aligns the two caps so any over-cap rollout fails
   # cleanly at the engine instead of mysteriously breaking the trainer.
   --sglang-context-length 65536
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
      --wandb-project slime-swe-env
      --wandb-group qwen3-235B-thinking-swe
      --wandb-key "${WANDB_KEY}"
   )
fi

MODAL_CONFIG_PATH="${MODAL_CONFIG_PATH:-${SLIME_ROOT}/.modal.toml}"

# Structured metric dump — rollout_metrics monkey-patches logging_utils.log
# to also append one JSON line per call to this path. Replaces ad-hoc log
# grepping for post-run analysis.
SLIME_METRICS_JSONL="${SLIME_METRICS_JSONL:-${SLIME_ROOT}/mnt/logs/metrics-${SLURM_JOB_ID:-manual}.jsonl}"

MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/root/Megatron-LM}"
RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${LIB_DIR}:${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${FULLY_ASYNC_DIR}:${SLIME_ROOT}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "MODAL_CONFIG_PATH": "${MODAL_CONFIG_PATH}",
    "SLIME_METRICS_JSONL": "${SLIME_METRICS_JSONL}",
    "SLIME_SWEBENCH_APP": "${SLIME_SWEBENCH_APP:-infx-slime-swebench-sandbox}",
    "SLIME_SWEBENCH_MAX_CONCURRENT": "${SLIME_SWEBENCH_MAX_CONCURRENT:-100}",
    "SLIME_SWEBENCH_SPAWN_QPS": "${SLIME_SWEBENCH_SPAWN_QPS:-10}",
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
