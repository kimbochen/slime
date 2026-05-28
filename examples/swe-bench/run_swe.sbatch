#!/bin/bash
#
# Slurm wrapper for the barebones SWE-bench training run.
# Same bring-up pattern as the qwen3-235b_fullasync_swe-env reference (exp 15):
#   1. Allocate N nodes.
#   2. Clean stale Ray/SGLang/python processes on every node.
#   3. Start Ray head on node 0 (backgrounded with --block).
#   4. Start Ray workers on the rest (also backgrounded with --block).
#   5. srun the inner script (run_swe.sh) on the head node with --overlap.
#   6. On exit, stop Ray and reap background PIDs.
#
# Submit from slime repo root:
#   sbatch examples/swe-bench/run_swe.sbatch
#
# Override knobs via env or --export:
#   NUM_ROLLOUT, ROLLOUT_BATCH_SIZE  — training extent
#   WANDB_KEY                        — enable wandb logging
#   CONTAINER_IMAGE                  — slime docker image
#   SLIME_SWEBENCH_*                 — sandbox tuning (see sandbox.py)
#
#SBATCH --job-name=infx-swebench-new
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=mnt/logs/infx-swebench-new-%j.out
#SBATCH --error=mnt/logs/infx-swebench-new-%j.err

set -ex

CONTAINER_IMAGE="${CONTAINER_IMAGE:-docker://slimerl/slime:nightly-dev-20260425a}"
SLIME_ROOT="${SLIME_ROOT:-$(pwd)}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${SLIME_ROOT}:/root/slime}"

# Modal credentials — rollout workers spawn sandboxes via this token.
export MODAL_CONFIG_PATH="${MODAL_CONFIG_PATH:-/root/slime/.modal.toml}"

JOB=${SLURM_JOB_ID}
LOGDIR="${LOGDIR:-${SLIME_ROOT}/mnt/logs}"
mkdir -p "${LOGDIR}"

# JSONL metrics sidecar path is set inside run_swe.sh (which has the
# container-relative SLIME_ROOT, i.e. /root/slime). We deliberately do NOT
# set SLIME_METRICS_JSONL here because the sbatch's $(pwd) is the host path
# /home/sa-shared/... which the container doesn't mount — would cause
# FileNotFoundError when log_patch.py tries to open the file.

# Resolve head IP from the first node in the allocation.
NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
HEAD="${NODES[0]}"
HEAD_IP=$(srun --nodes=1 --ntasks=1 --nodelist="$HEAD" \
            bash -c 'hostname -I | tr " " "\n" | grep -vE "^(169\.254|127\.|::)" | head -1')
if [ -z "$HEAD_IP" ]; then
    HEAD_IP="$HEAD"
fi
echo "Ray head: $HEAD ($HEAD_IP); workers: ${NODES[@]:1}"

# ---------------------------------------------------------------- cleanup --
# Stale Ray/SGLang/python from a prior crashed job can wedge the GPUs. Kill
# everything on every allocated node before bringing the cluster up.
srun --ntasks-per-node=1 \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     --container-workdir=/root/slime \
     bash -c '
        pkill -9 sglang || true
        ray stop --force || true
        pkill -9 ray || true
        pkill -9 python || true
        sleep 2
     '

# Bootstrap modal offline from a pre-downloaded wheel cache. We deviate
# from the reference's online pip install pattern because some cluster
# nodes have flaky outbound HTTPS (got "from versions: none" 5 retries in
# a row on one node during smoke testing, killing the whole job). Offline
# wheels eliminate the network dependency and made bootstrap reliable
# across 12 nodes in job 36551.
#
# To refresh after a modal version bump:
#   pip download --dest mnt/pkgs/modal --python-version 3.12 \
#       --platform manylinux2014_x86_64 --only-binary=:all: "modal>=0.65"
RAY_BOOT='python -c "import modal" 2>/dev/null || \
          pip install --quiet --break-system-packages \
              --no-index --find-links /root/slime/mnt/pkgs/modal/ "modal>=0.65" || \
          (echo "[bootstrap] FATAL: offline modal install failed; check /root/slime/mnt/pkgs/modal/ exists and has wheels" && exit 1)'

# ----------------------------------------------------------- ray cluster --
srun --nodes=1 --ntasks=1 --nodelist="$HEAD" \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     --container-workdir=/root/slime \
     --output="${LOGDIR}/infx-swebench-new-${JOB}-head.out" \
     bash -c "${RAY_BOOT} && ray start --head \
                       --node-ip-address=${HEAD_IP} \
                       --port=6379 \
                       --num-gpus=8 \
                       --disable-usage-stats \
                       --dashboard-host=0.0.0.0 \
                       --dashboard-port=8265 \
                       --block" &
HEAD_PID=$!

sleep 30

WORKER_PIDS=()
for node in "${NODES[@]:1}"; do
    srun --nodes=1 --ntasks=1 --nodelist="$node" \
         --container-image="$CONTAINER_IMAGE" \
         --container-mounts="$CONTAINER_MOUNTS" \
         --container-workdir=/root/slime \
         --output="${LOGDIR}/infx-swebench-new-${JOB}-${node}.out" \
         bash -c "${RAY_BOOT} && ray start --address=${HEAD_IP}:6379 \
                           --num-gpus=8 \
                           --disable-usage-stats \
                           --block" &
    WORKER_PIDS+=($!)
done

sleep 30

# ------------------------------------------------------------- training --
# --overlap so we share the head node with the ray head process. Pass the
# head IP positionally; run_swe.sh reads it as $1.
srun --nodes=1 --ntasks=1 --nodelist="$HEAD" --overlap \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     --container-workdir=/root/slime \
     --export=ALL,\
NUM_ROLLOUT="${NUM_ROLLOUT:-200}",\
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}",\
MODAL_CONFIG_PATH="${MODAL_CONFIG_PATH}",\
WANDB_KEY="${WANDB_KEY:-}",\
SLIME_SWEBENCH_APP="${SLIME_SWEBENCH_APP:-slime-swebench-sandbox}",\
SLIME_SWEBENCH_TIMEOUT="${SLIME_SWEBENCH_TIMEOUT:-1800}",\
SLIME_SWEBENCH_PER_CMD_TIMEOUT="${SLIME_SWEBENCH_PER_CMD_TIMEOUT:-120}",\
SLIME_SWEBENCH_MAX_OUTPUT_BYTES="${SLIME_SWEBENCH_MAX_OUTPUT_BYTES:-65536}",\
SLIME_SWEBENCH_CPU="${SLIME_SWEBENCH_CPU:-2.0}",\
SLIME_SWEBENCH_MEMORY_MB="${SLIME_SWEBENCH_MEMORY_MB:-8192}",\
SLIME_SWEBENCH_EVAL_TIMEOUT="${SLIME_SWEBENCH_EVAL_TIMEOUT:-1500}",\
SLIME_SWEBENCH_WARN_CONCURRENT="${SLIME_SWEBENCH_WARN_CONCURRENT:-128}" \
     bash examples/swe-bench/run_swe.sh "${HEAD_IP}"

echo "Training command finished — stopping Ray cluster."
srun --ntasks-per-node=1 --overlap \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     bash -c 'ray stop --force || true'
kill "${HEAD_PID}" "${WORKER_PIDS[@]}" 2>/dev/null || true
wait
