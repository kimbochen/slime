#!/bin/bash
#
# Slurm wrapper for the Qwen3-235B-Thinking SWE-bench-Verified training run.
# Same orchestration as run_async.sbatch for the retool variant — only the
# inner script differs (run_swe.sh wires generate_with_codingagent + the
# SWE-bench prompt-data).
#
# Submit from slime repo root:
#   sbatch examples/qwen3-235b_fullasync_swe-env/run_swe.sbatch
#
#SBATCH --job-name=infx-swe-async
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=mnt/logs/infx-swe-async-%j.out
#SBATCH --error=mnt/logs/infx-swe-async-%j.err

set -ex

CONTAINER_IMAGE="${CONTAINER_IMAGE:-docker://slimerl/slime:nightly-dev-20260425a}"
# Mount the slime repo root into the container. Defaults to the directory
# this sbatch was launched from; override SLIME_ROOT to a different host
# path if needed.
SLIME_ROOT="${SLIME_ROOT:-$(pwd)}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${SLIME_ROOT}:/root/slime}"

export MODAL_CONFIG_PATH="${MODAL_CONFIG_PATH:-/root/slime/.modal.toml}"

NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
HEAD="${NODES[0]}"
HEAD_IP=$(srun --nodes=1 --ntasks=1 --nodelist="$HEAD" \
            bash -c 'hostname -I | tr " " "\n" | grep -vE "^(169\.254|127\.|::)" | head -1')
if [ -z "$HEAD_IP" ]; then
    HEAD_IP="$HEAD"
fi
echo "Ray head: $HEAD ($HEAD_IP); workers: ${NODES[@]:1}"

JOB=${SLURM_JOB_ID}
LOGDIR="${LOGDIR:-${SLIME_ROOT}/mnt/logs}"
mkdir -p "${LOGDIR}"

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

RAY_BOOT='python -c "import modal" 2>/dev/null || pip install --quiet --no-cache-dir --break-system-packages "modal>=0.65"'

srun --nodes=1 --ntasks=1 --nodelist="$HEAD" \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     --container-workdir=/root/slime \
     --output="${LOGDIR}/infx-swe-async-${JOB}-head.out" \
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
         --output="${LOGDIR}/infx-swe-async-${JOB}-${node}.out" \
         bash -c "${RAY_BOOT} && ray start --address=${HEAD_IP}:6379 \
                           --num-gpus=8 \
                           --disable-usage-stats \
                           --block" &
    WORKER_PIDS+=($!)
done

sleep 30

srun --nodes=1 --ntasks=1 --nodelist="$HEAD" --overlap \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     --container-workdir=/root/slime \
     --export=ALL,NUM_ROLLOUT="${NUM_ROLLOUT:-200}",ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}",MODAL_CONFIG_PATH="${MODAL_CONFIG_PATH}" \
     bash examples/qwen3-235b_fullasync_swe-env/run_swe.sh "${HEAD_IP}"

echo "Training command finished — stopping Ray cluster."
srun --ntasks-per-node=1 --overlap \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     bash -c 'ray stop --force || true'
kill "${HEAD_PID}" "${WORKER_PIDS[@]}" 2>/dev/null || true
wait
