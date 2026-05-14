#!/bin/bash
#
# Slurm wrapper for the 2-node PD smoke test (GLM-4.5-Air + dapo math,
# upstream-aligned PD config). Submit from slime repo root:
#   sbatch examples/qwen3-235b_fullasync_swe-env/pd/smoke_2node/run.sbatch
#
#SBATCH --job-name=infx-pd-smoke
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=mnt/logs/infx-pd-smoke-%j.out
#SBATCH --error=mnt/logs/infx-pd-smoke-%j.err

set -ex

CONTAINER_IMAGE="${CONTAINER_IMAGE:-docker://slimerl/slime:nightly-dev-20260425a}"
SLIME_ROOT="${SLIME_ROOT:-$(pwd)}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${SLIME_ROOT}:/root/slime}"

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

srun --nodes=1 --ntasks=1 --nodelist="$HEAD" \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     --container-workdir=/root/slime \
     --output="${LOGDIR}/infx-pd-smoke-${JOB}-head.out" \
     bash -c "ray start --head \
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
         --output="${LOGDIR}/infx-pd-smoke-${JOB}-${node}.out" \
         bash -c "ray start --address=${HEAD_IP}:6379 \
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
     --export=ALL,NUM_ROLLOUT="${NUM_ROLLOUT:-5}" \
     bash examples/qwen3-235b_fullasync_swe-env/pd/smoke_2node/run.sh "${HEAD_IP}"

echo "Training command finished — stopping Ray cluster."
srun --ntasks-per-node=1 --overlap \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     bash -c 'ray stop --force || true'
kill "${HEAD_PID}" "${WORKER_PIDS[@]}" 2>/dev/null || true
wait
