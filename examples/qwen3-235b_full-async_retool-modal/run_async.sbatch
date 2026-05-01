#!/bin/bash
#
# Slurm wrapper for the Qwen3-235B-A22B-Thinking-2507 fully-async + Modal
# retool example.
#
# 32 H200 nodes total: 8 actor (training) + 24 rollout (SGLang inference),
# orchestrated as a single Ray cluster.
#
# This cluster's pyxis insists on extracting container images into /tmp,
# which is only 30GB on GPU nodes (slime image is ~80GB extracted). We work
# around it by using enroot directly, redirected to /scratch (per-node, 25T)
# via the ENROOT_*_PATH env vars.
#
# Job name follows cluster convention: 'infx-' prefix, no 'rl', no model id.
#
# Submit from the slime repo root:
#   sbatch examples/qwen3-235b_full-async_retool-modal/run_async.sbatch
#
#SBATCH --job-name=infx-retool-async
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=mnt/logs/infx-retool-async-%j.out
#SBATCH --error=mnt/logs/infx-retool-async-%j.err

set -ex

SLIME_SQSH="${SLIME_SQSH:-/data/outputs/slime-images/slime-nightly-dev-20260425a.sqsh}"
SLIME_HOST_ROOT=/home/primeteam/slime
HF_CACHE_HOST=/data/huggingface
DATA_OUTPUTS_HOST=/data/outputs

# Per-job container name + per-node enroot data path on /scratch (25 TB local).
JOB="${SLURM_JOB_ID}"
CTR_NAME="slime-${JOB}"
ENROOT_BASE="/scratch/enroot-job-${JOB}"
LOGDIR=/home/primeteam/slime/mnt/logs

# Modal credentials (used inside container).
export MODAL_CONFIG_PATH="${MODAL_CONFIG_PATH:-/root/slime/.modal.toml}"
export MODAL_PROFILE="${MODAL_PROFILE:-semianalysis}"

# Discover allocated nodes; the first one is the Ray head.
NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
HEAD="${NODES[0]}"
HEAD_IP=$(srun --nodes=1 --ntasks=1 --nodelist="$HEAD" \
            bash -c 'hostname -I | tr " " "\n" | grep -vE "^(169\.254|127\.|::)" | head -1')
if [ -z "$HEAD_IP" ]; then
    HEAD_IP="$HEAD"
fi
echo "Ray head: $HEAD ($HEAD_IP); workers: ${NODES[@]:1}"

# Common enroot env shared by every srun in this job.
ENROOT_ENV="ENROOT_DATA_PATH=${ENROOT_BASE}/data,ENROOT_RUNTIME_PATH=${ENROOT_BASE}/runtime,ENROOT_CACHE_PATH=${ENROOT_BASE}/cache,TMPDIR=${ENROOT_BASE}/tmp"
ENROOT_MOUNTS="--mount ${SLIME_HOST_ROOT}:/root/slime --mount ${HF_CACHE_HOST}:/data/huggingface --mount ${DATA_OUTPUTS_HOST}:/data/outputs"

# 1. Per-node: clean up stale state + create the container from the shared sqsh.
srun --ntasks-per-node=1 \
     --export=ALL,${ENROOT_ENV} \
     bash -c "
        set -e
        mkdir -p ${ENROOT_BASE}/{data,runtime,cache,tmp}
        # Stale Ray / SGLang / Python from prior runs on the same node.
        pkill -9 sglang || true
        pkill -9 ray || true
        pkill -9 python || true
        # Clean up any prior named container with the same name on this node.
        enroot remove --force ${CTR_NAME} 2>/dev/null || true
        sleep 2
        enroot create --name ${CTR_NAME} ${SLIME_SQSH}
        # Smoke-test that it starts.
        enroot start --rw ${ENROOT_MOUNTS} ${CTR_NAME} bash -c 'echo bring-up ok on \$(hostname); python --version'
     "

# 2. Bring up Ray head + workers. Each ray process runs inside its own
#    `enroot start` invocation; we background each so the orchestrator can
#    continue while the daemons hold their nodes.
RAY_BOOT='python -c "import modal" 2>/dev/null || pip install --quiet --no-cache-dir --break-system-packages "modal>=0.65"'

srun --nodes=1 --ntasks=1 --nodelist="$HEAD" \
     --export=ALL,${ENROOT_ENV} \
     --output="${LOGDIR}/infx-retool-async-${JOB}-head.out" \
     bash -c "
        enroot start --rw ${ENROOT_MOUNTS} \
            ${CTR_NAME} \
            bash -c '
                cd /root/slime
                ${RAY_BOOT} && ray start --head \
                    --node-ip-address=${HEAD_IP} \
                    --port=6379 \
                    --num-gpus=8 \
                    --disable-usage-stats \
                    --dashboard-host=0.0.0.0 \
                    --dashboard-port=8265 \
                    --block
            '
     " &
HEAD_PID=$!

sleep 30

WORKER_PIDS=()
for node in "${NODES[@]:1}"; do
    srun --nodes=1 --ntasks=1 --nodelist="$node" \
         --export=ALL,${ENROOT_ENV} \
         --output="${LOGDIR}/infx-retool-async-${JOB}-${node}.out" \
         bash -c "
            enroot start --rw ${ENROOT_MOUNTS} \
                ${CTR_NAME} \
                bash -c '
                    cd /root/slime
                    ${RAY_BOOT} && ray start --address=${HEAD_IP}:6379 \
                        --num-gpus=8 \
                        --disable-usage-stats \
                        --block
                '
         " &
    WORKER_PIDS+=($!)
done

# Wait for all workers to register before submitting the training job.
sleep 30

# 3. Submit the training job from the head node — same container, --overlap
# so it can share the head's slot while ray --block holds it.
srun --nodes=1 --ntasks=1 --nodelist="$HEAD" --overlap \
     --export=ALL,${ENROOT_ENV},NUM_ROLLOUT="${NUM_ROLLOUT:-3000}",ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}",MODAL_CONFIG_PATH="${MODAL_CONFIG_PATH}",MODAL_PROFILE="${MODAL_PROFILE}",WANDB_KEY="${WANDB_KEY}",HEAD_IP="${HEAD_IP}" \
     bash -c "
        enroot start --rw ${ENROOT_MOUNTS} \
            --env NUM_ROLLOUT --env ROLLOUT_BATCH_SIZE --env MODAL_CONFIG_PATH --env MODAL_PROFILE --env WANDB_KEY --env HEAD_IP \
            ${CTR_NAME} \
            bash -c 'cd /root/slime && bash examples/qwen3-235b_full-async_retool-modal/run_async.sh \"\${HEAD_IP}\"'
     "

# 4. Once train_async.py exits, tear down Ray + remove the per-job containers.
echo "Training command finished — cleaning up Ray + enroot containers."
srun --ntasks-per-node=1 --overlap \
     --export=ALL,${ENROOT_ENV} \
     bash -c "
        enroot start --rw ${CTR_NAME} bash -c 'ray stop --force || true; pkill -9 ray || true; pkill -9 python || true' || true
        enroot remove --force ${CTR_NAME} 2>/dev/null || true
        rm -rf ${ENROOT_BASE} 2>/dev/null || true
     "
kill "${HEAD_PID}" "${WORKER_PIDS[@]}" 2>/dev/null || true
wait
