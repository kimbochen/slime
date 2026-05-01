#!/bin/bash
#
# Slurm wrapper for the GLM-4.5 106B-A12B fully-async + Modal retool example.
#
# 12 H200 nodes total: 4 actor (training) + 8 rollout (SGLang inference),
# orchestrated as a single Ray cluster. Pyxis is used per-srun (NOT at the
# sbatch level) so the orchestrator can call slurm tools like scontrol.
#
# Job name follows the cluster convention: 'infx-' prefix, no 'rl' anywhere,
# no model identifier. Output filename follows the same rule.
#
# Submit from the slime repo root:
#   sbatch examples/glm45-air_full-async_retool-modal/run_async.sbatch
#
#SBATCH --job-name=infx-retool-async
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=mnt/logs/infx-retool-async-%j.out
#SBATCH --error=mnt/logs/infx-retool-async-%j.err

set -ex

CONTAINER_IMAGE="docker://slimerl/slime:nightly-dev-20260425a"
CONTAINER_MOUNTS="/home/sa-shared/kimbo/slime:/root/slime"

# Modal credentials path (used inside the container).
export MODAL_CONFIG_PATH="${MODAL_CONFIG_PATH:-/root/slime/.modal.toml}"

# Discover allocated nodes; the first one is the Ray head.
NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
HEAD="${NODES[0]}"
# Resolve a routable IP for the Ray head. `hostname -I` on Soperator nodes
# returns the link-local 169.254.x.x first, which workers can't reach across
# nodes — pick the first non-link-local IPv4 instead.
HEAD_IP=$(srun --nodes=1 --ntasks=1 --nodelist="$HEAD" \
            bash -c 'hostname -I | tr " " "\n" | grep -vE "^(169\.254|127\.|::)" | head -1')
if [ -z "$HEAD_IP" ]; then
    HEAD_IP="$HEAD"
fi
echo "Ray head: $HEAD ($HEAD_IP); workers: ${NODES[@]:1}"

JOB=${SLURM_JOB_ID}
LOGDIR=/home/sa-shared/kimbo/slime/mnt/logs

# 1. Clean up any stale Ray / SGLang on every node.
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

# 2. Start Ray. Modal SDK gets installed in the SAME container where
# `ray start` runs so the spawned Ray actors (which inherit this Python
# env) can `import modal`. A separate install srun would land in a
# throwaway container and the install wouldn't persist for Ray actors.
RAY_BOOT='python -c "import modal" 2>/dev/null || pip install --quiet --no-cache-dir --break-system-packages "modal>=0.65"'

# Ray head on $HEAD. --block keeps the srun task alive so Slurm treats it
# as a running task; backgrounding with & lets us continue.
srun --nodes=1 --ntasks=1 --nodelist="$HEAD" \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     --container-workdir=/root/slime \
     --output="${LOGDIR}/infx-retool-async-${JOB}-head.out" \
     bash -c "${RAY_BOOT} && ray start --head \
                       --node-ip-address=${HEAD_IP} \
                       --port=6379 \
                       --num-gpus=8 \
                       --disable-usage-stats \
                       --dashboard-host=0.0.0.0 \
                       --dashboard-port=8265 \
                       --block" &
HEAD_PID=$!

# Give the head time to come up (incl. modal install on first launch).
sleep 30

# 3. Ray workers on each remaining node — same modal-then-ray-start pattern.
WORKER_PIDS=()
for node in "${NODES[@]:1}"; do
    srun --nodes=1 --ntasks=1 --nodelist="$node" \
         --container-image="$CONTAINER_IMAGE" \
         --container-mounts="$CONTAINER_MOUNTS" \
         --container-workdir=/root/slime \
         --output="${LOGDIR}/infx-retool-async-${JOB}-${node}.out" \
         bash -c "${RAY_BOOT} && ray start --address=${HEAD_IP}:6379 \
                           --num-gpus=8 \
                           --disable-usage-stats \
                           --block" &
    WORKER_PIDS+=($!)
done

# Wait for all workers to register before submitting the job.
sleep 30

# 5. Submit the actual training job from the head node.
# --overlap lets this srun step share the head node with the Ray head srun
# that is currently holding the slot via `ray start --block`.
srun --nodes=1 --ntasks=1 --nodelist="$HEAD" --overlap \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     --container-workdir=/root/slime \
     --export=ALL,NUM_ROLLOUT="${NUM_ROLLOUT:-3000}",ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}",MODAL_CONFIG_PATH="${MODAL_CONFIG_PATH}" \
     bash examples/glm45-air_full-async_retool-modal/run_async.sh "${HEAD_IP}"

# 6. Once train_async.py exits, tear down the Ray cluster cleanly.
# --overlap because every node is still held by its `ray start --block` srun.
echo "Training command finished — stopping Ray cluster."
srun --ntasks-per-node=1 --overlap \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     bash -c 'ray stop --force || true'
kill "${HEAD_PID}" "${WORKER_PIDS[@]}" 2>/dev/null || true
wait
