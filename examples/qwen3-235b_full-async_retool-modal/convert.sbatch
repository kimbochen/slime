#!/bin/bash
#
# One-shot HF -> Megatron torch_dist conversion for Qwen3-235B-A22B-Thinking-2507.
# Uses direct enroot (NOT pyxis --container-image=...) because pyxis on this
# cluster extracts containers into /tmp (only 30 GB) which is too small for
# the slime image (~80 GB extracted). Direct enroot lets us redirect the
# rootfs to /scratch (per-node, 25 TB).
#
# Submit from the slime repo root:
#   sbatch examples/qwen3-235b_full-async_retool-modal/convert.sbatch
#
#SBATCH --job-name=infx-retool-convert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=mnt/logs/infx-retool-convert-%j.out
#SBATCH --error=mnt/logs/infx-retool-convert-%j.err

set -ex

SLIME_SQSH="${SLIME_SQSH:-/data/outputs/slime-images/slime-nightly-dev-20260425a.sqsh}"
SLIME_HOST_ROOT=/home/primeteam/slime
HF_CACHE_HOST=/data/huggingface
DATA_OUTPUTS_HOST=/data/outputs

JOB="${SLURM_JOB_ID}"
CTR_NAME="slime-convert-${JOB}"
ENROOT_BASE="/scratch/enroot-job-${JOB}"
ENROOT_ENV="ENROOT_DATA_PATH=${ENROOT_BASE}/data ENROOT_RUNTIME_PATH=${ENROOT_BASE}/runtime ENROOT_CACHE_PATH=${ENROOT_BASE}/cache TMPDIR=${ENROOT_BASE}/tmp"
ENROOT_MOUNTS="--mount ${SLIME_HOST_ROOT}:/root/slime --mount ${HF_CACHE_HOST}:/data/huggingface --mount ${DATA_OUTPUTS_HOST}:/data/outputs"

# Set up enroot dirs, create the container, then run torchrun inside it.
export ${ENROOT_ENV}
mkdir -p ${ENROOT_BASE}/{data,runtime,cache,tmp}
enroot remove --force ${CTR_NAME} 2>/dev/null || true
enroot create --name ${CTR_NAME} ${SLIME_SQSH}

# tools/convert_hf_to_torch_dist.py:55-71 auto-overrides PP to world_size
# whenever --pipeline-model-parallel-size==1 (see SETUP.md quirk #2). With 8
# ranks that gives PP=8, EP=1, ETP=1, TP=1 — Megatron's
# world_size %% (ETP * EP * PP) == 0 check is satisfied (1*1*8=8). The saved
# torch_dist checkpoint is parallelism-agnostic.
#
# Qwen3-Thinking-2507 has rope_theta=5_000_000 (not the 1M default in the
# slime model script). Set MODEL_ARGS_ROTARY_BASE=5000000 BEFORE sourcing.
HF_BF16=/data/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Thinking-2507/snapshots/6cbffae6d8e28b986a6b17bd36f42f9fa0f1f0a5
SAVE_OUT=/data/outputs/slime-checkpoints/Qwen3-235B-A22B-Thinking-2507_torch_dist

enroot start --rw ${ENROOT_MOUNTS} \
    ${CTR_NAME} \
    bash -c "
        set -ex
        cd /root/slime
        export MODEL_ARGS_ROTARY_BASE=5000000
        source scripts/models/qwen3-235B-A22B.sh
        PYTHONPATH=/root/Megatron-LM:/root/slime torchrun --nproc_per_node 8 \
            tools/convert_hf_to_torch_dist.py \
            \"\${MODEL_ARGS[@]}\" \
            --hf-checkpoint ${HF_BF16} \
            --save          ${SAVE_OUT}
    "

# Cleanup
enroot remove --force ${CTR_NAME} 2>/dev/null || true
rm -rf ${ENROOT_BASE} 2>/dev/null || true
echo "convert sbatch finished"
