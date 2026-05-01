#!/bin/bash
#
# One-shot HF -> Megatron torch_dist conversion for GLM-4.5 106B-A12B.
# Runs inside the slime docker image via pyxis, on a single GPU node.
#
# Submit from the slime repo root once the HF download is complete:
#   sbatch examples/glm45-air_full-async_retool-modal/convert.sbatch
#
#SBATCH --job-name=infx-retool-convert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=mnt/logs/infx-retool-convert-%j.out
#SBATCH --error=mnt/logs/infx-retool-convert-%j.err
#SBATCH --container-image=docker://slimerl/slime:nightly-dev-20260425a
#SBATCH --container-mounts=/home/sa-shared/kimbo/slime:/root/slime
#SBATCH --container-workdir=/root/slime

set -ex

source scripts/models/glm4.5-106B-A12B.sh

# Conversion runs with 8 ranks on the 8 GPUs of this node.
#
# tools/convert_hf_to_torch_dist.py:55-71 auto-overrides PP to world_size
# whenever --pipeline-model-parallel-size==1, so we end up at PP=8. Megatron
# then enforces  world_size %% (ETP * EP * PP) == 0, which means EP*ETP must
# be 1 here. We therefore use PP-only sharding for conversion (TP=EP=ETP=1,
# PP=8) — each rank holds ~5-6 of the 46 layers (~13 B params, ~26 GB BF16,
# comfortable on H200). The saved torch_dist checkpoint is parallelism-
# agnostic, so the training job can load it with any TP/PP/EP.
PYTHONPATH=/root/Megatron-LM:/root/slime torchrun --nproc_per_node 8 \
    tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint /root/slime/mnt/checkpoints/zai-org/GLM-4.5-Air \
    --save          /root/slime/mnt/checkpoints/zai-org/GLM-4.5-Air_torch_dist
