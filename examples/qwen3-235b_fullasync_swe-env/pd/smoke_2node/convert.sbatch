#!/bin/bash
#
# One-shot HF -> Megatron torch_dist conversion for Qwen3-4B, used by the
# PD smoke test on 2 nodes. Qwen3-4B is small enough that conversion
# finishes in a few minutes on a single H200 node.
#
# Submit from slime repo root after Qwen3-4B has been HF-downloaded into
# mnt/checkpoints/Qwen/Qwen3-4B:
#   sbatch examples/qwen3-235b_fullasync_swe-env/pd/smoke_2node/convert.sbatch
#
#SBATCH --job-name=infx-pd-convert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --output=mnt/logs/infx-pd-convert-%j.out
#SBATCH --error=mnt/logs/infx-pd-convert-%j.err
#SBATCH --container-image=docker://slimerl/slime:nightly-dev-20260425a
#SBATCH --container-mounts=/home/sa-shared/kimbo/slime:/root/slime
#SBATCH --container-workdir=/root/slime

set -ex

source scripts/models/qwen3-4B.sh

# tools/convert_hf_to_torch_dist.py auto-promotes PP=1 -> PP=world_size, so
# with 8 GPUs we end up at PP=8 (TP=EP=ETP=1). Qwen3-4B is 4B BF16 = 8 GB
# total weights, ~1 GB per PP rank — trivial on H200.
PYTHONPATH=/root/Megatron-LM:/root/slime torchrun --nproc_per_node 8 \
    tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint /root/slime/mnt/checkpoints/Qwen/Qwen3-4B \
    --save          /root/slime/mnt/checkpoints/Qwen/Qwen3-4B_torch_dist
