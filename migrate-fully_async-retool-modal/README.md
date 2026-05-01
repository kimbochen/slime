# Retool with Modal sandbox + fully-async (12 H200 nodes)

This example is the same retool-style tool-using RL recipe as `examples/retool/`, with three changes:

1. **Tool sandbox** — Python code execution runs in **Modal Sandboxes** instead of a local subprocess. The `block_network=True` Modal flag is the actual safety boundary, so the regex-based "dangerous pattern" check from the original example is dropped.
2. **Fully-async rollout** — uses `train_async.py` plus `examples/fully_async/fully_async_rollout.py` so rollout production and training overlap continuously rather than alternating per step.
3. **12-node H200 layout on Slurm** — 4 actor nodes (32 GPUs) for Megatron training and 8 rollout nodes (64 GPUs) for SGLang inference. Bring-up is via `sbatch` + `srun`, not the SSH-loop pattern that ships with slime's other multi-node scripts.

The default model is **GLM-4.5 106B-A12B** (config: `scripts/models/glm4.5-106B-A12B.sh`). To swap models, edit only the `source` line in `run_async.sh` and adjust the parallelism flags as needed.

## Files

```
examples/glm45-air_full-async_retool-modal/
├── README.md                # this file
├── requirements.txt         # jinja2, psutil, modal>=0.65
├── modal_tool_sandbox.py    # Modal-backed async sandbox (replaces tool_sandbox.py)
├── generate_with_retool.py  # custom generate function (mirrors examples/retool/)
├── run_async.sbatch         # 12-node Slurm wrapper
└── run_async.sh             # inner ray-job submission script
```

## Setup

### 1. Install deps

```bash
cd /home/sa-shared/kimbo/slime
pip install -e . --no-deps
pip install -r examples/glm45-air_full-async_retool-modal/requirements.txt
```

### 2. Modal credentials

The launcher reads Modal credentials from `/home/sa-shared/kimbo/slime/.modal.toml` via `MODAL_CONFIG_PATH`. Make sure that file exists and contains a valid `[default]` section with `token_id` and `token_secret`.

```bash
ls -la /home/sa-shared/kimbo/slime/.modal.toml   # should exist, mode 600
```

Confirm the credentials work before launching the cluster job:

```bash
MODAL_CONFIG_PATH=/home/sa-shared/kimbo/slime/.modal.toml \
  python -c "import modal; print(modal.App.lookup('infx-slime-retool-sandbox', create_if_missing=True))"
```

### 3. Datasets and checkpoints

All artifacts live under `mnt/` in the slime repo (this matches the run script's `CKPT_ROOT` / `DATA_ROOT` defaults):

```
mnt/
├── checkpoints/
│   └── zai-org/
│       ├── GLM-4.5-Air/                # raw HF weights (~210 GB, gated)
│       └── GLM-4.5-Air_torch_dist/     # converted Megatron-LM checkpoint
└── data/
    ├── dapo-math-17k/
    └── aime-2024/
```

Download the datasets:

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k \
    --local-dir mnt/data/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 \
    --local-dir mnt/data/aime-2024
```

GLM-4.5 is **gated** — log in first using the token in `.hf_token.txt`:

```bash
hf auth login --token "$(cat .hf_token.txt)"
hf download zai-org/GLM-4.5-Air \
    --local-dir mnt/checkpoints/zai-org/GLM-4.5-Air
```

Convert HF → Megatron `torch_dist` inside the slime image. This cluster uses
the pyxis Slurm plugin (`--container-image`), so submit the bundled
conversion sbatch:

```bash
sbatch examples/glm45-air_full-async_retool-modal/convert.sbatch
```

This allocates one GPU node, pulls `slimerl/slime:$(cat docker/version.txt)`,
mounts the slime repo at `/root/slime`, and runs
`tools/convert_hf_to_torch_dist.py`. Output lands at
`mnt/checkpoints/zai-org/GLM-4.5-Air_torch_dist/`.

## Launch

```bash
cd /home/sa-shared/kimbo/slime
sbatch examples/glm45-air_full-async_retool-modal/run_async.sbatch
```

Logs land in `logs/infx-retool-async-<jobid>.out`. The Ray dashboard is reachable at the head node on port 8265.

## Slurm naming convention

Per cluster policy, the sbatch `--job-name` is `infx-retool-async`:

- `infx-` prefix is required.
- The string `rl` is forbidden.
- Model identifiers (`glm`, `qwen`, etc.) are forbidden.

If you fork this script for another model or task, keep the same conventions in the new job name.

## Tuning

- `SLIME_MODAL_POOL_SIZE` (default 64) — number of warm Modal sandboxes the rollout workers share. Bump this if Modal sandbox acquisition becomes the rollout bottleneck.
- `SLIME_MODAL_PER_EXEC_TIMEOUT` (default 60) — per-`exec` wallclock cap, in seconds.
- `SLIME_MODAL_APP` (default `infx-slime-retool-sandbox`) — Modal App name used for the sandbox pool.
- The 12 → 4 + 8 split assumes rollout is the bottleneck. Flip to 6 + 6 or 8 + 4 by editing `--actor-num-nodes`, `--rollout-num-gpus`, and `#SBATCH --nodes` together.

## How this differs from `examples/retool/`

| | `examples/retool/` | `examples/glm45-air_full-async_retool-modal/` |
|---|---|---|
| Sandbox | local `subprocess` + regex safety check | Modal Sandbox pool, network-blocked |
| Train driver | `train.py` (sync) | `train_async.py` (async) |
| Rollout pattern | per-step | continuous fully-async worker |
| Scale | 1 node × 4 GPUs | 12 nodes × 8 H200 (4 actor + 8 rollout) |
| Default model | Qwen3-4B | GLM-4.5 106B-A12B |
| Orchestration | direct `bash` | Slurm sbatch + srun |
