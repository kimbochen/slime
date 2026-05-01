# Qwen3-235B-A22B-Thinking-2507 fully-async retool with Modal sandbox

Adapted from `examples/glm45-air_full-async_retool-modal/`. Same recipe, different model:

- **Model:** Qwen3-235B-A22B-Thinking-2507 (BF16 training, FP8 inference)
- **Topology:** 32 H200 nodes — 8 actor (Megatron, BF16) + 24 rollout (SGLang, FP8). All nodes have 8 GPUs.
- **Tool sandbox:** Modal (block_network=True), per-call timeouts, instrumented for sandbox/* metrics.
- **Driver:** `train_async.py` with `fully_async_rollout.generate_rollout_fully_async`.

## What changed vs. the GLM template

| Piece | GLM-4.5-Air | Qwen3-235B-A22B-Thinking-2507 |
|---|---|---|
| Chat template | GLM-native `<|system|>...<|assistant|>\n<think></think>\n` | Qwen-native `<|im_start|>assistant\n<think>\n` (think tag is *opened*) |
| Tool-call grammar | `<tool_call>name\n<arg_key>k</arg_key>\n<arg_value>v</arg_value>\n</tool_call>` | `<tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>` (JSON) |
| Tool response | `<|observation|>\n<tool_response>\n{r}\n</tool_response>\n<|assistant|>\n<think></think>\n` | `<|im_end|>\n<|im_start|>user\n<tool_response>\n{r}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n` |
| Parser | `_GLM_TOOL_CALL_RE`, key/value pairs | `<tool_call>{json}</tool_call>`, json.loads w/ raw-newline fallback for multi-line code |
| Cluster nodes | 12 (4 actor + 8 rollout) | 32 (8 actor + 24 rollout) |
| Train parallelism | TP=4 EP=8 PP=1 CP=1 | TP=4 PP=4 CP=2 EP=16 ETP=1 |
| Rollout layout | TP=8 per engine, 8 engines | TP=8 per engine, 24 engines |
| Inference precision | BF16 | **FP8** via `--hf-checkpoint <FP8 path>` |
| `<think>` handling | Closed in prompt (empty `<think></think>`) | **Opened** in prompt — model emits long reasoning then `</think>` |
| Container image | docker://slimerl/slime:... | Same image, but **pre-imported as a local .sqsh** on /data (cluster-specific quirk) |

## Cluster-specific quirks for THIS env

(See SETUP.md in `migrate-fully_async-retool-modal/` for the upstream quirks 1-12.)

Q13 — **GPU node /tmp is too small for inline pyxis docker:// pull.** The
`parallel: Cannot append to buffer file in /tmp` error in pyxis. Pre-import
the image as a sqsh on /data once, then point `--container-image=/data/.../*.sqsh`
in sbatch. Skips per-node /tmp extraction.

Q14 — **GPU nodes have 64 CPUs (not 128).** Both the upstream GLM example
hardcoded `--cpus-per-task=128`. We use 64.

Q15 — **Rotary base is 5,000,000 for Qwen3-Thinking-2507** (not the 1M default
in `scripts/models/qwen3-235B-A22B.sh`). Set `MODEL_ARGS_ROTARY_BASE=5000000`
before sourcing.

Q16 — **Modal token file uses [semianalysis] profile**, not [default]. Set
`MODAL_PROFILE=semianalysis` in the runtime_env.

## Files

```
examples/qwen3-235b_full-async_retool-modal/
├── README.md                # this file
├── EXPERIMENT_LOG.md        # tail-first incident log
├── requirements.txt         # jinja2, psutil, modal>=0.65 (for the .venv host tools)
├── modal_tool_sandbox.py    # unchanged from GLM
├── add_slime_metrics.py     # unchanged from GLM (sandbox/throughput/prompt_len metrics)
├── generate_with_retool.py  # Qwen3 chat template + JSON tool-call parsing
├── test_qwen_postprocess.py # 21 parser tests (rename of test_glm_postprocess.py)
├── test_modal_sandbox.py    # CPU-only Modal smoke test (4 sub-tests)
├── metrics_report.py        # standalone CLI for analyzing slime stdout logs
├── convert.sbatch           # one-shot HF BF16 → Megatron torch_dist
├── run_async.sh             # inner ray-job submitter
└── run_async.sbatch         # outer Slurm wrapper (32 nodes)
```

## Quick start

Verify host tooling:

```bash
cd examples/qwen3-235b_full-async_retool-modal
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python "modal>=0.65" "huggingface_hub"
.venv/bin/python test_qwen_postprocess.py   # parser tests (21/21)
MODAL_CONFIG_PATH=/home/primeteam/slime/.modal.toml \
  MODAL_PROFILE=semianalysis \
  SLIME_MODAL_POOL_SIZE=4 SLIME_MODAL_PER_EXEC_TIMEOUT=10 \
  .venv/bin/python test_modal_sandbox.py    # Modal smoke test (4/4)
```

Pre-import the slime container image (one-time, ~10 min):

```bash
ENROOT_TEMP_PATH=/data/outputs/slime-images/_cache \
ENROOT_CACHE_PATH=/data/outputs/slime-images/cache \
TMPDIR=/data/outputs/slime-images/_cache \
enroot import \
  -o /data/outputs/slime-images/slime-nightly-dev-20260425a.sqsh \
  docker://slimerl/slime:nightly-dev-20260425a
```

Convert weights (BF16 HF safetensors → Megatron torch_dist):

```bash
sbatch examples/qwen3-235b_full-async_retool-modal/convert.sbatch
# tail mnt/logs/infx-retool-convert-<jobid>.out for progress
```

Submit training (smoke test, 20 rollout iterations):

```bash
sbatch --export=ALL,NUM_ROLLOUT=20 \
  examples/qwen3-235b_full-async_retool-modal/run_async.sbatch
```

Analyze metrics after the run completes (or while it's running):

```bash
./examples/qwen3-235b_full-async_retool-modal/metrics_report.py --job <jobid>
```
