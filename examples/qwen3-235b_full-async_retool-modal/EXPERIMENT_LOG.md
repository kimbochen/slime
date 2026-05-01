# Experiment log — qwen3-235b_full-async_retool-modal

Tail-first record of the bring-up. Goal: 20 rollout iterations with usable
metrics for Qwen3-235B-A22B-Thinking-2507, fully-async + Modal retool.

---

## Run 0 — bring-up prerequisites (no Slurm job yet)

**Date:** 2026-04-27

**State:**
- New cluster (`vm01-chjpbwu` head, `qpn01gpu[003-011,041-063]` 32 GPU nodes,
  `vm0{2,3}-chjpbwu` CPU admin nodes). Slurm 25.05.3 + pyxis + enroot.
- HF weights cached in `/data/huggingface/hub/`:
  * BF16: `models--Qwen--Qwen3-235B-A22B-Thinking-2507` (438 GB, 118 shards)
  * FP8:  `models--Qwen--Qwen3-235B-A22B-Thinking-2507-FP8` (221 GB, 24 shards)
- `/data` (Weka, 50 TB, 43 TB free) is shared across all nodes.
- `/home` is NFS-mounted on GPU nodes (head exports it).
- Modal token at `/home/primeteam/slime/migrate-fully_async-retool-modal/modal.toml`
  uses `[semianalysis]` profile; copied to `/home/primeteam/slime/.modal.toml`.
- HF datasets: dapo-math-17k + aime-2024 downloaded to `mnt/data/`.

---

## Run 0a — Modal sandbox smoke test

**Cmd:** `.venv/bin/python test_modal_sandbox.py` with `MODAL_PROFILE=semianalysis`.

**Outcome:** PASS (4/4). Pool of 4 sandboxes warmed in ~1s; 8 concurrent
execs in 0.9s; network blocked confirmed; per-exec timeout fires.

---

## Run 0b — Parser unit tests (offline)

**Cmd:** `.venv/bin/python test_qwen_postprocess.py`

**Outcome:** PASS (21/21). Covers Qwen3-native JSON tool calls, multi-line
code with raw newlines (invalid JSON → escape fallback), `<think>` block
stripping (so a fake `<tool_call>` inside `<think>` is ignored), unknown
tool names, malformed JSON, multiple tool calls (first wins), final answer
beats earlier tool_call, JSON-string `arguments`, etc.

---

## Run 0c — sbatch dry-runs

**Cmd:**
```
sbatch --test-only examples/qwen3-235b_full-async_retool-modal/convert.sbatch
sbatch --test-only examples/qwen3-235b_full-async_retool-modal/run_async.sbatch
```

**Outcome:** Initial allocation failure — `Requested node configuration is not available`.
Root cause: each GPU node has `CPUTot=64`; sbatch defaulted to `--cpus-per-task=128`
(inherited from the GLM example's 128-CPU cluster). Cut to 64 and both pass.

---

## Run 0d — pre-import slime container image

**Why:** Inline `srun --container-image=docker://slimerl/slime:...` consistently
fails on GPU nodes with
`parallel: Cannot append to buffer file in /tmp ... Is the disk full?`.
GPU nodes have only 30G `/tmp` and the slime image extraction needs more.

**Workaround:** `enroot import` once on the head node, with cache + temp
redirected to `/data` (Weka, 43 TB free). Output: a single ~25 GB `.sqsh`
that pyxis mounts directly via `--container-image=/data/.../slime.sqsh`.

**Cmd:**
```
ENROOT_TEMP_PATH=/data/outputs/slime-images/_cache \
ENROOT_CACHE_PATH=/data/outputs/slime-images/cache \
TMPDIR=/data/outputs/slime-images/_cache \
enroot import \
  -o /data/outputs/slime-images/slime-nightly-dev-20260425a.sqsh \
  docker://slimerl/slime:nightly-dev-20260425a
```

**Outcome:** ⏳ in progress — see `mnt/logs/enroot-import.log`.

---

## Run 1 — convert.sbatch, job 131

**Date:** 2026-04-27 16:31 UTC

**Cmd:** `sbatch examples/qwen3-235b_full-async_retool-modal/convert.sbatch`

**Two predecessor failures:**
- Job 129: pyxis path issue — `enroot start --workdir` flag doesn't exist (pyxis-only). Fix: `cd /root/slime` inside the bash command.
- Job 130: cancelled (test-only).

**Run:** PASSED in 7m 18s on 1 GPU node (qpn01gpu003).

Output: `/data/outputs/slime-checkpoints/Qwen3-235B-A22B-Thinking-2507_torch_dist/release/` (438 GB, 19 files: 16 distcp shards + common.pt + metadata.json + modelopt_run_config.yaml). PP=8 auto-shard (per quirk #2). Tracker file: `latest_checkpointed_iteration.txt` = "release".

---

## Run 2 — first training submit, job 133

**Date:** 2026-04-27 16:38 UTC

**Cmd:** `sbatch --export=ALL,NUM_ROLLOUT=20 run_async.sbatch`

**Outcome:** FAILED at SGLang init.

**Bring-up worked through:**
- 32-node enroot create (parallel, ~15 s per node from sqsh on /data)
- Ray head + 31 workers up
- `train_async.py` Ray-job submitted (256-GPU placement group)
- RolloutManager actor created
- 48 SGLang engine actors spawned

**Failure:**
```
ValueError: The output_size of gate's and up's weight = 192 is not divisible by weight quantization block_n = 128.
```

**Root cause:** Qwen3-235B FP8 uses 128×128 weight blocks. With our SGLang
config TP=8 per engine, each rank holds 1536/8 = 192 channels of the MoE
gate_proj output — not divisible by 128. SGLang asserts during weight load.

**Fix (Run 3):** drop `--rollout-num-gpus-per-engine 8 → 4`. Per-rank
shard becomes 1536/4 = 384 = 3 × 128 ✓. With 192 GPUs and TP=4 we get 48
engines instead of 24, but the FP8 weights (~220 GB) still fit comfortably
across 4×141 GB H200s per engine.

This is **Q17 in our cluster-specific quirks** (in addition to Q13-Q16
documented in README.md): FP8 block_n=128 + Qwen3 moe_intermediate_size=1536
constrains per-engine TP to one of {1, 2, 3, 4, 6, 12}.

---

## Run 3 — TP=4 per engine, job 135

**Date:** 2026-04-27 16:43 UTC

**Cmd:** `sbatch --export=ALL,NUM_ROLLOUT=20 run_async.sbatch`

**Status:** RUNNING. Monitoring for first `perf 0:` line (expected ~t+15-20 min).

### Run 3 progress (job 135) — t+22 min snapshot

**Bring-up timeline** (from sbatch start to first `perf 0:`):
- 16:38: sbatch submitted
- 16:39: enroot create on all 32 nodes (~1 min)
- 16:40: ray head + 31 workers up
- 16:45: SGLang server_args dumped (48 engines, TP=4 each)
- 16:50–16:54: SGLang FP8 weight load + DeepGEMM warmup + CUDA graph capture (~5 min)
- 16:54: "Server is up and ready to roll" (48 engines)
- 16:56: Megatron actor weight load from torch_dist done
- 16:57: First `update_weights` push (46.4s)
- 16:57: First rollout starts
- 17:00: First rollout completes (256 samples in 136s)
- 17:03: First `perf 0` line — t+25 min from sbatch start

**Step 0 metrics:**
- `train/loss` = -1.02e-4, `train/entropy_loss` = 0.232, `train/grad_norm` = 0.083
- `step_time` = 393.6s (6.5 min)
- `actor_train_time` = 114.5s, `actor_train_tflops` = 33.7 TFLOPs/GPU (3.4% MFU)
- `wait_time_ratio` = 0.47 (47% of step idle waiting on rollouts — moderate fully-async overlap)
- `update_weights_time` = 46.4s
- `ref_log_probs_time` = 79s, `log_probs_time` = 14s

**Rollout-side stats (rollout 0–2):**
- mean response: 5705 → 6312 → 6203 tokens
- truncated_ratio: 31.6% → 48.4% → 37.1% (Qwen3-Thinking burns context on `<think>`)
- rollout_time: 136s → 121s → 57s (rollout 2 overlapping with step 0 training — fully-async paying off)
- sandbox calls/sample: 0.14 → 0.04 → … (model rarely tool-calls in early steps)
- sandbox error_rate: 0.0%
