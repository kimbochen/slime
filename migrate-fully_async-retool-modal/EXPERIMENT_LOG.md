# Experiment log — glm45-air_full-async_retool-modal

Running record of the bring-up loop for the GLM-4.5 106B-A12B fully-async +
Modal retool example. Each entry: config snapshot → launch command → outcome
→ next action. Goal: one run that completes ≥ 10 training steps.

The convention here is "tail-first" — newest run at the bottom, so a `tail`
of this file shows current status.

---

## Run 0 — bring-up prerequisites (no Slurm job yet)

**Date:** 2026-04-25

**State:**
- Authenticated to HF as `kimbochen` (org: `semianalysisai`).
- Datasets downloaded: `mnt/data/dapo-math-17k`, `mnt/data/aime-2024`.
- GLM-4.5 weights download: in progress, ~22 GB / ~210 GB at first check (background bash task `bc5c20b8h`).
- Modal credentials: `/home/sa-shared/kimbo/slime/.modal.toml` (read via `MODAL_CONFIG_PATH`).
- Slime docker image: `slimerl/slime:nightly-dev-20260425a` (per `docker/version.txt`).

**No launch yet.** Waiting on download.

---

## Run 0a — Modal sandbox smoke test (CPU-only)

**Date:** 2026-04-25

**Cmd:**
```bash
cd examples/glm45-air_full-async_retool-modal
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python "modal>=0.65"
MODAL_CONFIG_PATH=/home/sa-shared/kimbo/slime/.modal.toml \
  SLIME_MODAL_POOL_SIZE=4 \
  SLIME_MODAL_PER_EXEC_TIMEOUT=10 \
  .venv/bin/python test_modal_sandbox.py
```

**Result:** PASS (4/4) on second attempt.

**First attempt** failed test 4 (per-exec timeout): expected `timed out` in
output, got `'Error: Process exited with code -1\n'`. Modal's per-exec
timeout kills the process with a negative exit code rather than raising
`SandboxTimeoutError` from `process.wait.aio()`.

**Fix applied** (modal_tool_sandbox.py): wrap drain in `asyncio.wait_for()`
with `timeout + 5s` grace, AND treat any negative `return_code` from
`process.wait.aio()` as a timeout. Also fixed an `AsyncUsageWarning` by
switching `modal.App.lookup(...)` → `await modal.App.lookup.aio(...)`.

**Modal pool warmed:** 4 sandboxes brought up in ~1s; 8 concurrent execs
finished in 1.2s end-to-end. Network is confirmed blocked
(`urllib.urlopen` raises inside the sandbox).

---

## Run 0b — wrong HF repo caught at 224 GB

**Date:** 2026-04-25

**What happened:** Initial download targeted `zai-org/GLM-4.5`. After 224 GB
already on disk and 93 safetensors shards visible, a sanity check against
the HF model card revealed **`zai-org/GLM-4.5` is the 355B-A32B variant**
(~730 GB total). The 106B-A12B variant — the one that fits in 4 H200 nodes
and matches `scripts/models/glm4.5-106B-A12B.sh` — is named **`zai-org/GLM-4.5-Air`**.

**Action:** Stopped background download, `rm -rf` the partial 224 GB,
relaunched with `zai-org/GLM-4.5-Air` (background bash task `bz7m31808`).
Updated `run_async.sh`, `convert.sbatch`, and the README. Local paths now
end in `GLM-4.5-Air/` and `GLM-4.5-Air_torch_dist/`.

**Lesson:** For zai-org GLM 4.5 series, the suffix `-Air` denotes the smaller
sibling. `GLM-4.5` (no suffix) is the flagship 355B model. Add a
forward-looking note to the README so a future user doesn't repeat this.

---

## Run 0c — sbatch dry-run validates

**Date:** 2026-04-25

**Cmd:**
```bash
sbatch --test-only examples/glm45-air_full-async_retool-modal/run_async.sbatch
```

**Result:** OK — `sbatch: Job 23360 to start at 2026-04-25 23:20:03 UTC,
1536 processors on nodes worker-[0-3,5-6,8-13] in partition main`. 12 nodes
× 128 cpus = 1536, matches the request. Cluster has 14 idle workers
(`sinfo`), so the job will queue ~immediately on real submit.

**Datasets ready:** `mnt/data/dapo-math-17k/dapo-math-17k.jsonl` and
`mnt/data/aime-2024/aime-2024.jsonl` are both downloaded.

---

## Run 1 — conversion (HF → torch_dist), job 23361

**Date:** 2026-04-25 23:23 UTC

**Cmd:**
```bash
sbatch examples/glm45-air_full-async_retool-modal/convert.sbatch
# -> Submitted batch job 23361
```

**Allocation:** 1 node (worker-6), 8 GPUs, container `slimerl/slime:nightly-dev-20260425a` via pyxis.

**Pre-conversion state:** `mnt/checkpoints/zai-org/GLM-4.5-Air` = 206 GB,
47 safetensors shards. Container pull in progress at last check.

**Outcome:** FAILED (single-GPU OOM — see below). After fix, see Run 1b.

**Error (excerpt):**
```
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
...
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 22.00 MiB.
GPU 0 has a total capacity of 139.81 GiB of which 1.44 MiB is free.
Including non-PyTorch memory, this process has 139.80 GiB memory in use.
Of the allocated memory 132.20 GiB is allocated by PyTorch
```

**Root cause:** The convert command ran as a plain `python` invocation with
`world_size=1`, so the entire 106B model was loaded onto a single H200.
Every other slime/CI conversion command uses `torchrun --nproc_per_node 8`
(`scripts/run-gpt-oss-20B.sh:8`, `.github/workflows/conda-ci.yml:64`).

**Fix:** Updated `convert.sbatch` to invoke
`torchrun --nproc_per_node 8 tools/convert_hf_to_torch_dist.py` and pass
`--tensor-model-parallel-size 1 --expert-model-parallel-size 8` so the 128
experts split across 8 ranks (1/8 ≈ 13 B params per GPU).

`mnt/checkpoints/zai-org/GLM-4.5-Air_torch_dist` was empty so nothing to
clean up.

---

## Run 1b — torchrun multi-GPU conversion, job 23376 (FAILED)

**Date:** 2026-04-25 23:34 UTC

**Cmd:** `sbatch examples/glm45-air_full-async_retool-modal/convert.sbatch` after
updating to use `torchrun --nproc_per_node 8` with
`--tensor-model-parallel-size 1 --expert-model-parallel-size 8
--expert-tensor-parallel-size 1 --pipeline-model-parallel-size 1`.

**Outcome:** FAILED with
`RuntimeError: world_size (8) is not divisible by expert_tensor_model_pipeline_parallel size (64)`.

**Root cause:** `tools/convert_hf_to_torch_dist.py:55-71` contains:

```python
if args.pipeline_model_parallel_size == 1 and world_size > 1:
    pp_size = world_size
    while True:
        args.pipeline_model_parallel_size = pp_size
        ...
```

Whenever the caller passes PP=1 with multi-rank torchrun, the script
silently overrides PP to world_size (=8). Megatron's
`initialize_model_parallel` then enforces
`world_size % (ETP × EP × PP) == 0`. With my flags ETP×EP×PP = 1×8×8 = 64,
which does not divide 8.

**Fix (Run 1c):** Drop `--expert-model-parallel-size 8` and
`--expert-tensor-parallel-size 1` from the conversion command. The script's
auto-PP=world_size kicks in (PP=8), EP defaults to 1, ETP defaults to 1
→ ETMP = 1 × 1 × 8 = 8 ✓. Each rank then holds ~5-6 of the 46 layers
(~13 B params, ~26 GB BF16). The saved torch_dist format is parallelism-
agnostic so this doesn't constrain the training-time TP/PP/EP.

---

## Run 1c — torchrun PP-only conversion, job 23377 (PASSED)

**Date:** 2026-04-25 23:42 UTC

**Cmd:** `sbatch examples/glm45-air_full-async_retool-modal/convert.sbatch` after
removing the explicit EP/ETP flags.

**Outcome:** COMPLETED. Per-rank memory after load: 26.92 GB allocated,
106 GB free on each H200 — well within budget. Per-rank params:
~12.5–14.0 B (PP-rank 0 has the embedding layer so 12.5; the others 14).

**Output:** `mnt/checkpoints/zai-org/GLM-4.5-Air_torch_dist/release/`
(200 GB total, parallelism-agnostic torch_dist format).

---

## Run 2 — first 12-node training submit, job 23378

**Date:** 2026-04-25 23:46 UTC

**Cmd:**
```bash
sbatch --export=ALL,NUM_ROLLOUT=10 \
  examples/glm45-air_full-async_retool-modal/run_async.sbatch
# -> Submitted batch job 23378
```

**Config snapshot (delta from defaults in `run_async.sh`):**
- 12 nodes, 96 H200 GPUs total. 4 actor nodes (32 GPU) + 8 rollout nodes (64 GPU). No `--colocate`.
- Train parallelism: TP=4 PP=1 CP=1 EP=8 ETP=1, sequence-parallel, recompute=full/uniform/1.
- Rollout: 1 SGLang engine per rollout node, TP=8, mem-fraction-static=0.85, deepep a2a backend.
- MoE: `--moe-enable-deepep` added on Megatron side too.
- Modal sandbox pool size 64, per-exec timeout 60 s.
- `NUM_ROLLOUT=10` (smoke target — exit after 10 train iterations).

**Outcome:** FAILED at sbatch line 32: `scontrol: command not found`.

**Root cause:** I'd put `#SBATCH --container-image=docker://slimerl/slime:...`
at the sbatch level, which means the orchestrator script ran INSIDE the
slime docker image — and that image does not ship with slurm CLI tools.
`scontrol show hostnames` is needed to expand `$SLURM_JOB_NODELIST` into a
hostname array.

**Fix (Run 3 below):** Move pyxis to the per-`srun` level. The sbatch
orchestrator runs on the host (has slurm tools), and each `srun` step
enters the container via `--container-image=...` /
`--container-mounts=...`. Standard pyxis pattern.

Also forwarded `NUM_ROLLOUT`, `ROLLOUT_BATCH_SIZE`, `MODAL_CONFIG_PATH`
explicitly via `--export=ALL,...` on the training-launch srun step so they
reach the in-container shell.

---

## Run 3 — host-orchestrated, per-srun pyxis, job 23379

**Date:** 2026-04-25 23:48 UTC

**Cmd:** `sbatch --export=ALL,NUM_ROLLOUT=10 examples/glm45-air_full-async_retool-modal/run_async.sbatch`

**Outcome:** FAILED.

**Error 1:** `pip install modal` exited with
`error: externally-managed-environment` from PEP 668. The slime container's
system Python 3.12 (debian) marks the environment as externally managed.

**Error 2 (latent — would have hit next):** `Ray head: worker-13 (169.254.0.1)`.
`hostname -I` on this Soperator/k8s cluster returns the link-local
169.254.x.x address first, which is not routable across nodes — workers
would fail to join the head.

**Fixes (Run 4 below):**
- Add `--break-system-packages` to the `pip install modal` step.
- Pick the first non-link-local IPv4 from `hostname -I` when resolving the
  Ray head address; fall back to the hostname if none.

---

## Run 4 — pip break-system-packages + non-link-local IP, job 23380

**Date:** 2026-04-25 23:53 UTC

**Cmd:** `sbatch --export=ALL,NUM_ROLLOUT=10 examples/glm45-air_full-async_retool-modal/run_async.sbatch`

**Outcome:** STUCK for 1h, cancelled.

Ray came up cleanly across all 12 nodes (head on `worker-13` at routable
`10.49.172.11`, all 11 workers reported `Ray runtime started.`). But the
training-launch srun on `worker-13` looped forever with:

```
srun: Job 23380 step creation temporarily disabled, retrying (Requested nodes are busy)
```

**Root cause:** Slurm step exclusivity. The Ray-head srun uses
`ray start ... --block` to hold the slot for the duration of the cluster.
A second srun targeting the same node can't allocate resources unless
explicitly told it may overlap.

**Fix (Run 5 below):** Add `--overlap` to the training-launch srun (step 5)
and the final cleanup srun (step 6); both run on nodes still held by
`ray start --block`.

---

## Run 5 — --overlap on launch srun, job 23409

**Date:** 2026-04-26 00:55 UTC

**Cmd:** `sbatch --export=ALL,NUM_ROLLOUT=10 examples/glm45-air_full-async_retool-modal/run_async.sbatch`

**Outcome:** FAILED.

Ray cluster came up cleanly across all 12 nodes. The training driver
reached `train_async.py:22 router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())`
and the `RolloutManager` actor segfaulted before it could respond.

**Stack (RolloutManager pid 2360655):**
```
opentelemetry::v1::sdk::metrics::PeriodicExportingMetricReader::CollectAndExportOnce
opentelemetry::v1::exporter::otlp::OtlpGrpcMetricExporter::Export
grpc::internal::BlockingUnaryCallImpl::BlockingUnaryCallImpl
grpc_core::AresClientChannelDNSResolver::AresRequestWrapper::OnHostnameResolved
... Fatal Python error: Segmentation fault
```

**Root cause:** Ray spins up an OpenTelemetry OTLP metrics exporter per
worker; when the OTLP collector endpoint is not reachable (which it
isn't on this cluster), the periodic-export background thread crashes
inside gRPC/c-ares DNS resolution and takes the whole actor down.

**Fix (Run 6 below):** Disable OpenTelemetry SDK at the env-var level via
the Ray runtime env:
- `OTEL_SDK_DISABLED=true`
- `OTEL_TRACES_EXPORTER=none`
- `OTEL_METRICS_EXPORTER=none`
- `OTEL_LOGS_EXPORTER=none`

---

## Run 6 — OTel disabled in runtime_env, job 23410

**Date:** 2026-04-26 01:04 UTC

**Cmd:** `sbatch --export=ALL,NUM_ROLLOUT=10 examples/glm45-air_full-async_retool-modal/run_async.sbatch`

**Outcome:** FAILED at warmup. Got past every Ray issue from earlier runs:
12-node Ray cluster up, 8 SGLang engines created (TP=8, EP=8), all 47 of
47 weight shards loaded across all engines, 4 Megatron actors loaded
(124 GB used per H200 on actor nodes, matches expected ~13 B params/rank
+ optimizer + Adam state).

When the first warmup forward kicked off SGLang's MoE EP layer, every
engine hit:

```
File "/sgl-workspace/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py",
  line 242, in run_moe_core
  assert False, "forward_deepgemm_masked is deprecated"
AssertionError: forward_deepgemm_masked is deprecated
```

**Root cause:** `--sglang-moe-a2a-backend deepep --sglang-deepep-mode auto`
routes through `forward_deepgemm_masked`, which the SGLang shipped in
`slimerl/slime:nightly-dev-20260425a` has hard-asserted as deprecated.
The pattern works in slime's other MoE scripts (run-deepseek-r1, glm4.7-355B)
which were exercised against an older SGLang build.

**Fix (Run 7 below):** Drop both SGLang deepep flags; the default a2a
backend avoids the deprecated path. `--moe-enable-deepep` on the Megatron
training side is unrelated and stays — Megatron's deepep is a separate
implementation.

---

## Run 7 — drop SGLang deepep flags

**Date:** 2026-04-26 01:18 UTC

**Cmd:** `sbatch --export=ALL,NUM_ROLLOUT=10 examples/glm45-air_full-async_retool-modal/run_async.sbatch`

**Job:** 23411. PD; monitoring next.

**Outcome:** FAILED at Megatron actor init.

Big progress vs Run 6: SGLang fully initialized this time (no more
`forward_deepgemm_masked` assert), CUDA graphs captured, weight load done.
Failure moved to the Megatron actor side:

```
File "/root/Megatron-LM/megatron/core/transformer/transformer_config.py",
  line 1023, in __post_init__
  raise ValueError("DeepEP backend is only supported with flex token dispatcher.")
ValueError: DeepEP backend is only supported with flex token dispatcher.
```

**Root cause:** I'd added `--moe-enable-deepep` to PERF_ARGS to match
`scripts/run-deepseek-r1.sh` and `scripts/run-glm4.7-355B-A32B.sh`. But
Megatron's TransformerConfig requires `--moe-token-dispatcher-type flex`
when DeepEP is enabled, and the GLM-4.5 model config sets `alltoall`.
The reference scripts work because they target a different model config
that uses flex dispatch.

**Fix (Run 8 below):** Drop `--moe-enable-deepep` from PERF_ARGS. The model
config's default `alltoall` dispatcher works without it. The original
retool example also doesn't set this flag.

---

## Run 8 — drop Megatron deepep

**Date:** 2026-04-26 01:28 UTC

**Cmd:** `sbatch --export=ALL,NUM_ROLLOUT=10 examples/glm45-air_full-async_retool-modal/run_async.sbatch`

**Job:** 23412. PD; monitoring next.

**Outcome:** got the furthest yet — full bring-up succeeded:
- Ray cluster up (12 nodes)
- 8 SGLang engines: weight load 47/47 + CUDA graph capture done (~5 min total)
- 4 Megatron actors: torch_dist load complete, optimizer state allocated (128 GB / H200)
- `Timer update_weights end (elapsed: 14.1s)` — first weight push from
  actor → SGLang succeeded
- `Continuous async rollout worker started` from fully_async_rollout.py
- First rollout group submitted to RolloutManager…

…and immediately failed in the worker with:

```
File "/root/slime/examples/glm45-air_full-async_retool-modal/modal_tool_sandbox.py",
  line 28, in <module>
    import modal
ModuleNotFoundError: No module named 'modal'
```

**Root cause:** Each pyxis `srun --container-image=...` is an ephemeral
container. My sbatch installed Modal in a dedicated step #2 srun — but
that container was thrown away when the srun exited. Steps #3-#4 (`ray
start --head`, `ray start --address=...`) launched FRESH containers
without modal. The Ray actors spawned later inherited those containers'
Python env, so `import modal` failed at first rollout.

**Fix (Run 9 below):** install Modal *inside* the same `bash -c` that runs
`ray start` for both head and workers. The Ray daemon's process tree —
which hosts every Ray actor — then has modal in its env.

`scancel 23412` and resubmitted.

---

## Run 9 — install modal in the ray-start container, job 23413

**Date:** 2026-04-26 01:45 UTC

**Cmd:** `sbatch --export=ALL,NUM_ROLLOUT=10 examples/glm45-air_full-async_retool-modal/run_async.sbatch`

**Status:** PD on resources (waiting for 23412 to release nodes).

**Note:** This run inherits the `add_slime_metrics.*` paths I wired in
during the 23412 lifetime, so we'll see `sandbox/*`, `prompt_len/*`,
`throughput/*` keys in stdout (and wandb if a key were set).

**Outcome:** FAILED at training step 0 (compute_log_prob phase).

Got further than every prior run: cleared SGLang init, Megatron actor load,
14.1s update_weights push, completed rollout 0 (256 samples through Modal
sandbox + tool-call loop). Then **OOM at the first `compute_log_prob` call**,
inside Megatron's input-embedding `F.embedding`:

```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 317.62 GiB. GPU 1 has a total capacity of 139.81 GiB.
File "/root/Megatron-LM/megatron/core/tensor_parallel/layers.py", line 295,
  output_parallel = F.embedding(masked_input, self.weight)
```

Trace path: `MegatronTrainRayActor.train` → `train_actor` →
`compute_log_prob` → `forward_only` → `forward_step` → `model(...)` →
`_preprocess` → `language_model_embedding.forward` → `word_embeddings(input_ids)`.

**Root cause (best estimate):** dynamic batching's micro-batch sizing
isn't cutting compute_log_prob's input down to `max_tokens_per_gpu=16384`.
317 GiB / (4096 × 2 B) = 38.7 M tokens — i.e. roughly 18× the full rollout
batch (256 samples × 8192 max-resp ≈ 2.1 M tokens). Could be a TP/EP
broadcast multiplying the input, or compute_log_prob feeding the full
unsliced rollout into one forward.

Pre-train memory budget was already tight: 128 GB / 139.81 GB H200 used
just for weights + optimizer + Adam state, leaving 12 GB headroom. Any
oversized activation request blows through it.

**Fix (Run 10 below):** Three coordinated cuts to memory pressure:
1. `--max-tokens-per-gpu 16384 → 4096`
2. `--rollout-max-response-len 8192 → 4096` (halves per-sample size)
3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (mitigates fragmentation)

Kept TP=4 EP=8 PP=1 (matches the proven conversion + saves bring-up cost).
Submitted with `NUM_ROLLOUT=20` per user request — skipping the 10-step
smoke as we've already burned enough bring-up cycles.

---

## Run 10 — OOM mitigations + 20-step target, job 23428

**Date:** 2026-04-26 02:10 UTC

**Cmd:** `sbatch --export=ALL,NUM_ROLLOUT=20 examples/glm45-air_full-async_retool-modal/run_async.sbatch`

**Outcome:** FAILED. **OOM gone, new bug surfaced.**

What worked: SGLang init, Megatron actor load, update_weights (12.9s),
rollout 0 produced samples — and one of them solved the math problem
correctly via the code interpreter (`'score': 1.0, 'acc': True`,
predicted answer `41` for the rational-fraction problem). Modal sandbox
+ tool-call loop is functional end-to-end.

What broke: many `Returned aborted group N to data buffer` lines from
the fully-async worker (samples aborted mid-rollout when
`update_weights()` swapped the SGLang weights), then per-sample asserts
on the worker:

```
AssertionError: Token/logp length mismatch at turn 0:
  1133 tokens vs 4525 logps
```

…then on the trainer side:

```
File "/root/slime/slime/backends/megatron_utils/cp_utils.py", line 226
  assert len(log_prob) == response_length
```

**Root cause:** `generate_with_retool.generate()` *appends* to
`sample.rollout_log_probs` instead of resetting. When fully-async aborts
an in-flight rollout and the same sample object is rolled out a second
time, the new logprobs get concatenated onto the old ones. The original
synchronous retool example never hits this — only the
abort-and-retry path that fully-async exposes triggers it.

Also got a runtime warning: `PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead`.

**Fixes (Run 11 below):**
1. `add_slime_metrics.generate()` now explicitly resets
   `sample.rollout_log_probs = None`, `sample.tokens = []`,
   `sample.response_length = 0`, `sample.response = ""`,
   `sample.loss_mask = []` before calling the retool generate function.
2. `PYTORCH_CUDA_ALLOC_CONF` → `PYTORCH_ALLOC_CONF` in the runtime env.

---

## Run 11 — sample-state reset + new env var name, job 23429

**Date:** 2026-04-26 02:38 UTC

**Cmd:** `sbatch --export=ALL,NUM_ROLLOUT=20 examples/glm45-air_full-async_retool-modal/run_async.sbatch`

**Status:** PD on resources.

**Outcome:** stuck — 1/32 group completed in 50+ min of rollout 0,
then cancelled. Diagnosis revealed Qwen-style prompt template mismatched
GLM-4.5-Air's native chat format → model rambled in `<think>` blocks and
emitted broken JSON-style tool calls retool couldn't parse.

---

## Run 12 — GLM-native chat template + tool-call grammar, job 23766

**Date:** 2026-04-27 (after 21-test offline pass for the rewritten parser)

**What changed in `generate_with_retool.py`** (legacy code preserved as
commented-out blocks):
- `format_conversation_with_tools` now delegates to
  `tokenizer.apply_chat_template(messages, tools=…, add_generation_prompt=True)`
  → prompt uses GLM's native `[gMASK]<sop><|system|>…<|user|>…<|assistant|>\n<think></think>\n` framing.
- `postprocess_predictions` parses GLM-native tool calls:
  `<tool_call>name\n<arg_key>k</arg_key>\n<arg_value>v</arg_value>\n…\n</tool_call>`
  (no JSON; key/value pairs).
- `execute_predictions` wraps tool output in
  `\n<|observation|>\n<tool_response>\n{result}\n</tool_response>\n<|assistant|>\n<think></think>\n`
  — natively trained format, with the assistant turn re-opener appended so
  the model continues coherently.
- Tool-call counting + wandb markers updated from `<interpreter>` →
  `<tool_response>` and `<|im_start|>system` → `<|system|>`.
- Removed unused `from jinja2 import Template`.

**Pre-launch verification:** `test_glm_postprocess.py` — 21/21 pass on
realistic native and edge-case inputs (multiline code, nested-brace
answers, unclosed tool calls, multiple tool calls, unknown tool names).

**Other deltas vs. Run 11:** `--rollout-max-response-len` back to 8192
(was cut to 4096 chasing the 317 GiB OOM that turned out to be a
parser/template artifact, not real memory pressure).

**Cmd:** `sbatch --export=ALL,NUM_ROLLOUT=20 examples/glm45-air_full-async_retool-modal/run_async.sbatch`

**Status:** PD on resources.

---

## Side-quest — add_slime_metrics.py

**Date:** 2026-04-26 01:30 UTC

While 23412 is running, wrote `add_slime_metrics.py` to surface the
metrics slime doesn't emit by default:

| Metric | How |
|---|---|
| `sandbox/acquire_time_per_call_mean` | Monkey-patched `_acquire` in modal_tool_sandbox; ContextVar bucket per-sample. |
| `sandbox/exec_time_per_call_mean` | `total_execute_code_time - acquire_time` per bucket. |
| `sandbox/total_time_per_sample_mean` | Sum bucket totals / n_samples. |
| `sandbox/calls_per_sample_mean` | Bucket call count / n_samples. |
| `sandbox/error_rate` | Fraction of execs returning `Error: ...`. |
| `prompt_len/{mean,std,p50,p90,p95,...}` | Stashed `sample.prompt_token_count` in `generate()`. |
| `throughput/samples_per_sec` | `len(samples) / rollout_time`. |
| `throughput/effective_tokens_per_sec` | `Σ effective_response_length / rollout_time`. |

Wired in via three `--custom-*` flags so the CURRENT run keeps using the
plain retool path; the NEXT submission picks up the instrumented path
automatically:

```
--custom-generate-function-path add_slime_metrics.generate
--custom-rm-path add_slime_metrics.reward_func
--custom-rollout-log-function-path add_slime_metrics.custom_rollout_log_function
```

Slime's stock `rollout/*` and `perf/*` metrics are still emitted — the
custom log function calls slime's own helpers and then layers on top.

---

## Status while waiting for GLM-4.5-Air download

- Background bash task `bz7m31808`: at 27 GB / 7 safetensors after ~2 min.
- Slime docker image `slimerl/slime:nightly-dev-20260425a` will be pulled
  by pyxis on first submit — no pre-pull needed.
- `--num-rollout` and `--rollout-batch-size` made env-var overridable in
  `run_async.sh` so I can run a `NUM_ROLLOUT=10` smoke test first.
- Config catch (pre-launch): slime asserts
  `global_batch_size == rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`
  (slime/utils/arguments.py:1753-1757). With my initial `rollout_batch_size=64`,
  `n_samples_per_prompt=8`, `global_batch_size=256` → 64×8÷1=512 ≠ 256 would
  have hard-asserted. Set `rollout_batch_size=32` to match.

---
