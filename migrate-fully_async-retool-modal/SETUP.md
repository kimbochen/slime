# Fully-async retool + Modal sandbox — migration guide

This package is the working state of `examples/glm45-air_full-async_retool-modal/`
from a slime checkout, built up over ~12 iteration cycles to bring up:

- 12-node Slurm + pyxis training cluster
- Megatron-LM training (4 actor nodes) with fully-async rollout (8 SGLang nodes)
- Modal sandbox for tool-call code execution (with per-sample metrics)
- Custom metrics report layered on top of slime's stock wandb logging

The example was originally targeted at GLM-4.5-Air (106B-A12B). This guide
covers (a) standing it up on a fresh cluster and (b) retargeting to
**Qwen3-235B-A22B**, which has a different prompt template, parallelism,
and (most importantly) uses slime's `--colocate` 64-GPU layout instead
of the split actor/rollout layout we built for GLM-4.5-Air.

---

## Files in this package

| File | What it does | Origin |
|---|---|---|
| `README.md` | User-facing readme for the example | new |
| `requirements.txt` | `jinja2`, `psutil`, `modal>=0.65` | new |
| `modal_tool_sandbox.py` | Modal-backed async sandbox pool replacing the upstream `tool_sandbox.py`; per-call timing, `block_network=True` is the actual safety boundary | new |
| `generate_with_retool.py` | Custom generate function. **Two flavors of parsing live in this file**: the original Qwen-JSON style is commented out; the live code uses `tokenizer.apply_chat_template` + GLM-native `<tool_call>name\n<arg_key>k</arg_key>\n<arg_value>v</arg_value>\n</tool_call>` parsing. | rewritten |
| `add_slime_metrics.py` | Wraps `generate_with_retool.generate`, monkey-patches `modal_tool_sandbox` for per-sample timings, registers a `--custom-rollout-log-function-path` that emits richer wandb metrics (sandbox/*, prompt_len/*, throughput/*) | new |
| `metrics_report.py` | Standalone CLI that parses slime stdout logs and prints trainer/generator/system-efficiency tables. Self-bootstraps a uv venv via PEP-723. | new |
| `convert.sbatch` | One-shot HF → Megatron `torch_dist` conversion via pyxis (uses `slimerl/slime` image) | new |
| `run_async.sh` | Inner ray-job submitter. Sources `scripts/models/<model>.sh`, builds runtime_env, calls `ray job submit ... train_async.py`. | new |
| `run_async.sbatch` | Outer Slurm wrapper. Allocates 12 nodes, brings up Ray (head + 11 workers via per-srun pyxis), then runs `run_async.sh` on the head. | new |
| `test_modal_sandbox.py` | CPU-only smoke test for the Modal sandbox pool (4 sub-tests) | new |
| `test_glm_postprocess.py` | Offline parser unit tests for the GLM-native tool-call format (21 cases) | new |
| `EXPERIMENT_LOG.md` | Tail-first incident log of the 12-run bring-up. Read this if you hit a confusing failure — odds are it's already documented. | new |

---

## Cluster prerequisites

- **Slurm** with **pyxis** plugin (`srun --container-image=…`). Plain Docker on
  a non-Slurm node will need different bring-up logic — none of these scripts
  apply. Verify with `srun --container-image=help` returning pyxis options.
- **`enroot`** for image caching (pulled by pyxis automatically). First run
  pulls `slimerl/slime:nightly-dev-20260425a` (~tens of GB) and caches per
  node; subsequent runs are seconds.
- **Slurm partition** with H200 (or H100) nodes, 8 GPUs/node, ≥1.5 TB host RAM
  (the Megatron actor's optimizer state + activation buffers spill into host
  memory during distributed checkpoint load).
- **NFS** (or any shared filesystem) mounted identically on all nodes at the
  slime repo root. Container bind-mount `/home/.../slime:/root/slime` requires
  this. Datasets and checkpoints live under `mnt/` in the repo.
- **HuggingFace token** for any gated model (GLM-4.5, Qwen3-235B). Save to
  `/home/<user>/slime/.hf_token.txt` (mode 0600). Conversion + download steps
  read from there. **gitignore this file!**
- **Modal token** for the sandbox pool. Save to `/home/<user>/slime/.modal.toml`
  with `[default]` `token_id` and `token_secret`. The sbatch script forwards
  `MODAL_CONFIG_PATH` into the container's runtime_env.

---

## Step-by-step bring-up

### 0. Drop this dir into a slime checkout

The package is self-contained except for two paths it depends on inside slime:

- `slime/scripts/models/<model>.sh` (sourced by `run_async.sh`)
- `slime/examples/fully_async/fully_async_rollout.py` (used as
  `--rollout-function-path`)

Easiest: drop the package as a sibling under `examples/`:

```bash
cd <slime-repo>
cp -r migrate-fully_async-retool-modal examples/<your-name>
```

### 1. Verify Modal credentials

```bash
MODAL_CONFIG_PATH=$(pwd)/.modal.toml \
  python -c "import modal; modal.App.lookup('infx-slime-retool-sandbox', create_if_missing=True)"
```

If this returns without error, the SDK is wired up.

### 2. Run the Modal smoke test (CPU-only, no Slurm)

```bash
cd examples/<your-name>
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python "modal>=0.65"
MODAL_CONFIG_PATH=<slime-root>/.modal.toml \
  SLIME_MODAL_POOL_SIZE=4 SLIME_MODAL_PER_EXEC_TIMEOUT=10 \
  .venv/bin/python test_modal_sandbox.py
```

Should print `4 passed`. Validates: pool spin-up, concurrent execs,
`block_network=True` actually blocks, per-exec timeouts fire.

### 3. Run the parser unit tests (also offline)

```bash
./test_glm_postprocess.py     # uv run via shebang
```

Should print `21 passed, 0 failed`. **Do this every time you change
`generate_with_retool.py`** — the old retool example was pretty fragile
to template/parser drift and these tests catch most regressions.

### 4. Download datasets + model

Both gated models need `hf auth login --token "$(cat .hf_token.txt)"`.

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k \
    --local-dir mnt/data/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 \
    --local-dir mnt/data/aime-2024
hf download <model-repo> --local-dir mnt/checkpoints/<model-repo>
```

### 5. Convert weights → Megatron `torch_dist`

```bash
sbatch examples/<your-name>/convert.sbatch
```

This allocates 1 GPU node, pulls the slime image via pyxis, mounts the
slime repo at `/root/slime`, runs `tools/convert_hf_to_torch_dist.py`.
Output lands at `mnt/checkpoints/<model-repo>_torch_dist/`.

### 6. Submit the training run

```bash
sbatch --export=ALL,NUM_ROLLOUT=20 examples/<your-name>/run_async.sbatch
```

`NUM_ROLLOUT=20` is the smoke-target step count. Bump to thousands for
real training. Logs land in `mnt/logs/infx-retool-async-<jobid>.out`.

### 7. Analyze metrics after the run

```bash
./examples/<your-name>/metrics_report.py --job <jobid>
# or
./examples/<your-name>/metrics_report.py --log mnt/logs/infx-retool-async-<jobid>.out --json
```

`--latest` for just the most recent rollout.

---

## Iteration workflow we used (recommended for new-cluster bring-up)

You will hit cluster-specific failures we didn't see. Use the same loop we
ran during the 12-run bring-up — it kept the cycle time tight and the cost
of each failure low:

1. **One hypothesis per run.** Resist the urge to bundle "let me also try X
   while I'm at it" into a launch — when something fails, you want to know
   exactly which change caused it. Apply one targeted fix, submit, observe.

2. **Pre-launch sanity checks before any sbatch.**
   - `./test_modal_sandbox.py` if you touched `modal_tool_sandbox.py`
   - `./test_glm_postprocess.py` if you touched `generate_with_retool.py`
   - `sbatch --test-only run_async.sbatch` to confirm the allocation parses
   - `python -c "import ast; ast.parse(open('run_async.sh').read())"` (or
     `bash -n run_async.sh`) for the shell

   On the actual cluster these took ~5 seconds each and caught syntax /
   regex regressions that would have wasted a 17-minute pyxis bring-up.

3. **Watch for the *right* event, not progress noise.** SGLang and
   Megatron print thousands of warning + status lines per rollout. The
   only events worth alerting on are:
   - state transitions (PD → RUNNING → COMPLETING)
   - `^perf <N>:` lines (slime emits one per train step — this is the
     authoritative "we did a step" signal)
   - real Python tracebacks (`^Traceback`, `^[A-Z][a-z]+Error:`,
     `OutOfMemoryError`, `Segmentation fault`)
   The `Monitor` tool we used polls `squeue` + `grep` over the slurm log
   on a 90 s cadence. Avoid heuristic phase regexes (`MegatronTrainRayActor`,
   `update_weights`) — they fire false positives on the args dump and
   never actually correlate with progress.

4. **Tail-first incident log.** Every run gets one entry in
   `EXPERIMENT_LOG.md` with: config snapshot delta vs. previous run,
   submit command, outcome (one of: PASS / FAILED-at-bringup / FAILED-at-step-N
   / STUCK), the verbatim error excerpt, root-cause analysis, and the
   targeted fix going into the next run. New runs go at the bottom so
   `tail EXPERIMENT_LOG.md` shows current state. This file pays for itself
   the first time you have to revisit a failure mode three runs later.

5. **Cancel STUCK runs aggressively.** Slurm "RUNNING" with no progress
   markers for >2× the expected time is wasted compute. We hit one stuck
   run for an hour with 1/32 groups completed before cancelling — should
   have killed it sooner. If `perf 0:` hasn't fired within ~25 min of
   submission (post-bring-up budget), something is wrong.

6. **Bring-up budget.** On this cluster: pyxis pull ~3 min (cached after
   first run), Ray bring-up ~30 s, SGLang weight load ~5 min, CUDA-graph
   capture ~40 s, Megatron actor weight load ~3 min, first
   `update_weights` push ~13 s, first rollout's wall time is variable but
   should be O(minutes), not O(tens of minutes). End-to-end first
   `perf 0:` lands at roughly **t+18 min** when healthy.

7. **Each fix lands in two places.**
   - The actual code/config change (`run_async.sh`, `add_slime_metrics.py`,
     etc.).
   - A new entry in §"The N quirks we hit" of `SETUP.md`. Future-you and
     anyone migrating to a third cluster gets to skip this debug session.

---

## Cluster naming convention (NOT model-specific)

Internal cluster policy that bit us multiple times:

- Slurm `--job-name` MUST be prefixed `infx-`.
- Slurm `--job-name` MUST NOT contain `rl`.
- Slurm `--job-name` MUST NOT contain a model identifier (`glm`, `qwen`, …).
- Filenames may include model identifiers; only the slurm job name is restricted.

Current names in the package: `infx-retool-async`, `infx-retool-convert`. Both safe.

---

## The 11 quirks we hit (READ THIS BEFORE DEBUGGING ANYTHING)

These are in chronological order from `EXPERIMENT_LOG.md`. Every one of them
will hit you again on a new cluster unless you preempt.

1. **HF repo naming for the GLM-4.5 series.** `zai-org/GLM-4.5` is the 355 B
   flagship (~730 GB shards). The 106B-A12B "Air" variant is `zai-org/GLM-4.5-Air`.
   The slime model file `scripts/models/glm4.5-106B-A12B.sh` matches *the
   spec*, not the marketing name. Always check `config.json` shape vs. the
   slime `.sh` after downloading.

2. **HF→torch_dist conversion silently overrides PP.**
   `tools/convert_hf_to_torch_dist.py:55-71` checks `if pipeline_model_parallel_size == 1
   and world_size > 1` and forcibly sets PP to world_size. So passing
   `--tensor-model-parallel-size 1 --expert-model-parallel-size 8 --pipeline-model-parallel-size 1`
   on 8 GPUs becomes `PP=8 EP=8 ETP=1`, which violates Megatron's
   `world_size %% (ETP × EP × PP) == 0` (1×8×8=64 ≠ 8). **Fix:** drop EP/ETP
   flags from the convert command and let auto-PP=world_size do the sharding.
   See `convert.sbatch` for the working invocation.

3. **`scontrol` is not in the slime container.** Don't put
   `#SBATCH --container-image=…` at the sbatch level — the orchestrator script
   runs *inside* the container and can't expand `$SLURM_JOB_NODELIST`. Put
   `--container-image=` on each `srun` instead. Same goes for
   `--container-mounts=`, `--container-workdir=`. See `run_async.sbatch`.

4. **PEP 668 blocks `pip install` in the slime container.** Add
   `--break-system-packages`. Idempotent: `python -c "import modal" 2>/dev/null
   && exit 0`.

5. **`hostname -I` returns link-local first on Soperator/k8s nodes.**
   The first space-separated IP is `169.254.x.x`, which is not routable across
   nodes — Ray workers can't reach the head. Filter with
   `grep -vE "^(169\.254|127\.|::)"` and fall back to the hostname.

6. **Slurm step exclusivity blocks the training-launch srun.**
   `ray start --block` holds the head's slot. The next srun on the same node
   loops on `step creation temporarily disabled (Requested nodes are busy)`
   forever. **Fix:** `--overlap` on any srun that targets a node already held
   by `ray start --block` (training launch + final cleanup).

7. **OpenTelemetry kills the RolloutManager actor.** Ray spins up an OTLP
   metrics exporter background thread; on clusters without an OTLP collector
   it crashes inside gRPC/c-ares DNS resolution and segfaults the actor.
   Symptom: `RolloutManager` dies with no Python traceback, just a giant
   gRPC stacktrace ending in "Fatal Python error: Segmentation fault." Fix:
   set in the runtime_env env_vars: `OTEL_SDK_DISABLED=true`,
   `OTEL_TRACES_EXPORTER=none`, `OTEL_METRICS_EXPORTER=none`,
   `OTEL_LOGS_EXPORTER=none`.

8. **SGLang's deepep MoE backend asserts deprecated.**
   `--sglang-moe-a2a-backend deepep --sglang-deepep-mode auto` (lifted from
   `scripts/run-deepseek-r1.sh`) hits
   `assert False, "forward_deepgemm_masked is deprecated"` in
   `sglang/srt/layers/moe/ep_moe/layer.py:242` on the SGLang in this image.
   **Drop both flags.** Default a2a backend works.

9. **Megatron deepep requires flex token dispatcher.**
   `--moe-enable-deepep` on the training side requires
   `--moe-token-dispatcher-type flex`. The GLM-4.5-Air model config sets
   `alltoall`. Drop `--moe-enable-deepep` (or override the dispatcher).

10. **Pyxis containers are ephemeral per srun.** A separate
    "install modal" srun installs into a throwaway container — Ray actors
    later spawn in different containers and `import modal` fails. **Fix:**
    install Modal *inside* the same `bash -c` that runs `ray start`.

11. **Retool `generate()` doesn't reset `sample.rollout_log_probs` on retry.**
    Fully-async aborts in-flight rollouts when `update_weights()` swaps
    SGLang's weights. The aborted samples go back to the buffer and get
    re-rolled with stale state, so the second attempt's logprobs append to
    the first's, then `len(response_token_ids) != len(rollout_log_probs)`
    fires. **Fix lives in `add_slime_metrics.generate()`** which resets all
    sample state before delegating.

12. **Bonus: chat-template mismatch tanks rollout speed.** Retool ships a
    Qwen-style hand-rolled Jinja template. If the model has a different
    native chat template, the model rambles in `<think>` blocks for
    thousands of tokens and emits unparseable tool calls; rollouts crawl
    (1 group / 50 min). **Fix:** use `tokenizer.apply_chat_template(...)` so
    the model gets prompts in the format it was trained on. Then update the
    parser regex to match the model's native tool-call grammar. We did this
    for GLM-4.5-Air; you'll need to do the equivalent for Qwen3-235B (see
    next section).

---

## Retargeting to Qwen3-235B-A22B

Slime ships a **fundamentally different layout** for Qwen3-235B than the
12-node split-actor/rollout one we built for GLM-4.5-Air. From
`scripts/run-qwen3-235B-A22B.sh`:

```
--actor-num-nodes 8
--actor-num-gpus-per-node 8
--rollout-num-gpus 64                  ← same 64 GPUs as actor
--colocate                             ← time-slice the same pool
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 4
--context-parallel-size 2
--expert-model-parallel-size 16
--expert-tensor-parallel-size 1
--rollout-num-gpus-per-engine 32       ← TP=32 per SGLang engine, 2 engines
```

So **64 GPUs total, colocated, no fully-async.** That's a different example
than this one. **You have two paths:**

### Path A: keep fully-async (this package's spirit)

Likely needs ~16 nodes total (8 actor + 8 rollout) for Qwen3-235B; 12 nodes
won't fit comfortably. Hasn't been validated. Edit `run_async.sbatch`'s
`#SBATCH --nodes=12 → 16` and `run_async.sh`'s `--actor-num-nodes 4 → 8` /
`--rollout-num-gpus 64 → 64`. **Drop `add_slime_metrics`'s sample reset
caveat** — fully-async aborts during update_weights are still a risk; keep
the fix.

### Path B: switch to colocated (slime's native 235B layout) — recommended

Drop fully-async entirely:

1. Use `train.py`, not `train_async.py`. Edit `run_async.sh`.
2. Add `--colocate`, set `--actor-num-nodes 8` and **drop `--rollout-num-gpus`
   and `--rollout-function-path`**. (slime auto-derives rollout GPU count
   from the actor pool when `--colocate`.)
3. Use slime's standard rollout, not `fully_async_rollout`. Drop
   `--rollout-function-path`. Keep `--custom-generate-function-path` and
   `--custom-rm-path` pointing at this dir.
4. **Sbatch nodes = 8** (not 12). Update `#SBATCH --nodes=8`.

### Common to both paths: parser changes

`generate_with_retool.py` currently parses **GLM-4.5-Air's** native
`<tool_call>name\n<arg_key>...</arg_key>\n<arg_value>...</arg_value>\n</tool_call>`
format. Qwen3-235B uses a **different native tool format**. Steps:

1. Read `mnt/checkpoints/Qwen/Qwen3-235B-A22B-Instruct/chat_template.jinja`
   and find the `tool_calls` branch. Confirm whether Qwen3 emits JSON or
   key/value pairs — last I checked, Qwen3 still uses Qwen-style
   `<tool_call>{"name": "code_interpreter", "arguments": {...}}</tool_call>`.
2. If JSON: **uncomment the legacy `postprocess_predictions`** in
   `generate_with_retool.py` (it's right above the GLM version with
   `_legacy` suffixes) and comment out the GLM version. Same for
   `postprocess_responses`.
3. The `<|observation|>...<|assistant|>\n<think></think>\n` re-opener in
   `execute_predictions._glm_observation` is GLM-specific. Change to
   Qwen's format: `<|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>\n<|im_start|>assistant\n`.
4. Re-run `./test_glm_postprocess.py` (rename to `test_qwen_postprocess.py`
   and update test inputs to match Qwen's grammar). **Don't skip this** —
   parser bugs cost us multiple hours of cluster time on GLM-4.5-Air.

### Memory budget sanity check for 235B on 64 H200

20 B/param × 235 B params = 4.7 TB of state across the cluster.
Per GPU: 4700 / 64 = 73 GB. H200 has 141 GB. Leaves ~65 GB for activations,
KV cache, NCCL buffers — comfortable. With colocated SGLang the GPU memory
pool gets time-shared, so add `--sglang-mem-fraction-static 0.7-0.8`.

---

## What to gitignore

```
.modal.toml
.hf_token.txt
.venv/
mnt/                  # checkpoints + datasets + logs are huge
**/__pycache__/
```

---

## Where to look when something breaks

| Symptom | Look at |
|---|---|
| Slurm job pending forever | `squeue -u $USER`, `scontrol show job <id>`, `sinfo` |
| Pyxis fails to pull image | `mnt/logs/infx-retool-async-<id>.err`, search `pyxis: error` |
| Ray actors die mid-run | `mnt/logs/infx-retool-async-<id>.out` for `RolloutManager` / `MegatronTrainRayActor` traces |
| Training step OOMs | Same log; search `OutOfMemoryError`. First thing to try: `--max-tokens-per-gpu 4096`, then add `PYTORCH_ALLOC_CONF=expandable_segments:True` to `runtime_env` |
| Rollouts hang at "Collected: 1/N" | Almost certainly chat-template / parser mismatch. Eyeball one rollout sample's text in the log; check whether the model's `<tool_call>` shape matches what `postprocess_predictions` expects. |
| Modal sandbox timeouts | `SLIME_MODAL_PER_EXEC_TIMEOUT` env, `SLIME_MODAL_POOL_SIZE` env. Confirm via `test_modal_sandbox.py`. |
| Metrics report empty | No `perf <N>:` lines in the log → no train step completed. Use the metrics report's per-rollout view instead, then fix bring-up. |

---

## Quick reference: env vars consumed by this package

| Var | Default | Where read |
|---|---|---|
| `MODAL_CONFIG_PATH` | `<slime-root>/.modal.toml` | `run_async.sh`, `run_async.sbatch` (forwarded into Ray runtime_env) |
| `SLIME_MODAL_APP` | `infx-slime-retool-sandbox` | `modal_tool_sandbox.py` |
| `SLIME_MODAL_POOL_SIZE` | 64 | `modal_tool_sandbox.py` |
| `SLIME_MODAL_PER_EXEC_TIMEOUT` | 60 | `modal_tool_sandbox.py` |
| `SLIME_MODAL_SANDBOX_WALLCLOCK` | 3600 | `modal_tool_sandbox.py` |
| `NUM_ROLLOUT` | 3000 | `run_async.sh` (forwarded via `sbatch --export=ALL,NUM_ROLLOUT=…`) |
| `ROLLOUT_BATCH_SIZE` | 32 | `run_async.sh` |
| `WANDB_KEY` | unset | `run_async.sh` (if set, wandb is enabled; otherwise stdout-only) |
| `CKPT_ROOT` | `${SLIME_ROOT}/mnt/checkpoints` | `run_async.sh` |
| `DATA_ROOT` | `${SLIME_ROOT}/mnt/data` | `run_async.sh` |
| `MEGATRON_LM_PATH` | `/root/Megatron-LM` (in container) | `run_async.sh` |
| `OTEL_*_EXPORTER`, `OTEL_SDK_DISABLED` | `none` / `true` | `run_async.sh` runtime_env (DO NOT REMOVE — see quirk #7) |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | `run_async.sh` runtime_env |

## Metrics: what gets logged, and how to read it

This package layers extra metrics on top of slime's stock wandb output via
two pieces:

- **`add_slime_metrics.py`** — wraps `generate_with_retool.generate` and
  monkey-patches `modal_tool_sandbox` so per-sample sandbox timings get
  stashed on the `Sample` via an asyncio `ContextVar`. Registered through
  three CLI flags in `run_async.sh`:
  --custom-generate-function-path     add_slime_metrics.generate
  --custom-rm-path                    add_slime_metrics.reward_func
  --custom-rollout-log-function-path  add_slime_metrics.custom_rollout_log_function

The custom log function calls slime's own `compute_metrics_from_samples`
+ `compute_perf_metrics_from_samples` first (so you keep all stock
`rollout/*`, `perf/*` keys) and then layers on:

| Added key | Source |
|---|---|
| `prompt_len/{mean,std,p50,p90,p95,p99}` (tokens) | `sample.prompt_token_count = len(sample.tokens) - sample.response_length` |
| `throughput/samples_per_sec` | `len(samples) / rollout_time` |
| `throughput/effective_tokens_per_sec` | `Σ effective_response_length / rollout_time` |
| `sandbox/acquire_time_per_call_mean` | monkey-patched `_acquire` records pool wait into bucket |
| `sandbox/total_time_per_sample_mean` | bucket sums divided by sample count |
| `sandbox/calls_per_sample_mean` | `execute_code_calls / n_samples` |
| `sandbox/error_rate` | fraction of execs that returned `Error: …` |

All sandbox metrics use a per-coroutine `ContextVar` so concurrent
rollouts don't cross-contaminate.

- **`metrics_report.py`** — standalone CLI that parses the slime stdout
log (no wandb required) and renders four tables:

1. **Trainer config** (static, read from the args dump): num GPUs,
   parallelism (TP/PP/CP/EP/ETP), `global_batch_size`,
   `max_tokens_per_gpu`, recompute setting.
2. **Generator config** (static): rollout num GPUs, per-engine TP,
   number of SGLang engines, samples_per_step (= rollout_batch_size ×
   n_samples_per_prompt), `rollout_max_response_len`.
3. **Trainer per-step**: `step_time`, `train_time`, `wait_time`,
   `wait_ratio` (= `1 - actor_train_time/step_time` — the fully-async
   overlap signal), `weight_broadcast_s` (parsed from
   `Timer update_weights end (elapsed: …s)`), `samples/s`, `tok/s`,
   `TFLOP/s`, `MFU%` (vs. H200 BF16 peak 989 TFLOPS).

   Note: **trainer throughput uses `actor_train_time`, not `step_time`**
   — i.e. it excludes the wait-on-rollout window. So `samples/s` here
   is the raw compute rate; the wall-clock rate is lower.
4. **Generator per-step**: `rollout_s`, `samples/s`, `infer_s/sample`
   (= `rollout_time / samples`), `sandbox_s/sample`, `calls/sample`,
   `err%`, mean prompt / response / effective-response length **in
   tokens**, mean tool-call rounds.
5. **System efficiency**: `gen/train ratio` (= `rollout_time /
   actor_train_time`; >1 means rollout is the bottleneck and
   fully-async is paying off), `prefix_cache_hit`, `truncated`,
   `repetition`, p95 prompt + response.

Plus a **steady-state mean** row averaging the last 3 rollouts so you
can read off the typical step quickly.

Run:
  ./metrics_report.py --job
  ./metrics_report.py --log
  ./metrics_report.py --job  --latest    # just the most recent rollout
  ./metrics_report.py --job  --json      # machine-readable

PEP-723 inline metadata declares `numpy` and `rich` deps; the shebang
invokes `uv run --script` which builds the venv on first run.

### What to actually watch

- **`perf/wait_time_ratio`** is the single most useful number for "is
fully-async paying off?" — close to 0 means the trainer is fully
busy (rollout is overlapping perfectly); close to 1 means the trainer
is idling waiting for rollouts.
- **`gen/train ratio`** answers the same question from the rollout side
— if it's >>1, you'd benefit from more rollout GPUs (or fewer actor
GPUs).
- **`sandbox/error_rate`** spiking (>5%) usually means Modal sandbox
pool exhaustion or `SLIME_MODAL_PER_EXEC_TIMEOUT` too aggressive.
- **`rollout/truncated_ratio`** > ~0.2 means `--rollout-max-response-len`
is too low for the model's natural output distribution; you'll see
training instability since most samples never produce an `Answer`.
- **`MFU%`** below ~25% on H200 → check `--max-tokens-per-gpu` (too low
underutilizes), `--recompute-num-layers` (too high wastes flops),
parallelism balance.
- **`zero_std/count_*`** (slime stock) > a few per rollout means GRPO
variance collapse — every sample in a group got the same reward, so
advantage = 0. Sign you need to lower temperature or up
`n_samples_per_prompt`.

### Wandb (optional)

`run_async.sh` only enables wandb if `WANDB_KEY` is set in the sbatch
env:

sbatch --export=ALL,NUM_ROLLOUT=20,WANDB_KEY=xxx run_async.sbatch

When set, slime writes the same metric dict that `add_slime_metrics`
produces to wandb at `step_key=rollout/step`, with project
`slime-retool-async` and group `glm45-106b-a12b`. The static config
(num GPUs, parallelism, batch sizes, max_tokens_per_gpu, etc.) lands
as `wandb.config` automatically. Without `WANDB_KEY` everything is
stdout-only — that's why `metrics_report.py` exists.
