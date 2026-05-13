# Qwen3-235B-Thinking + SWE-bench Verified agentic coding

A fully-async slime example that trains Qwen3-235B-A22B-Thinking-2507-FP8 to
solve real GitHub bug fixes on **SWE-bench Verified**, using **per-instance
Modal sandboxes** as the execution environment for the agent's tools.

Built on top of slime's `examples/fully_async/` driver and modeled after
`examples/qwen3-235b-thinking_fullasync_retool_modal/`. Differences from
the retool example are in [Comparison vs. retool](#comparison-vs-retool).

```
                 ┌──────────────────────────────────────────────────────────────┐
                 │             slime rollout (custom-generate path)             │
                 │                                                              │
   prompt_data   │   for each sample (= one SWE-bench instance):                │
   (JSONL with   │     async with coding_sandbox.session(instance_id) as sb:    │
   instance_id)──┼───►  for turn in range(max_turns):                           │
                 │       sglang.generate → parse <tool_call>{...}</tool_call>   │
                 │       sb.run_command / read_file / write_file /              │
                 │         apply_patch / run_tests / submit                     │
                 │       feed result back as <tool_response> observation        │
                 │     stash test verdict on sample for reward_func             │
                 └──────────────────────────────────────────────────────────────┘
                                  ▲                              │
                                  │                              ▼
                          ┌───────┴──────────┐         ┌──────────────────────────┐
                          │ Modal app:        │ pulls   │ Epoch AI per-instance    │
                          │ infx-slime-       ├────────►│ Docker registry:         │
                          │ swebench-sandbox  │         │ ghcr.io/epoch-research/  │
                          │ (Sandbox.create)  │         │   swe-bench.eval.x86_64. │
                          │                   │         │   <instance_id>:latest   │
                          └───────────────────┘         └──────────────────────────┘
```

## What's in here

```
coding_sandbox.py             Modal-backed per-instance sandbox library
generate_with_codingagent.py  Slime custom-generate hook: prompt build,
                              tool-call parser, multi-turn agent loop, reward
add_slime_metrics.py          ContextVar-based per-tool / per-sandbox /
                              swebench-outcome / policy-staleness instrumentation
                              + custom_rollout_log_function for wandb
smoke_test_sandbox.py         End-to-end sandbox lifecycle test (real
                              Epoch AI image, ~30s including image pull)
test_codingagent.py           Parser + observation + dispatch unit tests
test_add_slime_metrics.py     22 unit tests for the metrics wrapper
gen_prompt_data.py            Pulls N SWE-bench Verified instances into JSONL
run_swe.sh                    Canonical launcher (64K context: TP=4 PP=2 CP=4)
run_swe.sbatch                Slurm wrapper for the canonical launcher

results/                      Per-run result packages (one subdir per run)
└── swe-env-64k-30045/        Latest 20-rollout run with add_slime_metrics
    ├── README.md, RESULTS.md, metrics.json, metrics_report.txt,
    └── compile_metrics_json.py, render_metrics_report.py

context-scaling/              16K / 32K / 64K / 128K context-cap experiments
├── run_swe_16k.{sh,sbatch}   baseline (TP=4 PP=4 CP=2, max_tok/gpu=8K)
├── run_swe_32k.{sh,sbatch}   (TP=4 PP=4 CP=2, max_tok/gpu=16K)
├── run_swe_64k.{sh,sbatch}   (TP=4 PP=2 CP=4, max_tok/gpu=16K) — same config
                              promoted to top-level run_swe.sh
├── run_swe_128k.{sh,sbatch}  (TP=4 PP=2 CP=4, max_tok/gpu=32K) — stretch
├── COMPARISON.md             cross-tier headline + analysis
└── README.md

pd/                           Prefill/Decode-disaggregated variant
├── run_swe_pd.{sh,sbatch}    64K config + --sglang-config sglang_pd.yaml
├── sglang_pd.yaml            5 prefill × TP=8 + 3 decode × TP=8 layout
├── mooncake_ib_per_gpu.json  per-GPU HCA binding (workaround for
                              get_ib_devices_for_gpu comma-list parser bug)
└── README.md                 STATUS: blocked on upstream Mooncake PD bug
                              (see mnt/mooncake_bug_report/BUG_REPORT.md)
```

## Sandbox design (`coding_sandbox.py`)

**One sandbox per agent rollout, not per shell command.** A single Modal
sandbox persists for the duration of one SWE-bench instance's multi-turn
trajectory, then terminates. Tinker-cookbook's sandbox creates per-request;
we don't, because applying patches + running tests + iterating only makes
sense on shared filesystem state across turns.

**Per-instance Docker images** — every SWE-bench Verified instance has a
prebuilt image on Epoch AI's ghcr.io registry (`ghcr.io/epoch-research/
swe-bench.eval.x86_64.<instance_id>:latest`). Each one already has the
repo cloned to `/testbed`, dependencies installed in a conda env at
`/opt/miniconda3`, and a `/eval.sh` that runs the FAIL_TO_PASS +
PASS_TO_PASS test set with the right invocation. Pulling a fresh image
takes ~30s the first time; Modal caches it after.

**General-purpose tool surface** — not the single-shot `execute_code` of
the retool sandbox:

| Method | What it does |
|---|---|
| `run_command(cmd, cwd=None, env=None, timeout=...)` | Shell exec via `bash -c`. Streams capped at 128 KB. |
| `read_file(path, max_bytes=...)` | `head -c MAX path`. Same byte cap. |
| `write_file(path, content, executable=False)` | Chunked stdin write via `tee`. 2 MB chunks. |
| `apply_patch(patch_text)` | `git apply --whitespace=nowarn` on the diff. |
| `run_tests(test_command=None)` | Default: `/eval.sh`. Override per-instance if needed. 1200s timeout. |
| `heartbeat()` | `bash -c true` to confirm the sandbox is alive. |
| `cleanup()` | Idempotent termination. |

Defensive features the retool sandbox doesn't have:

- **Byte-capped stream reads** (default 128 KB) — model can't OOM the rollout
  worker by writing megabytes to stdout.
- **Concurrency-bounded + rate-limited spawn** — Modal's control plane gets
  unhappy at >20 concurrent sandbox creates; the pool caps live sandboxes
  (default 32) and emits spawns at ≤4 QPS by default.
- **Lazy `App.lookup`** — Modal auth isn't forced at import time. Test
  files can be imported without Modal credentials.
- **Heartbeat** — proactively detect dead sandboxes rather than discovering
  at the next command.
- **`atexit` drain** — best-effort termination of any leaked sandboxes when
  the rollout process exits.

Comparison with our retool sandbox:

| aspect | retool `modal_tool_sandbox` | swe-env `coding_sandbox` |
|---|---|---|
| lifetime | reused across many python execs | one per agent rollout, then dies |
| image | `debian_slim` + `sympy/scipy` | per-instance `epochai/sweb.eval.*` |
| API | `execute_code(python_src)` | `run_command`, `read/write_file`, `apply_patch`, `run_tests` |
| network | `block_network=True` | unblocked (needs `pip`, `git`, etc.) |
| state retention | none — each call independent | filesystem persists across turns |
| byte cap on stdout | no | 128 KB (configurable) |
| heartbeat | no | yes |
| rate-limited spawn | no (eager gather) | yes (configurable QPS) |

## The agent loop (`generate_with_codingagent.py`)

Implements slime's `--custom-generate-function-path` and `--custom-rm-path`
contracts. The agent uses Qwen3-Thinking's native chat template via
`tokenizer.apply_chat_template(..., tools=TOOL_SPECS, add_generation_prompt=True)`,
so all tool injection and `<|im_start|>`/`<|im_end|>` framing is handled by
the model's own jinja template — no string fiddling.

Tools exposed to the model (Qwen3 native JSON `<tool_call>` grammar):

| name | parameters | semantics |
|---|---|---|
| `run_command` | `cmd`, optional `cwd` | shell exec |
| `read_file` | `path` | byte-capped read |
| `write_file` | `path`, `content` | write (creates parents) |
| `apply_patch` | `patch` | `git apply` of a unified diff |
| `run_tests` | (none) | run `/eval.sh` (the instance's test set) |
| `submit` | (none) | terminate the trajectory |

Loop ends on `submit`, on length-truncation, or on hitting `MAX_TURNS`
(default 30). Observations are wrapped in the Qwen3 tool-response shape:

```
<|im_end|>
<|im_start|>user
<tool_response>
... tool output (clipped to ~8 KB) ...
</tool_response><|im_end|>
<|im_start|>assistant
<think>
```

After the rollout, the agent's final `run_tests` call's exit code is stashed
on the sample. `reward_func` reads that verdict — no fresh sandbox needed,
which avoids a second image-pull per sample.

```python
score = 1.0 if tests_passed else 0.0
# Tiny shaping: 0.05 if the agent called submit but tests didn't pass,
# so the model learns to terminate cleanly rather than burning the turn budget.
```

## Comparison vs. retool

| aspect | retool (math) | swe-env (this) |
|---|---|---|
| sample shape | `prompt` (problem text) + `label` (answer string) | `prompt` (issue body) + `label` (instance_id) |
| sandbox lifetime | warm pool, many python execs each | one per rollout, dies after |
| tools | `code_interpreter(code)` | 6-tool set above |
| sandbox network | blocked | unblocked |
| reward | regex parse + math equivalence | `/eval.sh` exit code in the rollout sandbox |
| typical rollout wallclock | 5–8 min (math CoT + a few execs) | 5–10 min (agent + observations + tests) |

## Prereqs

1. **Modal account + credentials.** Save a Modal config TOML at
   `<slime_root>/.modal.toml`, or export `MODAL_CONFIG_PATH` to a custom
   path. The SDK reads either.
2. **Hugging Face token** (for downloading `Qwen3-235B-A22B-Thinking-2507-FP8`):
   put it at `<slime_root>/.hf_token.txt` or export `HF_TOKEN`.
3. **Qwen3-235B-A22B-Thinking-2507-FP8 weights + torch_dist conversion**
   under `<slime_root>/mnt/checkpoints/` — see slime's standard model setup.
4. **slime container image** that bundles SGLang 0.5.9 + Megatron-LM + Ray.
   `slimerl/slime:nightly-dev-20260425a` is what these scripts pin.

The smoke + unit tests don't need GPUs — only Modal credentials and ~30s
to pull one Epoch AI image.

## Quick start

### 1. Validate the sandbox library (no GPUs)

```bash
# from slime root
uv venv .venv-swe-env --python 3.11
uv pip install --python .venv-swe-env/bin/python "modal>=0.65"

MODAL_CONFIG_PATH=$PWD/.modal.toml \
  .venv-swe-env/bin/python examples/qwen3-235b_fullasync_swe-env/smoke_test_sandbox.py
# expect: 10 passed, 0 failed.
```

### 2. Validate the agent loop (no SGLang, real Modal sandbox)

```bash
MODAL_CONFIG_PATH=$PWD/.modal.toml \
  .venv-swe-env/bin/python examples/qwen3-235b_fullasync_swe-env/test_codingagent.py
# expect: 24 passed, 0 failed.
```

### 3. Build a prompt-data slice

```bash
.venv-swe-env/bin/python examples/qwen3-235b_fullasync_swe-env/gen_prompt_data.py
# writes mnt/data/swebench_verified/sample.jsonl with 10 instances
```

### 4. Launch a training run on 12 H200 nodes (4 actor + 8 rollout)

```bash
sbatch examples/qwen3-235b_fullasync_swe-env/run_swe.sbatch
```

Outputs land in `mnt/logs/infx-swe-async-<job_id>.out`. Watch for:

- `server is fired up and ready to roll!` × 8 — all SGLang engines up (~10 min)
- `First rollout sample: ...` — first dispatch landed
- `Finish rollout: ...` — first trajectory finished
- `data.py: rollout N: {...}` — trainer-side batch metrics
- `train_metric_utils.py: perf N: {...}` — trainer-side perf stats

## Configuration

### Env vars read by `coding_sandbox.py`

| var | default | meaning |
|---|---|---|
| `MODAL_CONFIG_PATH` | `~/.modal.toml` (Modal SDK default) | Modal credentials |
| `SLIME_SWEBENCH_APP` | `infx-slime-swebench-sandbox` | Modal app name |
| `SLIME_SWEBENCH_MAX_CONCURRENT` | `32` | Cap on live sandboxes |
| `SLIME_SWEBENCH_SPAWN_QPS` | `4` | Sandbox creates per second |
| `SLIME_SWEBENCH_TIMEOUT` | `1800` | Sandbox wallclock (seconds) |
| `SLIME_SWEBENCH_PER_CMD_TIMEOUT` | `120` | Per-command timeout |
| `SLIME_SWEBENCH_MAX_STREAM_BYTES` | `131072` | Stdout/stderr cap per command |
| `SLIME_SWEBENCH_IMAGE_REGISTRY` | `ghcr.io/epoch-research` | Image registry prefix |
| `SLIME_SWEBENCH_WORKDIR` | `/testbed` | Sandbox working dir |

### Env vars read by `generate_with_codingagent.py`

| var | default | meaning |
|---|---|---|
| `SLIME_SWEBENCH_MAX_TURNS` | `30` | Hard cap on agent turns per rollout |
| `SLIME_SWEBENCH_MAX_TOOL_CALLS` | `30` | Equivalent cap on tool dispatches |
| `SLIME_SWEBENCH_MAX_OBS_CHARS` | `8000` | Per-turn observation char cap |

### Key slime CLI flags in `run_swe.sh`

```bash
--custom-generate-function-path generate_with_codingagent.generate
--custom-rm-path                generate_with_codingagent.reward_func
--prompt-data    mnt/data/swebench_verified/sample.jsonl
--input-key      prompt
--label-key      label
--rollout-max-response-len 32768
--advantage-estimator gspo
--eps-clip 4e-4
```

## Results (baseline)

Untrained `Qwen3-235B-A22B-Thinking-2507-FP8` on a 10-instance Verified
slice, 20 rollouts × 4 samples per prompt = 80 trajectories per batch.
Effective context cap = 16 K tokens (CP × max_tokens_per_gpu = 2 × 8 K).

| metric | value |
|---|---|
| raw_reward (mean) | 0.017 |
| truncated_ratio (mean) | 0.66 |
| log_probs_time (steady-state) | 6.8 s |
| actor_train_time | 53.9 s |
| update_weights_time | 33.8 s |
| log_probs_tflops | 88.9 |
| actor_train_tflops | 32.9 |
| rollout_time | 343 s |

~1.5% zero-shot solve rate matches published agentic baselines for an
untrained 235B model. **66% truncation indicates the 16K cap is the
throttle** — see the context-scaling section below.

## Extension points

1. **Context cap.** A one-flag bump (`--max-tokens-per-gpu 8192 → 16384`)
   gives 32K effective context; in our runs that drove truncation from
   66% → 4.7% and raw_reward from 0.017 → 0.047. The trainer is also
   ~19% faster per step. Going beyond 32K (via PP=4→2 + CP=2→4) works
   but the model doesn't naturally produce trajectories that long, so the
   extra capacity goes unused.
2. **Partial-credit reward.** Currently `/eval.sh` exit code is binary.
   Parsing the test runner to count `FAIL_TO_PASS` tests passed would
   give denser signal.
3. **Sandbox network policy.** `block_network=False` by default — a model
   could in principle curl out for test answers. Curated allow-list
   (PyPI, GitHub) is a sensible hardening step.
4. **Image cold-start.** First pull of an instance image is ~30s. Modal
   caches after, but heterogeneous batches eat startup latency. Worth
   pre-pulling the top-N most common instances at process start.
5. **Patch staging.** `submit` is currently a special tool. Alternative:
   take the `git diff` at end of rollout as the implicit patch.
