# Qwen3-235B SWE-bench-Verified — resumable variant

Sibling of `examples/qwen3-235b_fullasync_swe-env/`. Same model, dataset,
trainer config, and overall structure — but the **rollout agent supports
slime's `--partial-rollout`** pipeline-RL flag.

In the base example, the agent has a hard
`assert not args.partial_rollout, "Partial rollout not supported."`
because each rollout is wrapped in `async with pool.session(instance_id)`
— if the trainer pushes new weights mid-trajectory and SGLang aborts
the in-flight generation, the agent returns immediately, the `async with`
tears down the Modal sandbox, and any filesystem state (files written,
commands run, packages installed) is lost. Re-dispatching the same sample
under partial-rollout would feed the model a conversation about effects
that no longer exist in the new sandbox.

This variant fixes both pieces.

## What's different

### `coding_sandbox.py` — pool gains a parking lot

Two new methods on `CodingSandboxPool`:

- `await pool.acquire_or_resume(instance_id, resume_key=key)` →
  `(sandbox, resumed)`. If a sandbox was previously parked under `key`,
  return it; otherwise create a fresh one. Same concurrency / spawn-rate
  guarantees as `acquire`.
- `await pool.park(sandbox, resume_key=key)` — move sandbox from active
  set to parked set, keyed by the per-sample resume key. **Concurrency
  slot is NOT released**: the sandbox is alive on Modal's side, holding
  state, awaiting resume.

Parked sandboxes still count against `SLIME_SWEBENCH_MAX_CONCURRENT`.
They get torn down by the `atexit` hook on process shutdown.

### `generate_with_codingagent.py` — resume-aware agent

- Drops the `assert not args.partial_rollout`.
- Generates a stable `sandbox_resume_key` on first call, stashes it on
  `sample.metadata` (which slime preserves across abort + re-dispatch).
- Calls `pool.acquire_or_resume(...)` instead of `pool.session(...)`.
- On entry, if `args.partial_rollout` is on and `sample.response_length >
  0`, reconstructs `response_token_ids`, `response`, `loss_masks`, and
  `tool_call_count` from `sample.*` (slime persists these across abort).
- On abort mid-trajectory: captures the partial output from SGLang into
  the same accumulators, sets `Sample.Status.ABORTED`, and **parks** the
  sandbox under the resume key. Tests are NOT run yet (will run on the
  final resume that hits submit / max_turns / length).
- On normal completion: releases the sandbox and clears the resume key
  from metadata.

### `run_swe.sh` — flag on

- `--partial-rollout` is enabled (was disabled with a comment in the base
  example explaining the assertion).
- `SLIME_SWEBENCH_TIMEOUT` bumped from 1800s to 7200s so parked sandboxes
  can survive multiple weight-update cycles (assuming ~5-min trainer
  steps and several abort/resume rounds per trajectory).

### `run_swe.sbatch`

- Job name changed to `infx-swe-rsm` (resumable) so it doesn't collide
  with concurrent base-variant runs in the queue.

## What's still the same

- All other model / parallelism / SGLang / GRPO args.
- `add_slime_metrics.py` (the metric wrapper). Same `policy_staleness`
  reporting — but note: with partial-rollout on, the stamp at
  `add_slime_metrics.py:157` will be **overwritten on each resume**, so
  reported staleness reflects the LAST resume, not the original first
  dispatch. Consider making the stamp sticky-on-first-dispatch if you
  want true max-staleness telemetry.

## Atomic-turn semantics

Aborts can only fire during the `await post(url, ...)` SGLang generation
call. Tool execution (`execute_tool(sandbox, ...)`) is local async work
that slime/SGLang can't interrupt. So at any abort point:

- The model output for the in-progress turn is partial (the SGLang
  response carries whatever tokens were generated before the abort, plus
  `finish_reason.type == "abort"`).
- All tool calls from earlier turns have already executed.
- Sandbox filesystem reflects all tool effects up through the last
  completed turn.

On resume, we pick up with whatever partial model output we got,
continue parsing it (it may or may not contain a complete `<tool_call>`
block), and proceed. SGLang re-prefills the conversation up through the
partial response under the new weights; the GSPO importance ratio
corrects for the on-policy/off-policy delta on the older tokens.

## Untested

This variant has been **implemented but not yet run end-to-end**. Things
that may need adjustment after a first launch:

- Resume key collision: if slime somehow re-dispatches the same Sample
  object as a different trajectory (not the same trajectory), the
  resume_key persistence could attach the wrong sandbox. Unlikely with
  slime's current sample lifecycle but worth verifying.
- Loss-mask reconstruction: we count `<tool_call>` substrings to
  approximate `tool_call_count`. Edge cases (model emits the literal
  string inside a comment, etc.) could miscount. Doesn't affect
  correctness, just the `max_tool_calls` cutoff timing on resume.
- Parked-sandbox lifetime: with `SLIME_SWEBENCH_TIMEOUT=7200`, sandboxes
  can survive up to 2 hours. If trainer steps stretch beyond that and a
  parked sandbox times out, the next resume will fail with a stale
  handle. The pool's atexit hook will clean these up; the agent will
  spawn a fresh sandbox (but lose all prior tool effects, the same
  failure mode the base example has). Bumping the env var higher is
  cheap.

## How to launch

```bash
sbatch examples/qwen3-235b_fullasync_swe-env_resumable/run_swe.sbatch
```

12 nodes, same as the base variant. Job name in slurm: `infx-swe-rsm`.
