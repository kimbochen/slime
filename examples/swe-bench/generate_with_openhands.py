"""Barebones custom generate function for SWE-bench-style multi-turn rollout.

Designed for fully-async training with --partial_rollout. Module layout:
  * `tool_specs.py` — TOOL_SPECS, parse_action, format_observation, markers.
  * `tool_runtime.py` — run_tool dispatcher + per-tool implementations.
  * `sandbox.py` — Sandbox ABC + Modal-backed SWEBenchSandbox.
  * `generate.py` (this file) — rollout state machine glue.

Launch requirements:
    - PYTHONPATH must include this directory so `import tool_specs`,
      `import tool_runtime`, and `import sandbox` resolve (slime's
      `load_function` uses importlib; the "swe-bench" path contains a hyphen
      so dotted package imports don't work — use simple module names and put
      the dir on PYTHONPATH at launch).
    - --rollout-max-context-len MUST be set explicitly. This script does not
      fall back to a derived value; an unset flag raises at function entry.
    - sample.metadata["instance_id"] must be set by the data loader.

Wire it in with:

    --rollout-function-path     fully_async_rollout.generate_rollout_fully_async
    --custom-generate-function-path generate_with_openhands.generate
    --partial_rollout
    --mask-offpolicy-in-partial-rollout
    --rollout-max-context-len   <int>     # REQUIRED

Note: do NOT pass --apply-chat-template or --tool-key. This module renders the
chat template internally via tokenizer.apply_chat_template(messages,
tools=TOOL_SPECS, ...) so the tool list lives with the generate code.
"""

import asyncio
import time
from argparse import Namespace

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

import tool_specs
from sandbox import (
    SWEBenchSandbox,
    SandboxCreateError,
    SandboxDiedError,
    SandboxReattachError,
)
from tool_runtime import run_tool


MAX_TURNS = 50           # total model invocations across all resumes
MAX_TOOL_CALLS = 30      # total tool invocations across all resumes


async def generate(args: Namespace, sample: Sample, sampling_params: dict) -> Sample:
    assert args.rollout_max_context_len is not None, (
        "swe-bench generate requires --rollout-max-context-len to be set explicitly"
    )

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sandbox = SWEBenchSandbox(
        sample.metadata["instance_id"],
        sandbox_id=sample.metadata.get("sandbox_id"),
    )

    # First call: render chat template + tool specs + system prompt and
    # tokenize into sample.tokens.
    # Resume call: sample.tokens already holds prompt + prior response prefix.
    if not sample.tokens:
        sample.tokens = tool_specs.format_initial_prompt(state.tokenizer, sample.prompt)
    if sample.loss_mask is None:
        sample.loss_mask = []
    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []

    # Lifetime budgets: survive resume via sample.metadata.
    turn_count = sample.metadata.get("turn_count", 0)
    tool_call_count = sample.metadata.get("tool_call_count", 0)
    # Bump resume_count every call. After call 1: 1. After resume call 2: 2. etc.
    # metrics.py reads this to track partial-rollout health.
    sample.metadata["resume_count"] = sample.metadata.get("resume_count", 0) + 1

    try:
        while turn_count < MAX_TURNS and tool_call_count < MAX_TOOL_CALLS:
            if state.aborted:
                sample.status = Sample.Status.ABORTED
                break

            if len(sample.tokens) >= args.rollout_max_context_len:
                sample.status = Sample.Status.TRUNCATED
                break

            try:
                payload = {
                    "input_ids": sample.tokens,
                    "sampling_params": sampling_params,
                    "return_logprob": True,
                }
                _inf_t0 = time.monotonic()
                output = await post(url, payload)
            except asyncio.CancelledError:
                sample.status = Sample.Status.ABORTED
                break
            finally:
                # Accumulate even on cancel — we still spent the time.
                sample.metadata["inference_time_s"] = (
                    sample.metadata.get("inference_time_s", 0.0)
                    + (time.monotonic() - _inf_t0)
                )

            meta = output["meta_info"]
            finish_type = meta["finish_reason"]["type"]
            if finish_type == "abort":
                sample.status = Sample.Status.ABORTED
                break

            # Successful model invocation → bump turn counter and persist.
            turn_count += 1
            sample.metadata["turn_count"] = turn_count

            # On-policy chunk: tokens + sampling logprobs + loss_mask=1.
            # SGLang returns (logprob, token_id, ...) tuples — sometimes 2 fields,
            # sometimes 3+ depending on version/flags. Index access instead of
            # unpacking is robust to either shape (matches reference pattern).
            new_tokens = [item[1] for item in meta["output_token_logprobs"]]
            sample.tokens.extend(new_tokens)

            new_logps = [item[0] for item in meta["output_token_logprobs"]]
            sample.rollout_log_probs.extend(new_logps)

            sample.loss_mask.extend([1] * len(new_tokens))
            sample.response_length += len(new_tokens)

            sample.response += output["text"]
            if "weight_version" in meta:
                sample.weight_versions.append(meta["weight_version"])

            # SGLang hit sampling_params["max_new_tokens"] mid-stream — no
            # complete action to parse, so end the trajectory as TRUNCATED.
            if finish_type == "length":
                sample.status = Sample.Status.TRUNCATED
                break

            parsed = tool_specs.parse_action(output["text"])
            if parsed is None:
                # No tool call parsed. Model either emitted thinking-only or
                # malformed JSON. Splice a nudge observation (same shape as a
                # real tool response) so the next iteration has a fresh
                # assistant turn opener to generate from, and let the model retry.
                nudge = tool_specs.format_observation(
                    "<error>Your last message did not contain a parseable "
                    "<tool_call>{...}</tool_call> block. Available tools: "
                    "bash, str_replace_editor, finish. Try again.</error>"
                )
                nudge_ids = state.tokenizer.encode(nudge, add_special_tokens=False)
                sample.tokens.extend(nudge_ids)
                sample.rollout_log_probs.extend([0.0] * len(nudge_ids))
                sample.loss_mask.extend([0] * len(nudge_ids))
                sample.response_length += len(nudge_ids)
                sample.response += nudge
                continue

            name, action_args = parsed
            _sb_t0 = time.monotonic()
            try:
                observation, done = await run_tool(sandbox, name, action_args)
            finally:
                # Accumulate even if run_tool raises a sandbox error — we still
                # spent the time (and the outer except will catch it).
                sample.metadata["sandbox_time_s"] = (
                    sample.metadata.get("sandbox_time_s", 0.0)
                    + (time.monotonic() - _sb_t0)
                )

            # Tool invocation completed → bump counter and persist.
            tool_call_count += 1
            sample.metadata["tool_call_count"] = tool_call_count

            # First tool call also makes the Modal sandbox real; persist its id
            # so the next resume can reattach. Idempotent on subsequent calls,
            # and a None write here is cleaned by the finally block on terminal.
            sample.metadata["sandbox_id"] = sandbox.sandbox_id

            if done:
                # `finish` tool: model declared the trajectory finished.
                sample.status = Sample.Status.COMPLETED
                break

            # Observation: never sampled by the model. Pad logprobs, mask out of loss.
            obs_token_ids = state.tokenizer.encode(observation, add_special_tokens=False)
            sample.tokens.extend(obs_token_ids)
            sample.rollout_log_probs.extend([0.0] * len(obs_token_ids))
            sample.loss_mask.extend([0] * len(obs_token_ids))
            sample.response_length += len(obs_token_ids)
            sample.response += observation
        else:
            # While-condition false at top of iteration → budget exhausted.
            sample.status = Sample.Status.TRUNCATED
    except (SandboxCreateError, SandboxReattachError, SandboxDiedError) as e:
        # Sandbox lifecycle failure (couldn't spawn, parked container dead,
        # or container died mid-rollout). Fail the sample with a reason the
        # buffer/metrics layer can pick up via sample.metadata["fail_reason"].
        sample.status = Sample.Status.FAILED
        sample.metadata["fail_reason"] = str(e)
    finally:
        # Terminal status → tear down the Modal container and forget the id.
        # Non-terminal (ABORTED) → detach: leave the container alive in Modal
        # for the next resume to reattach via sample.metadata["sandbox_id"].
        if sample.status in (
            Sample.Status.COMPLETED,
            Sample.Status.TRUNCATED,
            Sample.Status.FAILED,
        ):
            # Capture the model's net edits as a unified diff BEFORE closing.
            # reward.py (fresh-sandbox eval) reads this from sample.metadata
            # to apply against a clean container. Best-effort: if the sandbox
            # is already gone (FAILED for sandbox reasons) or git fails, skip
            # and let reward see an absent model_patch → score 0.0.
            if sample.status != Sample.Status.FAILED:
                try:
                    diff, rc = await sandbox.exec("cd /testbed && git diff HEAD")
                    if rc == 0:
                        sample.metadata["model_patch"] = diff
                except Exception:
                    pass
            await sandbox.close()
            sample.metadata.pop("sandbox_id", None)
        else:
            await sandbox.detach()

    return sample
