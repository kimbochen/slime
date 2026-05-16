"""
generate_with_codingagent.py
============================

Custom slime rollout generate + reward functions for SWE-bench-Verified-style
agentic coding, using Qwen3-235B-Thinking native tool-call grammar and the
per-instance Modal sandboxes from `coding_sandbox.py`.

Wire into slime via:
  --custom-generate-function-path generate_with_codingagent.generate
  --custom-rm-path                generate_with_codingagent.reward_func

Shape mirrors the retool example's `generate`:
  • Build initial prompt via tokenizer.apply_chat_template(..., tools=[...]).
  • For each turn: send to SGLang /generate, parse the model's first
    <tool_call>{...}</tool_call>, dispatch to the sandbox, wrap the result
    as a <tool_response> observation, append, repeat.
  • Loop ends on submit / max_turns / length truncation.

Key differences from retool:
  • One Modal sandbox per sample (per SWE-bench instance), not per python exec.
  • Six tools (run_command/read_file/write_file/apply_patch/run_tests/submit)
    instead of one code_interpreter.
  • Reward runs the per-instance test suite inside the same sandbox.

Sample input contract (loaded from `--prompt-data` jsonl):
  {
    "prompt": "<issue body / problem statement>",
    "label":  "<instance_id>"
  }
The instance_id determines which Epoch AI image is pulled.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

import uuid

from coding_sandbox import CodingSandbox, CodingSandboxPool, get_pool

# ---------------------------------------------------------------------------
# Tool registry: spec + dispatcher
# ---------------------------------------------------------------------------

TOOL_CONFIGS = {
    "max_turns": int(os.environ.get("SLIME_SWEBENCH_MAX_TURNS", "30")),
    "max_tool_calls": int(os.environ.get("SLIME_SWEBENCH_MAX_TOOL_CALLS", "30")),
    # Per-turn output character cap when serialising tool output back to the
    # model. We don't trust the model to handle 128 KB observations gracefully.
    "max_obs_chars": int(os.environ.get("SLIME_SWEBENCH_MAX_OBS_CHARS", "8000")),
}


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command inside the SWE-bench sandbox. "
                           "Sandbox starts at /testbed (the cloned repo). "
                           "stdout and stderr are returned together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to run"},
                    "cwd": {"type": "string", "description": "Optional working directory (defaults to /testbed)"},
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the sandbox filesystem. Output is byte-capped.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or /testbed-relative path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating parents as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a unified diff to the repo at /testbed via `git apply`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {"type": "string", "description": "Unified diff text"},
                },
                "required": ["patch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the SWE-bench test suite (FAIL_TO_PASS + PASS_TO_PASS) "
                           "for this instance via the image's /eval.sh. Returns exit code "
                           "and clipped output. Use this to verify your fix.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit your fix as final. Ends the trajectory.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---------------------------------------------------------------------------
# Qwen JSON tool-call parser (mirrors retool/generate_with_retool.py)
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_call(prediction: str) -> tuple[str | None, dict]:
    """Extract the first complete <tool_call>{...}</tool_call> JSON block.

    Returns (tool_name, arguments_dict). Returns (None, {}) if the model
    didn't emit a valid tool call.
    """
    m = _TOOL_CALL_RE.search(prediction)
    if not m:
        return None, {}
    raw = m.group(1)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        try:
            obj = json.loads(raw.replace("\n", "\\n"))
        except json.JSONDecodeError:
            return None, {}
    name = obj.get("name")
    args = obj.get("arguments", {}) or {}
    if not isinstance(args, dict):
        return None, {}
    return name, args


def postprocess_response(resp: str) -> str:
    """Truncate the model's output at the end of its last complete tool call
    so the observation we append doesn't follow a half-emitted tag."""
    if "</tool_call>" in resp:
        idx = resp.rfind("</tool_call>")
        return resp[: idx + len("</tool_call>")]
    return resp


def _clip(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    head = s[: limit // 2]
    tail = s[-limit // 2:]
    return f"{head}\n... [{len(s) - limit} chars elided] ...\n{tail}"


def _qwen_observation(content: str) -> str:
    """Wrap tool output in the Qwen3 tool-response observation shape and
    reopen an assistant <think> turn for the next step.

    Matches the format produced by Qwen3's chat_template for tool roles.
    """
    return (
        f"<|im_end|>\n<|im_start|>user\n"
        f"<tool_response>\n{content}\n</tool_response>"
        f"<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


async def execute_tool(
    sandbox: CodingSandbox,
    name: str,
    args: dict,
) -> tuple[str, bool]:
    """Dispatch a parsed tool call into the sandbox. Returns (observation, done).

    `done=True` only for `submit` — every other tool keeps the trajectory open.
    """
    max_chars = TOOL_CONFIGS["max_obs_chars"]

    if name == "run_command":
        cmd = args.get("cmd", "")
        cwd = args.get("cwd") or None
        if not cmd:
            return _qwen_observation("<error>missing 'cmd' argument</error>"), False
        res = await sandbox.run_command(cmd, cwd=cwd)
        return _qwen_observation(_clip(res.short_repr(max_lines=40), max_chars)), False

    if name == "read_file":
        path = args.get("path", "")
        if not path:
            return _qwen_observation("<error>missing 'path' argument</error>"), False
        content = await sandbox.read_file(path)
        return _qwen_observation(_clip(content, max_chars)), False

    if name == "write_file":
        path = args.get("path", "")
        content = args.get("content", "")
        if not path:
            return _qwen_observation("<error>missing 'path' argument</error>"), False
        res = await sandbox.write_file(path, content)
        return _qwen_observation(_clip(res.short_repr(max_lines=10), max_chars)), False

    if name == "apply_patch":
        patch = args.get("patch", "")
        if not patch:
            return _qwen_observation("<error>missing 'patch' argument</error>"), False
        res = await sandbox.apply_patch(patch)
        return _qwen_observation(_clip(res.short_repr(max_lines=40), max_chars)), False

    if name == "run_tests":
        res = await sandbox.run_tests()
        return _qwen_observation(_clip(res.short_repr(max_lines=80), max_chars)), False

    if name == "submit":
        return "", True

    return _qwen_observation(
        f"<error>unknown tool '{name}'. Available: "
        f"{', '.join(t['function']['name'] for t in TOOL_SPECS)}.</error>"
    ), False


# ---------------------------------------------------------------------------
# Slime entrypoint: per-sample generate
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are a software engineer solving a real GitHub issue inside a "
    "sandboxed development environment. The repository is checked out at "
    "/testbed. Use the provided tools to read code, modify files, and run "
    "the test suite to verify your fix. When you are confident the issue "
    "is fixed and tests pass, call the `submit` tool."
)


def _format_initial_prompt(tokenizer, prompt: str) -> list[int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tools=TOOL_SPECS, add_generation_prompt=True, tokenize=True,
    )


def _get_or_init_resume_key(sample: Sample, instance_id: str) -> str:
    """Resume key persisted in `sample.metadata` so slime's partial-rollout
    machinery can find the parked sandbox across abort+re-dispatch cycles.

    slime preserves `sample.metadata` (a dict) across the abort+resume
    flow, so anything we stash there comes back on the next generate()
    call for the same sample.
    """
    if sample.metadata is None:
        sample.metadata = {}
    key = sample.metadata.get("sandbox_resume_key")
    if not key:
        # 8-byte hex collision space is plenty for typical rollout batches.
        key = f"{instance_id}_{uuid.uuid4().hex[:12]}"
        sample.metadata["sandbox_resume_key"] = key
    return key


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Run one SWE-bench-Verified rollout: open (or resume) a sandbox,
    agent-loop until submit / max_turns / length truncation, return the
    full token trace.

    RESUMABLE: if slime aborts this sample mid-trajectory (e.g., a weight
    update fires during model generation), the sandbox is PARKED rather
    than terminated. The sample carries the resume key in metadata, the
    partial response in sample.tokens / sample.response / sample.loss_mask.
    On re-dispatch (with --partial-rollout enabled), the agent picks up
    the parked sandbox (filesystem state intact), restores its local
    accumulators from the sample, and continues the loop from the next
    turn under the new policy weights.

    Atomic-turn semantics: abort can only fire DURING the
    `await post(url, ...)` SGLang call (tool execution is local). So the
    state at abort-time is always:
      • Model output for THIS turn is partial (or empty if abort hit early)
      • All tool calls from PRIOR turns have already executed
      • Sandbox filesystem reflects all prior tool effects
    The partial model output for the in-progress turn IS preserved (slime
    returns the SGLang partial-text via output["text"]), so on resume the
    conversation continues with that text already present.
    """
    instance_id = sample.label if isinstance(sample.label, str) else None
    if not instance_id:
        # Hard failure: caller must supply instance_id via label.
        sample.status = Sample.Status.ABORTED
        sample.tokens = []
        sample.response_length = 0
        sample.response = ""
        sample.loss_mask = []
        return sample

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_tokens_ids = _format_initial_prompt(state.tokenizer, sample.prompt)
    prompt_text = state.tokenizer.decode(prompt_tokens_ids)

    # ---- Resume detection + state reconstruction ----
    # We're in "resume mode" if (a) --partial-rollout is on AND (b) the
    # sample already has response tokens from a prior aborted dispatch.
    is_resume = (
        getattr(args, "partial_rollout", False)
        and sample.response_length > 0
        and sample.tokens
        and sample.loss_mask
    )
    if is_resume:
        # Restore agent-local accumulators from the sample. The sample's
        # tokens are prompt + accumulated response; strip the prompt.
        response_token_ids = list(sample.tokens[len(prompt_tokens_ids):])
        response = sample.response or state.tokenizer.decode(response_token_ids)
        loss_masks = list(sample.loss_mask)
        # We can't perfectly reconstruct tool_call_count from text alone,
        # but the model's behavior doesn't depend on it; only the
        # max_tool_calls cap does. Counting <tool_call> tags in the
        # response is a decent approximation.
        tool_call_count = response.count("<tool_call>")
    else:
        response = ""
        response_token_ids = []
        loss_masks = []
        tool_call_count = 0
    submitted = False

    pool: CodingSandboxPool = get_pool()
    resume_key = _get_or_init_resume_key(sample, instance_id)
    sandbox, resumed = await pool.acquire_or_resume(
        instance_id, resume_key=resume_key
    )
    # Tracks whether we should park (mid-trajectory abort) or release
    # (normal completion / non-recoverable failure) on the way out.
    park_on_exit = False
    output: dict | None = None
    try:
        for turn in range(TOOL_CONFIGS["max_turns"]):
            # Per-turn context budget. SGLang rejects requests where
            # prompt + max_new_tokens exceeds context_length, so we have
            # to cap max_new_tokens dynamically by what's left.
            total_length = len(prompt_tokens_ids) + len(response_token_ids)
            if args.rollout_max_context_len is not None:
                max_context_length = args.rollout_max_context_len
            else:
                max_context_length = args.context_parallel_size * args.max_tokens_per_gpu

            SAFETY_MARGIN = 64        # special tokens / BOS slop
            MIN_VIABLE_COMPLETION = 256
            remaining = max_context_length - total_length - SAFETY_MARGIN
            if remaining < MIN_VIABLE_COMPLETION:
                sample.status = Sample.Status.TRUNCATED
                break

            per_turn_sp = dict(sampling_params)
            per_turn_sp["max_new_tokens"] = min(
                sampling_params.get("max_new_tokens", remaining),
                remaining,
            )

            payload = {
                "input_ids": prompt_tokens_ids + response_token_ids,
                "sampling_params": per_turn_sp,
                "return_logprob": True,
            }
            output = await post(url, payload)

            # ---- Abort handling: keep partial output, park sandbox ----
            finish_type = output["meta_info"]["finish_reason"]["type"]
            if finish_type == "abort":
                # Capture whatever partial tokens SGLang produced before
                # the abort. This is the "partial" in partial-rollout.
                if "output_token_logprobs" in output["meta_info"]:
                    cur_ids = [it[1] for it in output["meta_info"]["output_token_logprobs"]]
                    cur_text = state.tokenizer.decode(cur_ids)
                    cur_lp = [it[0] for it in output["meta_info"]["output_token_logprobs"]]
                    if sample.rollout_log_probs is None:
                        sample.rollout_log_probs = []
                    sample.rollout_log_probs += cur_lp
                else:
                    cur_text = postprocess_response(output.get("text", ""))
                    cur_ids = (
                        state.tokenizer(cur_text, add_special_tokens=False)["input_ids"]
                        if cur_text else []
                    )
                response += cur_text
                response_token_ids += cur_ids
                loss_masks += [1] * len(cur_ids)
                sample.status = Sample.Status.ABORTED
                # Park ONLY if --partial-rollout is on; otherwise old
                # behaviour (drop the sandbox on the way out) is correct.
                park_on_exit = bool(getattr(args, "partial_rollout", False))
                break

            if "output_token_logprobs" in output["meta_info"]:
                cur_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                cur_text = state.tokenizer.decode(cur_ids)
                cur_lp = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
                if sample.rollout_log_probs is None:
                    sample.rollout_log_probs = []
                sample.rollout_log_probs += cur_lp
            else:
                cur_text = postprocess_response(output["text"])
                cur_ids = state.tokenizer(cur_text, add_special_tokens=False)["input_ids"]

            response += cur_text
            response_token_ids += cur_ids
            loss_masks += [1] * len(cur_ids)

            # Length truncation ends the trajectory.
            if finish_type == "length":
                break

            name, arguments = parse_tool_call(cur_text)
            if name is None:
                # No tool call — model went off-script. Nudge it back.
                next_obs = _qwen_observation(
                    '<error>I could not parse a tool call. Emit one '
                    '<tool_call>{"name": "...", "arguments": {...}}</tool_call> '
                    'block. When done, call submit.</error>'
                )
                done = False
            else:
                next_obs, done = await execute_tool(sandbox, name, arguments)
                tool_call_count += 1

            if done:
                submitted = True
                break

            assert next_obs, "Observation must be non-empty for an open turn."
            obs_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
            response += next_obs
            response_token_ids += obs_ids
            loss_masks += [0] * len(obs_ids)

            if sample.rollout_log_probs is not None:
                sample.rollout_log_probs += [0.0] * len(obs_ids)
                assert len(response_token_ids) == len(sample.rollout_log_probs), (
                    f"Token/logp length mismatch at turn {turn}"
                )

            if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
                break

        # Evaluate inside the SAME sandbox only if we DIDN'T abort. On
        # abort we'll resume later; tests should run only at the actual
        # end of the trajectory.
        if not park_on_exit:
            sample._swebench_tests_passed = None
            try:
                test_res = await sandbox.run_tests()
                sample._swebench_tests_passed = test_res.ok
                sample._swebench_tests_summary = test_res.short_repr(max_lines=12)
            except Exception as e:
                sample._swebench_tests_summary = f"<run_tests crashed: {e}>"
    finally:
        if park_on_exit:
            # Abort mid-trajectory: keep the sandbox alive for resume.
            # The resume_key is already stashed on sample.metadata.
            await pool.park(sandbox, resume_key=resume_key)
        else:
            # Normal exit: terminate the sandbox and free its slot.
            await pool.release(sandbox)
            # Drop the resume_key from metadata so a future fresh dispatch
            # of the same Sample object doesn't try to resume a defunct
            # sandbox handle. (Harmless if absent on retry; defensive.)
            if sample.metadata is not None:
                sample.metadata.pop("sandbox_resume_key", None)

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks
    sample.payload_text = prompt_text + response
    sample.tool_call_count = tool_call_count
    sample.submitted = submitted

    if sample.status is None or sample.status == Sample.Status.COMPLETED:
        # output is the LAST SGLang response (may be None if we never
        # entered the loop because of an initial truncation check).
        finish_type = (
            output["meta_info"]["finish_reason"]["type"] if output else "stop"
        )
        match finish_type:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case _:
                sample.status = Sample.Status.COMPLETED

    return sample


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


async def reward_func(args, sample: Sample, **kwargs):
    """Read the tests-passed verdict that `generate` stashed on the sample."""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    passed = getattr(sample, "_swebench_tests_passed", None)
    summary = getattr(sample, "_swebench_tests_summary", "<no test summary>")

    if passed is True:
        score = 1.0
    elif passed is False:
        score = 0.0
    else:
        # Tests never ran (agent aborted before /eval.sh, or sandbox crash).
        score = 0.0

    # Light shaping: tiny bonus for actually calling submit, so the model
    # learns to terminate cleanly rather than burning the whole turn budget.
    if getattr(sample, "submitted", False) and score == 0.0:
        score = 0.05

    return {"score": float(score), "pred": summary[:512]}
