"""Model-side concerns for the SWE-bench rollout: tool schemas, parsing, and
observation formatting.

This module is pure — no I/O, no sandbox concept, no sibling imports. It
defines what the model sees (TOOL_SPECS) and helps translate between the
model's text output and structured tool calls (parse_action), plus the chat
markers used to splice tool observations back into the rollout token tape
(OBSERVATION_PREFIX/SUFFIX + format_observation).

Execution-side concerns — Sandbox interface, Modal-backed implementation,
tool dispatch — live in sandbox.py.
"""

from __future__ import annotations

import json
import re
from typing import Any


# ---------------------------------------------------------------------------
# Tool registry — OpenHands / Anthropic-style minimal toolset
# ---------------------------------------------------------------------------

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command in the sandbox. The sandbox starts at "
                "/testbed (the cloned repository). stdout and stderr are returned "
                "together. Long-running commands are subject to a wall-clock timeout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": (
                "Custom editing tool for viewing, creating, and editing files.\n"
                "* `view`: shows file contents with line numbers (or directory contents if path is a dir).\n"
                "* `create`: creates a new file with the given content. Fails if the file already exists.\n"
                "* `str_replace`: replaces a unique substring in a file. The `old_str` must match exactly "
                "(including whitespace) and appear exactly once.\n"
                "* `insert`: inserts `new_str` after line number `insert_line` in `path`.\n"
                "* `undo_edit`: reverts the last edit made to `path`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                        "description": "The editor operation to perform.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file (or directory, for view).",
                    },
                    "file_text": {
                        "type": "string",
                        "description": "Required for `create`. Full contents of the new file.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Required for `str_replace`. Exact substring to replace.",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement for `str_replace`, or inserted text for `insert`.",
                    },
                    "insert_line": {
                        "type": "integer",
                        "description": "Required for `insert`. Insert after this line number (0 = at top).",
                    },
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional [start, end] (1-indexed, inclusive) for `view`.",
                    },
                },
                "required": ["command", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Declare the task complete. Ends the trajectory.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# Per-observation byte cap so we don't hand the model 128 KB of test output.
MAX_OBS_CHARS = 8000


# ---------------------------------------------------------------------------
# Initial prompt rendering — system message + user issue + tool spec injection
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a software engineer solving a real GitHub issue inside a "
    "sandboxed development environment. The repository is checked out at "
    "/testbed. Use the provided tools to read code, modify files, and run "
    "the test suite to verify your fix. When you are confident the issue "
    "is fixed and tests pass, call the `finish` tool."
)


def format_initial_prompt(tokenizer, user_prompt: str) -> list[int]:
    """Render the chat template with TOOL_SPECS attached so the model sees
    the available tools as part of its initial context. The Qwen3 template
    consumes `tools=...` and emits a system-level <tools>...</tools> block
    plus the `<tool_call>{...}</tool_call>` invocation instructions; without
    those, the model has no API to call.

    Co-located with TOOL_SPECS so changing the tool list or system prompt
    is a one-file edit and the chat-template wiring is testable without
    pulling in slime / SGLang.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    out = tokenizer.apply_chat_template(
        messages, tools=TOOL_SPECS, add_generation_prompt=True, tokenize=True,
    )
    # Newer transformers returns a BatchEncoding (with .input_ids) when the
    # template emits generation indices; older returns a plain list[int].
    # Normalize so callers can always assume list[int] (and .extend() works).
    if hasattr(out, "input_ids"):
        out = out["input_ids"]
    return list(out)


# ---------------------------------------------------------------------------
# Parsing — extract one tool call from the model's response
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_action(text: str) -> tuple[str, dict] | None:
    """Extract the first complete <tool_call>{...}</tool_call> JSON block.

    Returns (name, args) on success, None if no parseable tool call was found.
    """
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    raw = m.group(1)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Forgiving second pass: literal newlines inside string values are
        # common in model output and break strict JSON.
        try:
            obj = json.loads(raw.replace("\n", "\\n"))
        except json.JSONDecodeError:
            return None
    name = obj.get("name")
    args = obj.get("arguments", {}) or {}
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    return name, args


# ---------------------------------------------------------------------------
# Observation formatting — hard-coded Qwen3-Thinking markers
# ---------------------------------------------------------------------------
#
# Slime's GenerateState sets no_stop_trim=True (sglang_rollout.py:105) and
# --rollout-skip-special-tokens defaults to False, so the model's response
# already ends with the <|im_end|> token (and its rendered text). We must NOT
# emit another one or we'd produce <|im_end|><|im_end|> back-to-back.
#
# Splice structure (what `format_observation` produces, appended after the
# model's response which already ends with <|im_end|>):
#
#     \n                        ← separator after model's <|im_end|>
#     <|im_start|>user\n        ← open user turn (tool roles render as user)
#     <tool_response>\n
#     {tool output}
#     \n</tool_response>
#     <|im_end|>\n              ← close user turn
#     <|im_start|>assistant\n   ← open next assistant turn
#     <think>\n                 ← Qwen3-Thinking reopens the thinking block
#
# For non-thinking Qwen3 variants, drop the trailing "<think>\n".
# For Llama / Mistral / GLM, the marker bytes are different — re-derive.
# tests/test_tools.py verifies these match apply_chat_template byte-for-byte.

OBSERVATION_PREFIX = "\n<|im_start|>user\n<tool_response>\n"
OBSERVATION_SUFFIX = "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"


def format_observation(content: str) -> str:
    return OBSERVATION_PREFIX + content + OBSERVATION_SUFFIX


def clip(s: str, limit: int = MAX_OBS_CHARS) -> str:
    """Head/tail elision so observations never exceed `limit` characters."""
    if len(s) <= limit:
        return s
    head = s[: limit // 2]
    tail = s[-limit // 2 :]
    return f"{head}\n... [{len(s) - limit} chars elided] ...\n{tail}"
