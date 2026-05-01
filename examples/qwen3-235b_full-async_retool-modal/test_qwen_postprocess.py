#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Offline unit tests for the Qwen3-Thinking-2507 native tool-call parser.

Run:  ./test_qwen_postprocess.py

These exercise `postprocess_predictions` and `postprocess_responses` against
realistic Qwen3 outputs:
  - <think>...</think>\\n\\n<tool_call>{json}</tool_call>
  - <think>...</think>\\n\\nAnswer: \\boxed{42}
  - unclosed <think> (still reasoning — no decision yet)
  - multi-line code in arguments.code (raw newlines, invalid JSON)
  - multiple tool calls in one response (parser picks last complete)
  - JSON with nested braces in answer
  - unknown tool name → no decision
  - trailing <|im_end|> stripped
  - arguments as JSON-string (some Qwen variants do this)
  - empty code field → no decision

We don't depend on slime / sglang / modal — pure-python parser only.
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

# Stub the slime/sglang/modal imports the module pulls in at the top.
import types

_stub_pkg = types.ModuleType("slime")
sys.modules["slime"] = _stub_pkg
sys.modules["slime.rollout"] = types.ModuleType("slime.rollout")
sys.modules["slime.rollout.sglang_rollout"] = types.ModuleType("slime.rollout.sglang_rollout")
sys.modules["slime.rollout.sglang_rollout"].GenerateState = object  # type: ignore[attr-defined]
sys.modules["slime.utils"] = types.ModuleType("slime.utils")
sys.modules["slime.utils.http_utils"] = types.ModuleType("slime.utils.http_utils")
sys.modules["slime.utils.http_utils"].post = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules["slime.utils.types"] = types.ModuleType("slime.utils.types")


class _SampleStub:
    class Status:
        TRUNCATED = "TRUNCATED"
        ABORTED = "ABORTED"
        COMPLETED = "COMPLETED"


sys.modules["slime.utils.types"].Sample = _SampleStub  # type: ignore[attr-defined]
sys.modules["slime.rollout.rm_hub"] = types.ModuleType("slime.rollout.rm_hub")
sys.modules["slime.rollout.rm_hub.math_dapo_utils"] = types.ModuleType("slime.rollout.rm_hub.math_dapo_utils")
sys.modules["slime.rollout.rm_hub.math_dapo_utils"].compute_score = lambda *a, **kw: {  # type: ignore[attr-defined]
    "score": 0.0, "acc": False, "pred": ""
}

# Stub Modal sandbox module entirely.
modal_stub = types.ModuleType("modal_tool_sandbox")
modal_stub.SEMAPHORE = None
modal_stub.TOOL_CONFIGS = {"max_turns": 16, "max_tool_calls": 16}
modal_stub.tool_registry = None
sys.modules["modal_tool_sandbox"] = modal_stub

import generate_with_retool as gwr  # noqa: E402


def case(label: str, ok: bool, detail: str = "") -> tuple[bool, str]:
    return ok, f"{'PASS' if ok else 'FAIL'} | {label}{(' | ' + detail) if detail else ''}"


def main() -> int:
    results: list[tuple[bool, str]] = []

    # 1) think + simple tool call (one-line code)
    s = '<think>\nLet me compute 2+2.\n</think>\n\n<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(2+2)"}}\n</tool_call><|im_end|>'
    a, c = gwr.postprocess_predictions(s)
    results.append(case("think + simple tool call", a == "code" and c == "print(2+2)", f"got ({a!r}, {c!r})"))

    # 2) think + final answer
    s = "<think>\nThe value is 41.\n</think>\n\nAnswer: \\boxed{41}<|im_end|>"
    a, c = gwr.postprocess_predictions(s)
    results.append(case("think + final answer", a == "answer" and c == "41", f"got ({a!r}, {c!r})"))

    # 3) unclosed <think> (still reasoning)
    s = "<think>\nstill thinking..."
    a, c = gwr.postprocess_predictions(s)
    results.append(case("unclosed <think> -> no decision", a is None and c == "", f"got ({a!r}, {c!r})"))

    # 4) multi-line code w/ raw newlines in JSON (invalid strict JSON)
    s = (
        "<think>compute the polynomial root</think>\n\n"
        "<tool_call>\n"
        '{"name": "code_interpreter", "arguments": {"code": "import numpy as np\nroots = np.roots([1, -3, 2])\nprint(roots)"}}\n'
        "</tool_call>"
    )
    a, c = gwr.postprocess_predictions(s)
    expected_code = "import numpy as np\nroots = np.roots([1, -3, 2])\nprint(roots)"
    results.append(case("multi-line code w/ raw newlines", a == "code" and c == expected_code, f"got ({a!r})"))

    # 5) multiple tool calls — first one should be picked
    s = (
        "<think>two-step plan</think>\n\n"
        '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "x = 5"}}\n</tool_call>'
        "\nMid text...\n"
        '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "y = 10"}}\n</tool_call>'
    )
    a, c = gwr.postprocess_predictions(s)
    results.append(case("multiple tool calls -> picks first", a == "code" and c == "x = 5", f"got ({a!r}, {c!r})"))

    # 6) Answer with nested braces (e.g. \frac{a}{b})
    s = "<think>fraction</think>\n\nAnswer: \\boxed{\\frac{1}{2}}"
    a, c = gwr.postprocess_predictions(s)
    results.append(case("answer with nested braces", a == "answer" and c == "\\frac{1}{2}", f"got ({a!r}, {c!r})"))

    # 7) Unknown tool name → no decision
    s = '<tool_call>\n{"name": "calculator", "arguments": {"expr": "2+2"}}\n</tool_call>'
    a, c = gwr.postprocess_predictions(s)
    results.append(case("unknown tool name", a is None and c == "", f"got ({a!r}, {c!r})"))

    # 8) Trailing <|im_end|> stripped
    s = '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(1)"}}\n</tool_call><|im_end|>'
    a, c = gwr.postprocess_predictions(s)
    results.append(case("trailing <|im_end|> stripped", a == "code" and c == "print(1)", f"got ({a!r}, {c!r})"))

    # 9) Arguments as a JSON-string (Qwen variant)
    s = '<tool_call>\n{"name": "code_interpreter", "arguments": "{\\"code\\": \\"print(99)\\"}"}\n</tool_call>'
    a, c = gwr.postprocess_predictions(s)
    results.append(case("arguments as JSON-string", a == "code" and c == "print(99)", f"got ({a!r}, {c!r})"))

    # 10) Empty code field → no decision
    s = '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": ""}}\n</tool_call>'
    a, c = gwr.postprocess_predictions(s)
    results.append(case("empty code field", a is None and c == "", f"got ({a!r}, {c!r})"))

    # 11) Tool call BEFORE answer in same response — answer wins
    s = (
        "<think>first compute, then answer</think>\n\n"
        '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(1)"}}\n</tool_call>\n'
        "Answer: \\boxed{42}"
    )
    a, c = gwr.postprocess_predictions(s)
    results.append(case("answer wins over earlier tool_call", a == "answer" and c == "42", f"got ({a!r}, {c!r})"))

    # 12) Final answer with leading whitespace + multi-digit answer
    s = "<think>x</think>\n\nAnswer:  \\boxed{12345}\n"
    a, c = gwr.postprocess_predictions(s)
    results.append(case("answer with extra whitespace", a == "answer" and c == "12345", f"got ({a!r}, {c!r})"))

    # 13) Invalid JSON inside tool_call (totally malformed) → no decision
    s = "<tool_call>\nthis is not json {at all}\n</tool_call>"
    a, c = gwr.postprocess_predictions(s)
    results.append(case("malformed JSON in tool_call", a is None and c == "", f"got ({a!r}, {c!r})"))

    # 14) Tool call with extra surrounding whitespace inside the body
    s = '<tool_call>\n\n   {"name": "code_interpreter", "arguments": {"code": "print(7)"}}   \n\n</tool_call>'
    a, c = gwr.postprocess_predictions(s)
    results.append(case("tool_call with extra whitespace", a == "code" and c == "print(7)", f"got ({a!r}, {c!r})"))

    # 15) Tool call SPANS multiple lines — newlines OUTSIDE strings (trivially valid JSON)
    s = (
        "<tool_call>\n"
        '{\n  "name": "code_interpreter",\n  "arguments": {\n    "code": "print(\'hi\')"\n  }\n}\n'
        "</tool_call>"
    )
    a, c = gwr.postprocess_predictions(s)
    results.append(case("tool_call multi-line JSON, newlines outside strings", a == "code" and c == "print('hi')", f"got ({a!r}, {c!r})"))

    # 16) postprocess_responses: truncate at last </tool_call>
    raw = "stuff <tool_call>...</tool_call> trailing garbage <tool_call>...broken"
    out = gwr.postprocess_responses(raw)
    results.append(case("trim at last </tool_call>", out.endswith("</tool_call>"), f"got {out!r}"))

    # 17) postprocess_responses: trim at last Answer/\boxed
    raw = "Answer: \\boxed{1} ... continues with garbage"
    out = gwr.postprocess_responses(raw)
    results.append(case("trim at last Answer:\\boxed{}", out == "Answer: \\boxed{1}", f"got {out!r}"))

    # 18) postprocess_responses: leaves intact when neither marker present
    raw = "<think>still going..."
    out = gwr.postprocess_responses(raw)
    results.append(case("trim no-op when no closer", out == raw, f"got {out!r}"))

    # 19) Two adjacent <think> blocks — both stripped, then parsed
    s = "<think>plan</think>\n<think>refine</think>\n\nAnswer: \\boxed{99}"
    a, c = gwr.postprocess_predictions(s)
    results.append(case("two <think> blocks stripped", a == "answer" and c == "99", f"got ({a!r}, {c!r})"))

    # 20) <think> contains a fake tool_call literal — should be IGNORED
    s = (
        '<think>I could call <tool_call>{"name": "code_interpreter", "arguments": {"code": "fake"}}</tool_call> but won\'t.</think>'
        "\n\nAnswer: \\boxed{7}"
    )
    a, c = gwr.postprocess_predictions(s)
    results.append(case("tool_call inside <think> ignored", a == "answer" and c == "7", f"got ({a!r}, {c!r})"))

    # 21) JSON with embedded backslashes (e.g., LaTeX in code)
    s = (
        '<tool_call>\n'
        '{"name": "code_interpreter", "arguments": {"code": "import sympy\\nprint(sympy.latex(sympy.pi))"}}\n'
        '</tool_call>'
    )
    a, c = gwr.postprocess_predictions(s)
    expected = "import sympy\nprint(sympy.latex(sympy.pi))"
    results.append(case("JSON with embedded \\n escape", a == "code" and c == expected, f"got ({a!r})"))

    passed = sum(1 for ok, _ in results if ok)
    failed = len(results) - passed
    for ok, msg in results:
        print(msg)
    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
