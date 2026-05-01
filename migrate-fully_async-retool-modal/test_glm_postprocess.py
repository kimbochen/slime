#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["modal>=0.65"]
# ///
"""
Unit tests for the GLM-native parser/observation logic in generate_with_retool.py.
Does NOT call SGLang or Modal; just exercises the pure-Python pieces.

Run from repo root:
  ./examples/glm45-air_full-async_retool-modal/test_glm_postprocess.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Make the example dir importable.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Modal SDK reads creds from this path on this host.
os.environ.setdefault(
    "MODAL_CONFIG_PATH", "/home/sa-shared/kimbo/slime/.modal.toml"
)

# Stub out the heavy slime imports so generate_with_retool can be imported
# without spinning up Megatron/sglang. We only need the postprocess fns.
import types as _types
_fake_slime = _types.ModuleType("slime")
_fake_rollout = _types.ModuleType("slime.rollout")
_fake_sglang = _types.ModuleType("slime.rollout.sglang_rollout")
_fake_sglang.GenerateState = object
_fake_utils = _types.ModuleType("slime.utils")
_fake_http = _types.ModuleType("slime.utils.http_utils")
_fake_http.post = None
_fake_types_mod = _types.ModuleType("slime.utils.types")
class _FakeSample:
    class Status:
        TRUNCATED = "TRUNCATED"
        ABORTED = "ABORTED"
        COMPLETED = "COMPLETED"
_fake_types_mod.Sample = _FakeSample
_fake_rmhub = _types.ModuleType("slime.rollout.rm_hub")
_fake_mathdapo = _types.ModuleType("slime.rollout.rm_hub.math_dapo_utils")
_fake_mathdapo.compute_score = lambda *a, **kw: {"score": 0, "pred": ""}
sys.modules.update({
    "slime": _fake_slime,
    "slime.rollout": _fake_rollout,
    "slime.rollout.sglang_rollout": _fake_sglang,
    "slime.utils": _fake_utils,
    "slime.utils.http_utils": _fake_http,
    "slime.utils.types": _fake_types_mod,
    "slime.rollout.rm_hub": _fake_rmhub,
    "slime.rollout.rm_hub.math_dapo_utils": _fake_mathdapo,
})

import generate_with_retool as gwr  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"

results: list[tuple[str, str, str]] = []


def check(name: str, got, expected, *, transform=lambda x: x):
    g = transform(got)
    e = transform(expected)
    if g == e:
        results.append((PASS, name, ""))
    else:
        results.append((FAIL, name, f"\n   expected: {e!r}\n   got:      {g!r}"))


# ---- postprocess_predictions ----

def t_simple_native_call():
    out = (
        "Some chain of thought.\n"
        "<tool_call>code_interpreter\n"
        "<arg_key>code</arg_key>\n"
        "<arg_value>print(2 + 2)</arg_value>\n"
        "</tool_call>"
    )
    check("simple native tool call", gwr.postprocess_predictions(out),
          ("code", "print(2 + 2)"))


def t_multiline_code():
    code = "x = 1\nfor i in range(3):\n    x *= 2\nprint(x)"
    out = (
        f"<tool_call>code_interpreter\n"
        f"<arg_key>code</arg_key>\n"
        f"<arg_value>{code}</arg_value>\n"
        f"</tool_call>"
    )
    check("multiline code in arg_value",
          gwr.postprocess_predictions(out), ("code", code))


def t_extra_whitespace_around_name():
    out = (
        "<tool_call>  code_interpreter  \n"
        "<arg_key>code</arg_key>\n"
        "<arg_value>print('hi')</arg_value>\n"
        "</tool_call>"
    )
    check("whitespace around name", gwr.postprocess_predictions(out),
          ("code", "print('hi')"))


def t_final_answer():
    out = "Reasoning blah blah\n\nAnswer: \\boxed{42}"
    check("final answer plain", gwr.postprocess_predictions(out),
          ("answer", "42"))


def t_final_answer_nested_braces():
    out = "Answer: \\boxed{\\frac{1}{2}}"
    check("final answer with nested braces",
          gwr.postprocess_predictions(out), ("answer", "\\frac{1}{2}"))


def t_no_match_rambling():
    out = "I think the answer might be 41 but let me reconsider..."
    check("rambling, no tool / no answer",
          gwr.postprocess_predictions(out), (None, ""))


def t_unclosed_tool_call_does_not_match():
    out = "<tool_call>code_interpreter\n<arg_key>code</arg_key>\n<arg_value>print('x')</arg_value>"
    check("unclosed tool_call returns None",
          gwr.postprocess_predictions(out), (None, ""))


def t_first_tool_call_wins():
    out = (
        "<tool_call>code_interpreter\n"
        "<arg_key>code</arg_key>\n"
        "<arg_value>print('first')</arg_value>\n"
        "</tool_call>\n"
        "<tool_call>code_interpreter\n"
        "<arg_key>code</arg_key>\n"
        "<arg_value>print('second')</arg_value>\n"
        "</tool_call>"
    )
    check("first complete tool call wins",
          gwr.postprocess_predictions(out), ("code", "print('first')"))


def t_unknown_tool_name_no_match():
    out = (
        "<tool_call>shell\n"
        "<arg_key>cmd</arg_key>\n"
        "<arg_value>ls</arg_value>\n"
        "</tool_call>"
    )
    check("unknown tool name → None",
          gwr.postprocess_predictions(out), (None, ""))


# ---- postprocess_responses ----

def t_truncate_at_tool_call():
    raw = (
        "blah\n"
        "<tool_call>code_interpreter\n"
        "<arg_key>code</arg_key>\n"
        "<arg_value>print(1)</arg_value>\n"
        "</tool_call>"
        "...trailing garbage from sglang stop overshoot..."
    )
    expected = (
        "blah\n"
        "<tool_call>code_interpreter\n"
        "<arg_key>code</arg_key>\n"
        "<arg_value>print(1)</arg_value>\n"
        "</tool_call>"
    )
    check("postprocess_responses truncates at </tool_call>",
          gwr.postprocess_responses(raw), expected)


def t_truncate_at_answer():
    raw = "...lots of stuff... Answer: \\boxed{17}\nthen more"
    expected = "...lots of stuff... Answer: \\boxed{17}"
    check("postprocess_responses truncates at \\boxed{...}",
          gwr.postprocess_responses(raw), expected)


def t_passthrough_when_nothing():
    raw = "no tags here"
    check("passthrough", gwr.postprocess_responses(raw), raw)


# ---- execute_predictions (mocks the sandbox) ----

class _FakeSandboxResult:
    def __init__(self, code: str):
        self.code = code

    async def execute_tool(self, name, args):
        # Pretend we ran it; echo back a deterministic result.
        return f"OK: ran {len(args.get('code', ''))} chars of code"


async def _run_exec_tests():
    # Inject a fake tool_registry into the imported module so we don't hit
    # Modal during the execute_predictions tests.
    real = gwr.tool_registry
    gwr.tool_registry = _FakeSandboxResult("noop")  # type: ignore
    try:
        out = (
            "<tool_call>code_interpreter\n"
            "<arg_key>code</arg_key>\n"
            "<arg_value>print('hello')</arg_value>\n"
            "</tool_call>"
        )
        next_obs, done = await gwr.execute_predictions(out)
        check("execute_predictions(code) returns done=False",
              done, False)
        check("execute_predictions(code) emits <|observation|>",
              "<|observation|>" in next_obs, True)
        check("execute_predictions(code) emits <tool_response>",
              "<tool_response>" in next_obs, True)
        check("execute_predictions(code) reopens assistant turn",
              next_obs.endswith("<|assistant|>\n<think></think>\n"), True)
        check("execute_predictions(code) embeds the sandbox output",
              "OK: ran" in next_obs, True)

        next_obs, done = await gwr.execute_predictions("Answer: \\boxed{99}")
        check("execute_predictions(answer) returns done=True",
              (next_obs, done), ("", True))

        next_obs, done = await gwr.execute_predictions("just rambling, no tags")
        check("execute_predictions(none) returns done=False",
              done, False)
        check("execute_predictions(none) emits a parse-failure hint",
              "I could not parse" in next_obs, True)
        check("execute_predictions(none) hints native format",
              "<arg_key>code</arg_key>" in next_obs, True)
    finally:
        gwr.tool_registry = real


def main():
    t_simple_native_call()
    t_multiline_code()
    t_extra_whitespace_around_name()
    t_final_answer()
    t_final_answer_nested_braces()
    t_no_match_rambling()
    t_unclosed_tool_call_does_not_match()
    t_first_tool_call_wins()
    t_unknown_tool_name_no_match()

    t_truncate_at_tool_call()
    t_truncate_at_answer()
    t_passthrough_when_nothing()

    asyncio.run(_run_exec_tests())

    n_pass = sum(1 for s, _, _ in results if s == PASS)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    for status, name, info in results:
        marker = "✓" if status == PASS else "✗"
        print(f"  {marker} {name}{info}")
    print(f"\n{n_pass} passed, {n_fail} failed.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
