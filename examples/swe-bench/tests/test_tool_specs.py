#!/usr/bin/env python3
"""Tests for examples/swe-bench/tool_specs.py.

The marker-vs-template test verifies that the hard-coded OBSERVATION_PREFIX /
OBSERVATION_SUFFIX in tool_specs.py produce byte-identical output to what
tokenizer.apply_chat_template would emit for the same conversation. Catches
regressions if the model's chat template changes or if slime's stop-token
handling changes.

Run from anywhere:
    python examples/swe-bench/tests/test_tool_specs.py

Tokenizer path defaults to the local Qwen3-235B-A22B-Thinking-2507-FP8
checkpoint; override with SWEBENCH_TEST_TOKENIZER=/path/to/dir.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../tests
EXAMPLE_DIR = HERE.parent                       # .../swe-bench
sys.path.insert(0, str(EXAMPLE_DIR))

import tool_specs  # noqa: E402


DEFAULT_TOKENIZER = (
    "/home/sa-shared/kimbo/slime/mnt/checkpoints/Qwen3-235B-A22B-Thinking-2507-FP8"
)
TOKENIZER_PATH = os.environ.get("SWEBENCH_TEST_TOKENIZER", DEFAULT_TOKENIZER)


def _load_tokenizer():
    if not os.path.exists(TOKENIZER_PATH):
        print(
            f"SKIP: tokenizer not found at {TOKENIZER_PATH}. "
            f"Set SWEBENCH_TEST_TOKENIZER to override."
        )
        sys.exit(77)  # POSIX skip convention
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)


def test_observation_markers_match_chat_template() -> None:
    """tool_specs.format_observation produces byte-identical output to
    apply_chat_template for an equivalent tool turn.

    Builds a synthetic 4-turn conversation two ways:
      A. Splice: initial-prompt-rendered + assistant + <|im_end|> + format_observation
         (The trailing "<|im_end|>" simulates what no_stop_trim=True leaves in
         sample.tokens after the model emits stop. The literal string here
         must NOT contain another <|im_end|> in the assistant content because
         the chat template auto-appends one after assistant content.)
      B. Template: apply_chat_template on the full [sys, user, asst, tool] list.

    Assumes slime defaults: no_stop_trim=True + --rollout-skip-special-tokens
    unset (False). If you flip --rollout-skip-special-tokens to True, drop the
    " + '<|im_end|>'" from the splice line and restore the leading
    "<|im_end|>\\n" on tool_specs.OBSERVATION_PREFIX, then re-run this test.
    """
    tokenizer = _load_tokenizer()

    system = "you are a swe agent"
    user = "fix the bug"
    # Qwen3-Thinking's chat template auto-opens the assistant turn with
    # "<think>\n" (via add_generation_prompt=True), so the model generates
    # *inside* an already-open thinking block — its output starts after the
    # opener, not with another "<think>\n".
    assistant = (
        'let me look\n</think>\n\n'
        '<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>'
    )
    tool_out = "file1.py\nfile2.py"

    initial = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tools=tool_specs.TOOL_SPECS, tokenize=False, add_generation_prompt=True,
    )
    spliced = initial + assistant + "<|im_end|>" + tool_specs.format_observation(tool_out)

    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
            {"role": "tool", "content": tool_out},
        ],
        tools=tool_specs.TOOL_SPECS, tokenize=False, add_generation_prompt=True,
    )

    if spliced != rendered:
        n = min(len(spliced), len(rendered))
        diverge = next((i for i in range(n) if spliced[i] != rendered[i]), n)
        raise AssertionError(
            f"splice does not match apply_chat_template\n"
            f"  diverge at char {diverge} "
            f"of {len(spliced)} (splice) / {len(rendered)} (template)\n"
            f"  shared prefix tail: {spliced[max(0, diverge - 60):diverge]!r}\n"
            f"  spliced  continues: {spliced[diverge:diverge + 80]!r}\n"
            f"  template continues: {rendered[diverge:diverge + 80]!r}"
        )


def test_format_initial_prompt_injects_tools_and_system() -> None:
    """format_initial_prompt should produce a token list that, decoded,
    contains: system prompt, the Qwen3 <tools> block listing every tool by
    name, the <tool_call>{...}</tool_call> invocation instruction, the user
    prompt, and the Qwen3-Thinking assistant opener (<think>).

    This is the inverse of the chat-template test above — it locks in that
    generate.py's *initial* prompt construction wires TOOL_SPECS through
    correctly. Without this, the model would see no tool API and produce
    code blocks our parse_action regex never matches.
    """
    tokenizer = _load_tokenizer()

    user_prompt = "Fix the broken Foo class in module bar.py — see traceback above."
    ids = tool_specs.format_initial_prompt(tokenizer, user_prompt)

    assert isinstance(ids, list), f"expected list[int], got {type(ids).__name__}"
    assert len(ids) > 100, f"rendered prompt is suspiciously short ({len(ids)} tokens)"
    assert all(isinstance(t, int) for t in ids[:10]), "expected list of ints"

    text = tokenizer.decode(ids)

    # System prompt + Qwen3 chat markers
    assert "<|im_start|>system" in text
    # A distinctive fragment of SYSTEM_PROMPT (avoids tying the test to the
    # exact wording — just confirms it's actually there)
    assert "/testbed" in text, "system prompt should mention the /testbed checkout"
    assert "finish" in text, "system prompt should mention the finish tool"

    # Qwen3 template emits the tool registry as a <tools>...</tools> block
    assert "<tools>" in text and "</tools>" in text
    # Each registered tool name must appear inside the tools block (else the
    # model can't know what to call)
    for tool in TOOL_SPECS_NAMES:
        assert tool in text, f"tool {tool!r} missing from rendered tools block"

    # Function-call format instructions
    assert "<tool_call>" in text and "</tool_call>" in text

    # User prompt verbatim
    assert user_prompt in text

    # Qwen3-Thinking auto-opens the assistant turn with <think>
    assert text.rstrip().endswith("<think>"), (
        f"expected prompt to end with <think> (Qwen3-Thinking opener), "
        f"got tail: {text[-100:]!r}"
    )


# Source of truth for tool names — kept here (not derived from TOOL_SPECS)
# so the test fails loudly if someone renames a tool without updating tests.
TOOL_SPECS_NAMES = ("bash", "str_replace_editor", "finish")


def test_parse_action_extracts_tool_call() -> None:
    text = '<think>thinking</think>\n<tool_call>{"name": "bash", "arguments": {"command": "ls -la"}}</tool_call>'
    result = tool_specs.parse_action(text)
    assert result is not None, "should parse a valid tool call"
    name, args = result
    assert name == "bash"
    assert args == {"command": "ls -la"}


def test_parse_action_returns_none_on_no_tool_call() -> None:
    assert tool_specs.parse_action("just plain text, no tool call here") is None


def test_parse_action_returns_none_on_malformed_json() -> None:
    assert tool_specs.parse_action("<tool_call>{not valid json}</tool_call>") is None


def test_parse_action_handles_unescaped_newlines_in_strings() -> None:
    # Models often emit literal newlines inside JSON string values, breaking
    # strict JSON. parse_action falls back to a forgiving second pass.
    text = '<tool_call>{"name": "bash", "arguments": {"command": "echo hello\nworld"}}</tool_call>'
    result = tool_specs.parse_action(text)
    assert result is not None, "should fall back to newline-escaped reparse"
    name, args = result
    assert name == "bash"
    assert "hello" in args["command"]


def test_clip_passes_short_content_through() -> None:
    assert tool_specs.clip("hello") == "hello"


def test_clip_truncates_long_content_with_elision_marker() -> None:
    long = "x" * (tool_specs.MAX_OBS_CHARS + 1000)
    clipped = tool_specs.clip(long)
    assert len(clipped) < len(long)
    assert "elided" in clipped
    assert clipped.startswith("x")
    assert clipped.endswith("x")


def test_clip_respects_custom_limit() -> None:
    # 100-char input clipped to 10 chars: head is first 5, tail is last 5,
    # middle replaced with elision marker. Note: the marker text itself is
    # constant-length, so for very small limits the formatted output may
    # exceed the original — clip optimizes for "model can see boundaries"
    # not "minimal length."
    s = "abcdefghij" * 10
    out = tool_specs.clip(s, limit=10)
    assert "elided" in out
    assert out.startswith("abcde")
    assert out.endswith("fghij")


def test_format_observation_wraps_content() -> None:
    """format_observation should prepend OBSERVATION_PREFIX and append OBSERVATION_SUFFIX."""
    out = tool_specs.format_observation("HELLO")
    assert out.startswith(tool_specs.OBSERVATION_PREFIX)
    assert out.endswith(tool_specs.OBSERVATION_SUFFIX)
    assert "HELLO" in out


def test_tool_specs_has_three_tools() -> None:
    """Sanity check: TOOL_SPECS lists exactly bash, str_replace_editor, finish."""
    names = {t["function"]["name"] for t in tool_specs.TOOL_SPECS}
    assert names == {"bash", "str_replace_editor", "finish"}, f"got {names}"


def test_tool_specs_have_function_schema_shape() -> None:
    """Every TOOL_SPECS entry should have type=function + function.name +
    function.description + function.parameters."""
    for spec in tool_specs.TOOL_SPECS:
        assert spec["type"] == "function"
        fn = spec["function"]
        assert "name" in fn and isinstance(fn["name"], str)
        assert "description" in fn and isinstance(fn["description"], str)
        assert "parameters" in fn and isinstance(fn["parameters"], dict)


def main() -> int:
    test_fns = [
        test_observation_markers_match_chat_template,
        test_format_initial_prompt_injects_tools_and_system,
        test_parse_action_extracts_tool_call,
        test_parse_action_returns_none_on_no_tool_call,
        test_parse_action_returns_none_on_malformed_json,
        test_parse_action_handles_unescaped_newlines_in_strings,
        test_clip_passes_short_content_through,
        test_clip_truncates_long_content_with_elision_marker,
        test_clip_respects_custom_limit,
        test_format_observation_wraps_content,
        test_tool_specs_has_three_tools,
        test_tool_specs_have_function_schema_shape,
    ]
    failures = 0
    for fn in test_fns:
        try:
            fn()
        except SystemExit:
            raise  # let _load_tokenizer's skip propagate
        except Exception as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}")
            print(f"      {exc}")
        else:
            print(f"PASS  {fn.__name__}")
    print(f"\n{len(test_fns) - failures}/{len(test_fns)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
