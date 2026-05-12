#!/usr/bin/env python3
"""
Unit tests for generate_with_codingagent.py.

Pure-python pieces (parser, observation wrapper) are tested without any
Modal/SGLang. The tool-dispatch path is tested against a REAL Modal sandbox
via coding_sandbox.py — this is the only way to validate that the tool
dispatch + observation shape is what the slime runtime would actually feed
back to the model.

Skip the sandbox-driven tests by setting SKIP_LIVE_SANDBOX=1.

Run from slime repo root (MODAL_CONFIG_PATH defaults to ./.modal.toml):
  python examples/qwen3-235b_fullasync_swe-env/test_codingagent.py
"""

import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLIME_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SLIME_ROOT))

os.environ.setdefault("MODAL_CONFIG_PATH", str(SLIME_ROOT / ".modal.toml"))

# Stub slime imports so generate_with_codingagent can be imported standalone.
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
sys.modules.update({
    "slime": _fake_slime,
    "slime.rollout": _fake_rollout,
    "slime.rollout.sglang_rollout": _fake_sglang,
    "slime.utils": _fake_utils,
    "slime.utils.http_utils": _fake_http,
    "slime.utils.types": _fake_types_mod,
})

import generate_with_codingagent as gca  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, got, expected) -> None:
    if got == expected:
        results.append((PASS, name, ""))
    else:
        results.append((FAIL, name, f"\n   expected: {expected!r}\n   got:      {got!r}"))


def check_truthy(name: str, got, note: str = "") -> None:
    if got:
        results.append((PASS, name, ""))
    else:
        results.append((FAIL, name, note or f"\n   got falsy: {got!r}"))


# ---- pure-python tests --------------------------------------------------


def t_parse_run_command():
    raw = (
        "Let me explore the repo.\n"
        '<tool_call>\n{"name": "run_command", "arguments": {"cmd": "ls /testbed"}}\n</tool_call>'
    )
    name, args = gca.parse_tool_call(raw)
    check("parse run_command name", name, "run_command")
    check("parse run_command cmd", args.get("cmd"), "ls /testbed")


def t_parse_apply_patch_multiline_drift():
    # Model emits literal newlines inside the JSON string value — common drift.
    raw = (
        '<tool_call>\n{"name": "apply_patch", "arguments": {"patch": "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n"}}\n</tool_call>'
    )
    name, args = gca.parse_tool_call(raw)
    check("parse apply_patch (literal-\\n drift)", name, "apply_patch")
    check_truthy("apply_patch arg contains diff body", "+new" in args.get("patch", ""))


def t_parse_submit_no_args():
    raw = '<tool_call>\n{"name": "submit", "arguments": {}}\n</tool_call>'
    name, args = gca.parse_tool_call(raw)
    check("parse submit name", name, "submit")
    check("parse submit args", args, {})


def t_parse_no_match_rambling():
    name, args = gca.parse_tool_call("I think the fix is in core/foo.py but I'm not sure")
    check("no tool call → (None, {})", (name, args), (None, {}))


def t_parse_unknown_tool_still_parses():
    raw = '<tool_call>\n{"name": "rm_rf", "arguments": {"path": "/"}}\n</tool_call>'
    name, args = gca.parse_tool_call(raw)
    # parser doesn't validate — dispatch does. We want unknown tools to be
    # surfaceable so we can return a useful "unknown tool" observation.
    check("unknown tool name parses through", name, "rm_rf")


def t_postprocess_truncates_at_last_close():
    raw = (
        '<tool_call>\n{"name": "run_command", "arguments": {"cmd": "ls"}}\n</tool_call>'
        "\n... garbage ...\n"
    )
    out = gca.postprocess_response(raw)
    check_truthy("postprocess strips trailing garbage",
                 out.endswith("</tool_call>"),
                 f"out={out!r}")


def t_observation_has_qwen_tags():
    obs = gca._qwen_observation("hello world")
    check_truthy("obs has <|im_start|>user", "<|im_start|>user" in obs)
    check_truthy("obs has <tool_response>", "<tool_response>" in obs)
    check_truthy("obs reopens assistant <think>",
                 obs.endswith("<|im_start|>assistant\n<think>\n"))


def t_six_tools_in_spec():
    names = {t["function"]["name"] for t in gca.TOOL_SPECS}
    check("tool spec set", names,
          {"run_command", "read_file", "write_file", "apply_patch", "run_tests", "submit"})


# ---- sandbox-driven tests (real Modal sandbox) --------------------------


async def t_dispatch_against_real_sandbox(instance_id: str):
    """Exercise execute_tool() end-to-end against a real Epoch AI sandbox."""
    from coding_sandbox import CodingSandboxPool

    pool = CodingSandboxPool(max_concurrent=1, spawn_qps=2.0)
    async with pool.session(instance_id) as sandbox:
        # 1. run_command — list /testbed
        obs, done = await gca.execute_tool(sandbox, "run_command", {"cmd": "ls /testbed | wc -l"})
        check("run_command done=False", done, False)
        check_truthy("run_command obs has <tool_response>", "<tool_response>" in obs)
        # /testbed is the cloned repo root — must have >0 entries.
        check_truthy("run_command stdout looks like a count",
                     any(c.isdigit() for c in obs.split("stdout:")[-1]),
                     f"obs={obs!r}")

        # 2. read_file — read a real file in the repo
        obs, _ = await gca.execute_tool(sandbox, "read_file", {"path": "/testbed/.gitignore"})
        check_truthy("read_file returns content", len(obs) > 50, f"obs={obs[:200]!r}")

        # 3. write_file then read_file round trip
        obs, _ = await gca.execute_tool(
            sandbox, "write_file",
            {"path": "/tmp/agent_unit_test.txt", "content": "slime e2e\n"},
        )
        check_truthy("write_file done flagged false", "<tool_response>" in obs)
        readback = await sandbox.read_file("/tmp/agent_unit_test.txt")
        check("write/read round trip exact", readback, "slime e2e\n")

        # 4. apply_patch — empty patch should fail with non-zero exit (handled)
        obs, _ = await gca.execute_tool(sandbox, "apply_patch", {"patch": "not a real patch"})
        check_truthy("apply_patch returns observation",
                     "exit_code" in obs)

        # 5. unknown tool → useful error observation
        obs, _ = await gca.execute_tool(sandbox, "rm_rf", {"path": "/"})
        check_truthy("unknown tool yields <error>", "<error>" in obs)
        check_truthy("unknown tool lists available", "submit" in obs)

        # 6. submit terminates the loop
        obs, done = await gca.execute_tool(sandbox, "submit", {})
        check("submit done=True", done, True)
        check("submit returns empty obs", obs, "")


def main() -> int:
    t_parse_run_command()
    t_parse_apply_patch_multiline_drift()
    t_parse_submit_no_args()
    t_parse_no_match_rambling()
    t_parse_unknown_tool_still_parses()
    t_postprocess_truncates_at_last_close()
    t_observation_has_qwen_tags()
    t_six_tools_in_spec()

    if os.environ.get("SKIP_LIVE_SANDBOX") != "1":
        instance_id = os.environ.get("SWE_SMOKE_INSTANCE", "astropy__astropy-12907")
        try:
            asyncio.run(t_dispatch_against_real_sandbox(instance_id))
        except Exception as e:
            results.append((FAIL, "sandbox dispatch smoke", f"\n   exception: {e!r}"))

    n_pass = sum(1 for s, _, _ in results if s == PASS)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    for status, name, note in results:
        marker = "✓" if status == PASS else "✗"
        print(f"  {marker} {name}{note}")
    print(f"\n{n_pass} passed, {n_fail} failed.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
