#!/usr/bin/env python3
"""Tests for examples/swe-bench/tool_runtime.py.

Uses an in-memory MockSandbox so dispatch and editor logic can be tested
without Modal. Covers:
  * run_tool top-level dispatch (finish, bash, str_replace_editor, unknown)
  * _bash error paths (missing command, nonzero exit)
  * str_replace_editor: missing/unknown sub-command
  * each editor sub-command: view (file vs dir), create (new vs existing),
    str_replace (unique, not-found, ambiguous), insert (success + out-of-range),
    undo_edit (with and without history)
  * undo stack interaction with str_replace and insert

Run from anywhere:
    python examples/swe-bench/tests/test_tool_runtime.py
"""

import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE_DIR = HERE.parent
sys.path.insert(0, str(EXAMPLE_DIR))

import tool_runtime  # noqa: E402
from sandbox import PER_CMD_TIMEOUT_S, Sandbox  # noqa: E402


# ---------------------------------------------------------------------------
# MockSandbox — in-memory fake to test dispatch + editor logic without Modal.
# ---------------------------------------------------------------------------


class MockSandbox(Sandbox):
    """Records exec calls and stores a virtual filesystem in a dict.

    `exec_returns` is checked by substring against the incoming command; the
    first matching entry wins. Falls back to (output="", exit_code=0).

    `files` is a dict[path, content] backing read_file / write_file directly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.files: dict[str, str] = {}
        self.exec_log: list[str] = []
        self.exec_returns: dict[str, tuple[str, int]] = {}

    async def exec(self, command: str, *, timeout: int = PER_CMD_TIMEOUT_S) -> tuple[str, int]:
        self.exec_log.append(command)
        for substr, ret in self.exec_returns.items():
            if substr in command:
                return ret
        return "", 0

    async def read_file(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write_file(self, path: str, content: str) -> None:
        self.files[path] = content

    async def close(self) -> None: pass
    async def detach(self) -> None: pass


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# run_tool top-level dispatch
# ---------------------------------------------------------------------------


def test_finish_returns_done() -> None:
    sb = MockSandbox()
    out, done = _run(tool_runtime.run_tool(sb, "finish", {}))
    assert done is True
    assert out == "", f"finish should return empty observation, got {out!r}"


def test_unknown_tool_returns_error_observation() -> None:
    sb = MockSandbox()
    out, done = _run(tool_runtime.run_tool(sb, "nonexistent_tool", {}))
    assert done is False
    assert "unknown tool" in out
    assert "nonexistent_tool" in out


# ---------------------------------------------------------------------------
# _bash
# ---------------------------------------------------------------------------


def test_bash_success_returns_output() -> None:
    sb = MockSandbox()
    sb.exec_returns["echo hi"] = ("hi\n", 0)
    out, done = _run(tool_runtime.run_tool(sb, "bash", {"command": "echo hi"}))
    assert done is False
    assert "hi" in out


def test_bash_missing_command_returns_error() -> None:
    sb = MockSandbox()
    out, _ = _run(tool_runtime.run_tool(sb, "bash", {}))
    assert "missing 'command'" in out


def test_bash_nonzero_exit_prefixes_exit_code() -> None:
    sb = MockSandbox()
    sb.exec_returns["false"] = ("stderr blob", 1)
    out, _ = _run(tool_runtime.run_tool(sb, "bash", {"command": "false"}))
    assert "exit_code: 1" in out
    assert "stderr blob" in out


# ---------------------------------------------------------------------------
# str_replace_editor sub-dispatch validation
# ---------------------------------------------------------------------------


def test_editor_missing_command() -> None:
    sb = MockSandbox()
    out, _ = _run(tool_runtime.run_tool(sb, "str_replace_editor", {"path": "/x"}))
    assert "requires 'command' and 'path'" in out


def test_editor_missing_path() -> None:
    sb = MockSandbox()
    out, _ = _run(tool_runtime.run_tool(sb, "str_replace_editor", {"command": "view"}))
    assert "requires 'command' and 'path'" in out


def test_editor_unknown_sub_command() -> None:
    sb = MockSandbox()
    out, _ = _run(tool_runtime.run_tool(
        sb, "str_replace_editor", {"command": "delete", "path": "/x"}
    ))
    assert "unknown editor command 'delete'" in out


# ---------------------------------------------------------------------------
# view
# ---------------------------------------------------------------------------


def test_view_file_returns_cat_output() -> None:
    sb = MockSandbox()
    sb.exec_returns["test -d"] = ("", 1)         # not a dir
    sb.exec_returns["cat -n"] = ("     1\thello\n", 0)
    out, _ = _run(tool_runtime.run_tool(
        sb, "str_replace_editor", {"command": "view", "path": "/file"}
    ))
    assert "hello" in out


def test_view_directory_returns_ls_output() -> None:
    sb = MockSandbox()
    sb.exec_returns["test -d"] = ("", 0)         # is a dir
    sb.exec_returns["ls -la"] = ("total 8\ndrwxr-xr-x ...", 0)
    out, _ = _run(tool_runtime.run_tool(
        sb, "str_replace_editor", {"command": "view", "path": "/dir"}
    ))
    assert "total" in out


def test_view_with_range_uses_sed() -> None:
    sb = MockSandbox()
    sb.exec_returns["test -d"] = ("", 1)
    sb.exec_returns["sed -n"] = ("     5\tline5\n", 0)
    _run(tool_runtime.run_tool(
        sb, "str_replace_editor",
        {"command": "view", "path": "/f", "view_range": [5, 5]},
    ))
    assert any("sed -n" in cmd and "'5,5p'" in cmd for cmd in sb.exec_log), sb.exec_log


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_new_file_writes() -> None:
    sb = MockSandbox()
    sb.exec_returns["test -e"] = ("", 1)         # doesn't exist
    out, _ = _run(tool_runtime.run_tool(
        sb, "str_replace_editor",
        {"command": "create", "path": "/new", "file_text": "hello world"},
    ))
    assert "created /new" in out
    assert sb.files["/new"] == "hello world"


def test_create_existing_file_returns_error() -> None:
    sb = MockSandbox()
    sb.exec_returns["test -e"] = ("", 0)         # exists
    out, _ = _run(tool_runtime.run_tool(
        sb, "str_replace_editor",
        {"command": "create", "path": "/existing", "file_text": "x"},
    ))
    assert "already exists" in out


# ---------------------------------------------------------------------------
# str_replace
# ---------------------------------------------------------------------------


def test_str_replace_unique_succeeds_and_pushes_undo() -> None:
    sb = MockSandbox()
    sb.files["/file"] = "foo bar baz"
    out, _ = _run(tool_runtime.run_tool(
        sb, "str_replace_editor",
        {"command": "str_replace", "path": "/file", "old_str": "bar", "new_str": "BAZ"},
    ))
    assert "replaced 1" in out
    assert sb.files["/file"] == "foo BAZ baz"
    assert sb._undo_stack["/file"] == ["foo bar baz"], sb._undo_stack


def test_str_replace_not_found_returns_error() -> None:
    sb = MockSandbox()
    sb.files["/file"] = "foo bar"
    out, _ = _run(tool_runtime.run_tool(
        sb, "str_replace_editor",
        {"command": "str_replace", "path": "/file", "old_str": "missing", "new_str": "y"},
    ))
    assert "not found" in out
    assert sb.files["/file"] == "foo bar"        # unchanged


def test_str_replace_not_unique_returns_error() -> None:
    sb = MockSandbox()
    sb.files["/file"] = "x x x"
    out, _ = _run(tool_runtime.run_tool(
        sb, "str_replace_editor",
        {"command": "str_replace", "path": "/file", "old_str": "x", "new_str": "y"},
    ))
    assert "must be unique" in out
    assert "appears 3 times" in out
    assert sb.files["/file"] == "x x x"          # unchanged


def test_str_replace_on_missing_file_returns_error() -> None:
    sb = MockSandbox()
    out, _ = _run(tool_runtime.run_tool(
        sb, "str_replace_editor",
        {"command": "str_replace", "path": "/nope", "old_str": "a", "new_str": "b"},
    ))
    assert "cannot read" in out


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------


def test_insert_succeeds_at_valid_line() -> None:
    sb = MockSandbox()
    sb.files["/f"] = "a\nb\nc"
    out, _ = _run(tool_runtime.run_tool(
        sb, "str_replace_editor",
        {"command": "insert", "path": "/f", "insert_line": 1, "new_str": "INS"},
    ))
    assert "inserted at line 1" in out
    assert sb.files["/f"] == "a\nINS\nb\nc"


def test_insert_at_top_with_line_zero() -> None:
    sb = MockSandbox()
    sb.files["/f"] = "a\nb"
    _run(tool_runtime.run_tool(
        sb, "str_replace_editor",
        {"command": "insert", "path": "/f", "insert_line": 0, "new_str": "TOP"},
    ))
    assert sb.files["/f"] == "TOP\na\nb"


def test_insert_out_of_range_returns_error() -> None:
    sb = MockSandbox()
    sb.files["/f"] = "a\nb"
    out, _ = _run(tool_runtime.run_tool(
        sb, "str_replace_editor",
        {"command": "insert", "path": "/f", "insert_line": 99, "new_str": "x"},
    ))
    assert "out of range" in out
    assert sb.files["/f"] == "a\nb"


# ---------------------------------------------------------------------------
# undo_edit
# ---------------------------------------------------------------------------


def test_undo_edit_reverts_last_str_replace() -> None:
    sb = MockSandbox()
    sb.files["/f"] = "foo bar"
    _run(tool_runtime.run_tool(
        sb, "str_replace_editor",
        {"command": "str_replace", "path": "/f", "old_str": "bar", "new_str": "BAZ"},
    ))
    assert sb.files["/f"] == "foo BAZ"
    out, _ = _run(tool_runtime.run_tool(
        sb, "str_replace_editor", {"command": "undo_edit", "path": "/f"},
    ))
    assert "reverted" in out
    assert sb.files["/f"] == "foo bar"
    # Stack should be empty after popping the one entry
    assert sb._undo_stack["/f"] == []


def test_undo_edit_with_no_history_returns_error() -> None:
    sb = MockSandbox()
    out, _ = _run(tool_runtime.run_tool(
        sb, "str_replace_editor", {"command": "undo_edit", "path": "/no/edits"},
    ))
    assert "no edits to undo" in out


def test_undo_stacks_per_path_independently() -> None:
    sb = MockSandbox()
    sb.files["/a"] = "a-original"
    sb.files["/b"] = "b-original"
    _run(tool_runtime.run_tool(
        sb, "str_replace_editor",
        {"command": "str_replace", "path": "/a", "old_str": "a-original", "new_str": "a-new"},
    ))
    _run(tool_runtime.run_tool(
        sb, "str_replace_editor",
        {"command": "str_replace", "path": "/b", "old_str": "b-original", "new_str": "b-new"},
    ))
    # Undo /a only
    _run(tool_runtime.run_tool(
        sb, "str_replace_editor", {"command": "undo_edit", "path": "/a"},
    ))
    assert sb.files["/a"] == "a-original"
    assert sb.files["/b"] == "b-new"             # /b untouched


# ---------------------------------------------------------------------------
# Output formatting (observations are chat-marker-wrapped)
# ---------------------------------------------------------------------------


def test_observation_is_chat_marker_wrapped() -> None:
    """run_tool always wraps the inner content via tool_specs.format_observation."""
    import tool_specs
    sb = MockSandbox()
    sb.exec_returns["echo hi"] = ("hi", 0)
    out, _ = _run(tool_runtime.run_tool(sb, "bash", {"command": "echo hi"}))
    assert out.startswith(tool_specs.OBSERVATION_PREFIX)
    assert out.endswith(tool_specs.OBSERVATION_SUFFIX)


def main() -> int:
    test_fns = [
        test_finish_returns_done,
        test_unknown_tool_returns_error_observation,
        test_bash_success_returns_output,
        test_bash_missing_command_returns_error,
        test_bash_nonzero_exit_prefixes_exit_code,
        test_editor_missing_command,
        test_editor_missing_path,
        test_editor_unknown_sub_command,
        test_view_file_returns_cat_output,
        test_view_directory_returns_ls_output,
        test_view_with_range_uses_sed,
        test_create_new_file_writes,
        test_create_existing_file_returns_error,
        test_str_replace_unique_succeeds_and_pushes_undo,
        test_str_replace_not_found_returns_error,
        test_str_replace_not_unique_returns_error,
        test_str_replace_on_missing_file_returns_error,
        test_insert_succeeds_at_valid_line,
        test_insert_at_top_with_line_zero,
        test_insert_out_of_range_returns_error,
        test_undo_edit_reverts_last_str_replace,
        test_undo_edit_with_no_history_returns_error,
        test_undo_stacks_per_path_independently,
        test_observation_is_chat_marker_wrapped,
    ]
    failures = 0
    for fn in test_fns:
        try:
            fn()
        except Exception as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}")
            print(f"      {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {fn.__name__}")
    print(f"\n{len(test_fns) - failures}/{len(test_fns)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
