"""Tool-call dispatch and per-tool implementations.

Translates parsed `(name, args)` tool calls from the model into sandbox
operations, and wraps the resulting strings in chat-template observation
markers ready to splice into the rollout token tape.

Imports:
  * `tool_specs` for TOOL_SPECS schema, output formatting (format_observation,
    clip), and observation marker constants.
  * `sandbox.Sandbox` for the execution interface — sandbox.exec /
    sandbox.read_file / sandbox.write_file. Concrete sandboxes are
    constructed elsewhere (generate.py); this module just dispatches against
    whichever Sandbox handle it receives.

The undo stack for str_replace_editor (per-path backup history) lives on
`sandbox._undo_stack` (initialized in the Sandbox ABC) — same lifecycle as
the sandbox itself, so it's auto-GC'd when the sandbox closes.
"""

from __future__ import annotations

import shlex

import tool_specs
from sandbox import Sandbox


# ---------------------------------------------------------------------------
# Top-level dispatch — name → handler
# ---------------------------------------------------------------------------


async def run_tool(sandbox: Sandbox, name: str, args: dict) -> tuple[str, bool]:
    """Dispatch one parsed tool call. Returns (observation_str, done).

    `done=True` only for `finish`. The observation string is chat-template-
    wrapped and ready to splice into the rollout token tape via
    tokenizer.encode().
    """
    def obs(content: str) -> str:
        return tool_specs.format_observation(tool_specs.clip(content))

    if name == "finish":
        return "", True
    if name == "bash":
        return obs(await _bash(sandbox, args.get("command", ""))), False
    if name == "str_replace_editor":
        return obs(await _editor(sandbox, args)), False
    return obs(f"<error>unknown tool '{name}'</error>"), False


async def _bash(sandbox: Sandbox, command: str) -> str:
    if not command:
        return "<error>missing 'command' argument</error>"
    out, rc = await sandbox.exec(command)
    return out if rc == 0 else f"exit_code: {rc}\n{out}"


# ---------------------------------------------------------------------------
# str_replace_editor sub-dispatch
# ---------------------------------------------------------------------------


async def _editor(sandbox: Sandbox, args: dict) -> str:
    cmd = args.get("command", "")
    path = args.get("path", "")
    if not cmd or not path:
        return "<error>str_replace_editor requires 'command' and 'path'</error>"
    op = _EDITOR_OPS.get(cmd)
    if op is None:
        return f"<error>unknown editor command '{cmd}'</error>"
    return await op(sandbox, args)


async def _view(sandbox: Sandbox, args: dict) -> str:
    path = args["path"]
    view_range = args.get("view_range")
    _, is_dir_rc = await sandbox.exec(f"test -d {shlex.quote(path)}")
    if is_dir_rc == 0:
        out, _ = await sandbox.exec(f"ls -la {shlex.quote(path)}")
        return out
    if view_range and len(view_range) == 2:
        start, end = view_range
        out, _ = await sandbox.exec(
            f"sed -n '{start},{end}p' {shlex.quote(path)} | nl -ba -v {start}"
        )
    else:
        out, _ = await sandbox.exec(f"cat -n {shlex.quote(path)}")
    return out


async def _create(sandbox: Sandbox, args: dict) -> str:
    path = args["path"]
    file_text = args.get("file_text", "")
    _, exists_rc = await sandbox.exec(f"test -e {shlex.quote(path)}")
    if exists_rc == 0:
        return f"<error>{path} already exists; use str_replace or insert</error>"
    try:
        await sandbox.write_file(path, file_text)
    except OSError as e:
        return f"<error>{e}</error>"
    return f"created {path} ({len(file_text)} bytes)"


async def _str_replace(sandbox: Sandbox, args: dict) -> str:
    path = args["path"]
    old_str = args.get("old_str", "")
    new_str = args.get("new_str", "")
    try:
        content = await sandbox.read_file(path)
    except FileNotFoundError as e:
        return f"<error>cannot read {path}: {e}</error>"

    count = content.count(old_str)
    if count == 0:
        return f"<error>old_str not found in {path}</error>"
    if count > 1:
        return f"<error>old_str appears {count} times in {path}; must be unique</error>"

    new_content = content.replace(old_str, new_str, 1)
    sandbox._undo_stack.setdefault(path, []).append(content)
    try:
        await sandbox.write_file(path, new_content)
    except OSError as e:
        sandbox._undo_stack[path].pop()
        return f"<error>{e}</error>"
    return f"replaced 1 occurrence in {path}"


async def _insert(sandbox: Sandbox, args: dict) -> str:
    path = args["path"]
    insert_line = args.get("insert_line", 0)
    new_str = args.get("new_str", "")
    try:
        content = await sandbox.read_file(path)
    except FileNotFoundError as e:
        return f"<error>cannot read {path}: {e}</error>"

    lines = content.split("\n")
    if insert_line < 0 or insert_line > len(lines):
        return f"<error>insert_line {insert_line} out of range (0..{len(lines)})</error>"

    new_lines = lines[:insert_line] + new_str.split("\n") + lines[insert_line:]
    new_content = "\n".join(new_lines)
    sandbox._undo_stack.setdefault(path, []).append(content)
    try:
        await sandbox.write_file(path, new_content)
    except OSError as e:
        sandbox._undo_stack[path].pop()
        return f"<error>{e}</error>"
    return f"inserted at line {insert_line} in {path}"


async def _undo_edit(sandbox: Sandbox, args: dict) -> str:
    path = args["path"]
    stack = sandbox._undo_stack.get(path, [])
    if not stack:
        return f"<error>no edits to undo for {path}</error>"
    prior = stack.pop()
    try:
        await sandbox.write_file(path, prior)
    except OSError as e:
        stack.append(prior)
        return f"<error>{e}</error>"
    return f"reverted last edit to {path}"


_EDITOR_OPS = {
    "view": _view,
    "create": _create,
    "str_replace": _str_replace,
    "insert": _insert,
    "undo_edit": _undo_edit,
}
