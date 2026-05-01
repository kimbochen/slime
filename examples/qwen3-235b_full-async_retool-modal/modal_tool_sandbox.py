"""
Modal-backed async tool sandbox.

Drop-in replacement for examples/retool/tool_sandbox.py. Same public surface
(SEMAPHORE, TOOL_CONFIGS, tool_registry) so generate_with_retool.py imports
identically. Differences:

  * Code execution runs in a pool of warm Modal Sandboxes (modal.Sandbox).
  * Network is blocked at the sandbox boundary (block_network=True), which is
    the actual safety boundary — the regex "dangerous patterns" check from
    the original module is gone.
  * Everything is async-native: sandbox creation, exec, and stdout reads use
    Modal's `.aio()` variants and never call subprocess.

Pool sizing, per-exec timeout, and Modal App name are configurable via env:

  * SLIME_MODAL_APP            (default: infx-slime-retool-sandbox)
  * SLIME_MODAL_POOL_SIZE      (default: 64)
  * SLIME_MODAL_PER_EXEC_TIMEOUT (default: 60 seconds)
  * SLIME_MODAL_SANDBOX_WALLCLOCK (default: 3600 seconds)
"""

import asyncio
import atexit
import os
from typing import Any

import modal

TOOL_CONFIGS = {
    "max_turns": 16,
    "max_tool_calls": 16,
    "tool_concurrency": int(os.environ.get("SLIME_MODAL_POOL_SIZE", "64")),
    "python_timeout": int(os.environ.get("SLIME_MODAL_PER_EXEC_TIMEOUT", "60")),
    "python_memory_limit_mb": 4096,
    "python_cpu_limit": 1.0,
    "sandbox_wallclock": int(os.environ.get("SLIME_MODAL_SANDBOX_WALLCLOCK", "3600")),
    "modal_app_name": os.environ.get("SLIME_MODAL_APP", "infx-slime-retool-sandbox"),
}

SEMAPHORE = asyncio.Semaphore(TOOL_CONFIGS["tool_concurrency"])

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "sympy", "scipy", "mpmath")
)
_app: modal.App | None = None
_pool: asyncio.Queue | None = None
_pool_lock = asyncio.Lock()


async def _get_app() -> modal.App:
    global _app
    if _app is None:
        _app = await modal.App.lookup.aio(
            TOOL_CONFIGS["modal_app_name"], create_if_missing=True
        )
    return _app


async def _spawn_sandbox() -> modal.Sandbox:
    app = await _get_app()
    return await modal.Sandbox.create.aio(
        image=_image,
        app=app,
        cpu=TOOL_CONFIGS["python_cpu_limit"],
        memory=TOOL_CONFIGS["python_memory_limit_mb"],
        timeout=TOOL_CONFIGS["sandbox_wallclock"],
        block_network=True,
    )


async def _ensure_pool() -> asyncio.Queue:
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is not None:
            return _pool
        size = TOOL_CONFIGS["tool_concurrency"]
        pool: asyncio.Queue = asyncio.Queue(maxsize=size)
        sandboxes = await asyncio.gather(*[_spawn_sandbox() for _ in range(size)])
        for sb in sandboxes:
            pool.put_nowait(sb)
        _pool = pool
        return _pool


async def _acquire() -> modal.Sandbox:
    pool = await _ensure_pool()
    return await pool.get()


async def _release(sandbox: modal.Sandbox, healthy: bool) -> None:
    pool = await _ensure_pool()
    if healthy:
        pool.put_nowait(sandbox)
        return
    try:
        await sandbox.terminate.aio()
    except Exception:
        pass
    replacement = await _spawn_sandbox()
    pool.put_nowait(replacement)


def _wrap_user_code(code: str) -> str:
    indented = "\n".join("    " + line for line in code.split("\n"))
    return (
        "import sys, traceback\n"
        "from io import StringIO\n"
        "old_stdout, old_stderr = sys.stdout, sys.stderr\n"
        "stdout_capture, stderr_capture = StringIO(), StringIO()\n"
        "sys.stdout, sys.stderr = stdout_capture, stderr_capture\n"
        "try:\n"
        f"{indented}\n"
        "    out = stdout_capture.getvalue()\n"
        "    err = stderr_capture.getvalue()\n"
        "    sys.stdout, sys.stderr = old_stdout, old_stderr\n"
        "    msg = ''\n"
        "    if out:\n"
        "        msg += 'Output:\\n' + out\n"
        "    if err:\n"
        "        msg += '\\nErrors:\\n' + err\n"
        "    print(msg)\n"
        "except Exception as e:\n"
        "    sys.stdout, sys.stderr = old_stdout, old_stderr\n"
        "    print('Error: ' + str(e) + '\\nTraceback:\\n' + traceback.format_exc())\n"
    )


class ModalPythonSandbox:
    """Async Python sandbox backed by a pool of Modal Sandboxes."""

    def __init__(self, timeout: int | None = None):
        self.timeout = timeout or TOOL_CONFIGS["python_timeout"]

    async def execute_code(self, code: str) -> str:
        if not code.strip():
            return "Error: No code provided"

        wrapped = _wrap_user_code(code)
        sandbox = await _acquire()
        healthy = True
        try:
            try:
                process = await sandbox.exec.aio(
                    "python", "-c", wrapped, timeout=self.timeout
                )
            except Exception as e:
                healthy = False
                return f"Error: Failed to start exec in sandbox: {e}"

            try:
                # Belt-and-suspenders: rely on Modal's per-exec timeout, but
                # also enforce a client-side deadline (with a small grace
                # margin) so we always surface a clear "timed out" message
                # rather than a -1 exit code.
                async def _drain():
                    out = await process.stdout.read.aio()
                    err = await process.stderr.read.aio()
                    rc = await process.wait.aio()
                    return out, err, rc

                stdout, stderr, return_code = await asyncio.wait_for(
                    _drain(), timeout=self.timeout + 5
                )
            except (asyncio.TimeoutError, modal.exception.SandboxTimeoutError):
                healthy = False
                try:
                    await process.terminate.aio()
                except Exception:
                    pass
                return f"Error: Code execution timed out after {self.timeout} seconds"
            except Exception as e:
                healthy = False
                return f"Error: Sandbox exec failed: {e}"

            # Modal sends SIGKILL on its own per-exec timeout, surfacing as a
            # negative return code. Treat that as a timeout too.
            if return_code is not None and return_code < 0:
                healthy = False
                return f"Error: Code execution timed out after {self.timeout} seconds"

            if return_code != 0:
                stderr_text = stderr or ""
                return f"Error: Process exited with code {return_code}\n{stderr_text}"
            return (stdout or "").strip()
        finally:
            await _release(sandbox, healthy)


class ToolRegistry:
    """Tool registry — manages available tools and their async execution."""

    def __init__(self):
        self.tools: dict[str, dict[str, Any]] = {}
        self.python_sandbox = ModalPythonSandbox()
        self._register_default_tools()

    def _register_default_tools(self):
        self.register_tool(
            "code_interpreter",
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "description": "A tool for executing Python code in a safe sandbox environment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "The Python code to execute"}
                        },
                        "required": ["code"],
                    },
                },
            },
        )

    def register_tool(self, name: str, tool_spec: dict[str, Any]):
        self.tools[name] = tool_spec

    def get_tool_specs(self) -> list[dict[str, Any]]:
        return list(self.tools.values())

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        async with SEMAPHORE:
            if tool_name == "code_interpreter":
                return await self._execute_python(arguments)
            return f"Error: Tool '{tool_name}' not implemented"

    async def _execute_python(self, arguments: dict[str, Any]) -> str:
        code = arguments.get("code", "")
        if not code.strip():
            return "Error: No code provided"
        return await self.python_sandbox.execute_code(code)


tool_registry = ToolRegistry()


def _drain_pool_sync():
    """atexit hook — best-effort terminate all warm sandboxes."""
    global _pool
    if _pool is None:
        return
    sandboxes = []
    while not _pool.empty():
        try:
            sandboxes.append(_pool.get_nowait())
        except asyncio.QueueEmpty:
            break

    async def _terminate_all():
        await asyncio.gather(
            *[sb.terminate.aio() for sb in sandboxes], return_exceptions=True
        )

    try:
        asyncio.run(_terminate_all())
    except Exception:
        pass


atexit.register(_drain_pool_sync)
