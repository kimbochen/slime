"""
coding_sandbox.py
=================

Per-instance Modal sandbox pool for SWE-bench-Verified-style agentic coding.

Design influenced by tinker-cookbook/sandbox/modal_sandbox.py:
  - One-shot per AGENT ROLLOUT (not per shell command): a single sandbox
    persists for the duration of one SWE-bench instance's multi-turn agent
    trajectory, then terminates. (Tinker terminates per-request; we keep
    them alive across turns of the same instance because applying patches
    + running tests + iterating only makes sense on shared state.)
  - General-purpose API: run_command, write_file, read_file, apply_patch,
    run_tests, heartbeat, cleanup — not the single-purpose `execute_code`
    of our retool sandbox.
  - Per-instance Docker images from Epoch AI's SWE-bench registry
    (epochai/sweb.eval.x86_64.<instance_id>) instead of a generic
    debian_slim. Each image already has the repo cloned, deps installed,
    and the SWE-bench test invocation configured.
  - Byte-capped stream reads (128 KB default) to prevent unbounded
    stdout/stderr from OOMing the rollout worker.
  - Heartbeats to detect dead sandboxes proactively (vs. discovering at
    the next command).
  - Concurrency-bounded + rate-limited spawn (Modal's control plane gets
    sad if you create 200 sandboxes in 100ms).

Compared to our retool `modal_tool_sandbox.py`:
  | aspect             | modal_tool_sandbox (retool)     | coding_sandbox (this file)        |
  | ----               | ----                             | ----                              |
  | lifetime           | reused across many python execs  | one per agent rollout, then dies  |
  | image              | debian_slim + sympy/scipy        | per-instance epochai/sweb.eval.*  |
  | API                | execute_code(python_src)         | run_command, read/write_file, ... |
  | network            | block_network=True               | unblocked (pip install, git, etc.)|
  | state retention    | none (each call independent)     | filesystem persists across turns  |
  | byte cap on stdout | no                               | 128 KB (configurable)             |
  | heartbeat          | no                               | yes                               |
  | rate-limited spawn | no (eager gather)                | yes (configurable QPS)            |

Env vars (all optional):
  SLIME_SWEBENCH_APP            (default: infx-slime-swebench-sandbox)
  SLIME_SWEBENCH_MAX_CONCURRENT (default: 32) — max sandboxes alive at once
  SLIME_SWEBENCH_SPAWN_QPS      (default: 4)  — spawns per second cap
  SLIME_SWEBENCH_TIMEOUT        (default: 1800) — sandbox wallclock seconds
  SLIME_SWEBENCH_PER_CMD_TIMEOUT (default: 120) — per-command timeout
  SLIME_SWEBENCH_MAX_STREAM_BYTES (default: 131072) — 128 KB
  SLIME_SWEBENCH_IMAGE_REGISTRY (default: docker.io/epochai)
  SLIME_SWEBENCH_WORKDIR        (default: /testbed) — SWE-bench convention
"""

from __future__ import annotations

import asyncio
import atexit
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "app_name": os.environ.get(
        "SLIME_SWEBENCH_APP", "infx-slime-swebench-sandbox"
    ),
    "max_concurrent": int(os.environ.get("SLIME_SWEBENCH_MAX_CONCURRENT", "32")),
    "spawn_qps": float(os.environ.get("SLIME_SWEBENCH_SPAWN_QPS", "4")),
    "timeout": int(os.environ.get("SLIME_SWEBENCH_TIMEOUT", "1800")),
    "per_cmd_timeout": int(os.environ.get("SLIME_SWEBENCH_PER_CMD_TIMEOUT", "120")),
    "max_stream_bytes": int(os.environ.get("SLIME_SWEBENCH_MAX_STREAM_BYTES", "131072")),
    "image_registry": os.environ.get("SLIME_SWEBENCH_IMAGE_REGISTRY", "ghcr.io/epoch-research"),
    "workdir": os.environ.get("SLIME_SWEBENCH_WORKDIR", "/testbed"),
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class CommandResult:
    """Result of a single `run_command` call inside the sandbox."""
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def short_repr(self, max_lines: int = 20) -> str:
        """For LLM observation: clip to N lines per stream."""
        def clip(s: str) -> str:
            lines = s.splitlines()
            if len(lines) <= max_lines:
                return s
            return "\n".join(lines[: max_lines // 2]) + f"\n... [{len(lines) - max_lines} lines elided] ...\n" + "\n".join(lines[-max_lines // 2:])
        out = []
        if self.timed_out:
            out.append(f"<TIMEOUT after {self.duration_s:.1f}s>")
        out.append(f"exit_code: {self.exit_code}")
        if self.stdout.strip():
            out.append(f"stdout:\n{clip(self.stdout)}")
        if self.stderr.strip():
            out.append(f"stderr:\n{clip(self.stderr)}")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _instance_image_tag(instance_id: str, registry: str | None = None) -> str:
    """Compute the Docker image tag for an SWE-bench-Verified instance.

    Epoch AI publishes prebuilt per-instance images at:
      ghcr.io/epoch-research/swe-bench.eval.x86_64.<instance_id>:latest

    No normalization of `__` — ghcr.io accepts the raw instance_id. (The
    SWE-bench Docker Hub mirror uses `_1776_` for `__`, but Epoch's ghcr.io
    registry — what the official harness defaults to — does not.)
    """
    registry = registry or CONFIG["image_registry"]
    return f"{registry}/swe-bench.eval.x86_64.{instance_id}:latest"


async def _read_stream_capped(stream: Any, max_bytes: int) -> str:
    """Read from a Modal ContainerProcess output stream with a byte cap.

    Drains any remaining data so the underlying process can exit cleanly,
    but discards bytes beyond the cap.
    """
    buf = bytearray()
    remaining = max_bytes
    async for chunk in stream:
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8", errors="replace")
        if remaining > 0:
            take = chunk[:remaining]
            buf.extend(take)
            remaining -= len(take)
        # else: drain to EOF but discard
    return bytes(buf).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# CodingSandbox: one sandbox = one SWE-bench instance's lifetime
# ---------------------------------------------------------------------------


class CodingSandbox:
    """A Modal sandbox bound to one SWE-bench-Verified instance.

    Lifetime model: created at the START of an agent rollout for a specific
    instance_id, used across many turns (run_command, file I/O, patch
    application, test runs), then cleaned up at end of rollout.
    """

    def __init__(
        self,
        *,
        sandbox: "modal.Sandbox",
        instance_id: str,
        max_stream_bytes: int,
        workdir: str,
        per_cmd_timeout: int,
    ):
        self._sandbox = sandbox
        self.instance_id = instance_id
        self.max_stream_bytes = max_stream_bytes
        self.workdir = workdir
        self.per_cmd_timeout = per_cmd_timeout
        self.id: str = sandbox.object_id
        self._terminated = False

    @classmethod
    async def create(
        cls,
        instance_id: str,
        *,
        app: "modal.App",
        image_registry: str | None = None,
        timeout: int | None = None,
        per_cmd_timeout: int | None = None,
        max_stream_bytes: int | None = None,
        workdir: str | None = None,
        cpu: float = 2.0,
        memory_mb: int = 8192,
        block_network: bool = False,
    ) -> "CodingSandbox":
        """Spawn a sandbox using the per-instance Epoch AI image."""
        image_tag = _instance_image_tag(instance_id, registry=image_registry)
        # `force_build=False` reuses Modal's image cache; the first time
        # Modal pulls this tag it caches the layers on its side.
        image = modal.Image.from_registry(image_tag, force_build=False)
        sandbox = await modal.Sandbox.create.aio(
            image=image,
            app=app,
            cpu=cpu,
            memory=memory_mb,
            timeout=timeout or CONFIG["timeout"],
            workdir=workdir or CONFIG["workdir"],
            block_network=block_network,
        )
        return cls(
            sandbox=sandbox,
            instance_id=instance_id,
            max_stream_bytes=max_stream_bytes or CONFIG["max_stream_bytes"],
            workdir=workdir or CONFIG["workdir"],
            per_cmd_timeout=per_cmd_timeout or CONFIG["per_cmd_timeout"],
        )

    # ---- shell ---------------------------------------------------------

    async def run_command(
        self,
        cmd: str,
        *,
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Run a shell command via `bash -c`, return captured streams."""
        wall_timeout = timeout or self.per_cmd_timeout
        t0 = time.monotonic()
        # Prefix with cd if cwd specified; otherwise sandbox stays at workdir
        full = f"cd {cwd} && {cmd}" if cwd else cmd
        try:
            process = await self._sandbox.exec.aio(
                "bash", "-c", full, timeout=wall_timeout,
                **({"env": env} if env else {}),
            )
        except modal.exception.SandboxTimeoutError:
            return CommandResult(
                stdout="", stderr="<sandbox-level timeout>",
                exit_code=-1, timed_out=True, duration_s=time.monotonic() - t0,
            )

        try:
            async def _drain():
                stdout = await _read_stream_capped(process.stdout, self.max_stream_bytes)
                stderr = await _read_stream_capped(process.stderr, self.max_stream_bytes)
                rc = await process.wait.aio()
                return stdout, stderr, rc

            stdout, stderr, rc = await asyncio.wait_for(_drain(), timeout=wall_timeout + 5)
        except (asyncio.TimeoutError, modal.exception.SandboxTimeoutError):
            try:
                await process.terminate.aio()
            except Exception:
                pass
            return CommandResult(
                stdout="", stderr=f"<timed out after {wall_timeout}s>",
                exit_code=-1, timed_out=True, duration_s=time.monotonic() - t0,
            )

        # Modal's wait() returns negative exit code on SIGKILL (e.g., from
        # per-exec timeout). Treat negative as a timeout.
        timed_out = rc is not None and rc < 0
        return CommandResult(
            stdout=stdout, stderr=stderr,
            exit_code=int(rc or 0),
            timed_out=timed_out,
            duration_s=time.monotonic() - t0,
        )

    # ---- file I/O ------------------------------------------------------

    async def write_file(self, path: str, content: bytes | str, *, executable: bool = False) -> CommandResult:
        """Write a file via chunked stdin to `tee`. Chunked at 2MB to
        avoid blowing up the stdin pipe for large patches/test results."""
        if isinstance(content, str):
            content = content.encode("utf-8")

        try:
            process = await self._sandbox.exec.aio(
                "bash", "-c", f"mkdir -p $(dirname {path}) && cat > {path}",
                timeout=self.per_cmd_timeout,
            )
        except Exception as e:
            return CommandResult("", f"<failed to start tee: {e}>", -1, False, 0.0)

        chunk_size = 2 * 1024 * 1024  # 2 MB
        try:
            # Modal 1.x: stdin.write/write_eof are sync (just buffer); drain
            # is the async flush. There is no `close.aio` — use write_eof.
            for i in range(0, len(content), chunk_size):
                process.stdin.write(content[i: i + chunk_size])
            process.stdin.write_eof()
            await process.stdin.drain.aio()
            rc = await process.wait.aio()
        except Exception as e:
            return CommandResult("", f"<write failure: {e}>", -1, False, 0.0)

        if executable and rc == 0:
            return await self.run_command(f"chmod +x {path}")
        return CommandResult("", "", int(rc or 0), False, 0.0)

    async def read_file(self, path: str, *, max_bytes: int | None = None) -> str:
        """Read a file with byte cap. Uses `head -c MAX` to avoid loading
        massive files into the rollout worker's memory."""
        cap = max_bytes or self.max_stream_bytes
        result = await self.run_command(f"head -c {cap} {path}")
        return result.stdout

    # ---- SWE-bench-specific conveniences -------------------------------

    async def apply_patch(self, patch_text: str) -> CommandResult:
        """Apply a unified-diff patch via `git apply`. Returns the result of
        the `git apply` invocation — caller should check `.ok`."""
        # Write to a temp file because passing huge patches on stdin can
        # be flaky; also gives us a useful filename for error reporting.
        write_res = await self.write_file("/tmp/agent_patch.diff", patch_text)
        if write_res.exit_code != 0:
            return write_res
        return await self.run_command("git apply --whitespace=nowarn /tmp/agent_patch.diff", cwd=self.workdir)

    async def run_tests(self, test_command: str | None = None) -> CommandResult:
        """Run the SWE-bench evaluation tests inside the instance.

        If `test_command` is None, falls back to the convention from
        SWE-bench Verified images: a `/eval.sh` (or `/run_tests.sh`) sits
        at the image root, encoding the FAIL_TO_PASS + PASS_TO_PASS set
        and the project's test invocation. The Epoch AI images bake this in.
        """
        cmd = test_command or "/eval.sh"
        # Tests can run long; bump timeout. SWE-bench instances cap at ~1200s
        # in the official harness; we mirror that.
        return await self.run_command(cmd, timeout=1200)

    # ---- lifecycle -----------------------------------------------------

    async def heartbeat(self) -> bool:
        """Run `true` to confirm the sandbox is still alive."""
        try:
            res = await self.run_command("true", timeout=5)
            return res.exit_code == 0
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Terminate the sandbox. Idempotent."""
        if self._terminated:
            return
        try:
            await self._sandbox.terminate.aio()
        except Exception:
            pass
        self._terminated = True


# ---------------------------------------------------------------------------
# CodingSandboxPool: spawn-bounded, rate-limited launcher
# ---------------------------------------------------------------------------


class CodingSandboxPool:
    """Manages live coding sandboxes — bounds concurrency and rate-limits
    spawns. Unlike the retool sandbox pool, this is NOT a warm pool: each
    sandbox is per-instance, so pre-warming is pointless. The pool here
    is just an acquire/release wrapper with concurrency control."""

    def __init__(
        self,
        *,
        app_name: str | None = None,
        max_concurrent: int | None = None,
        spawn_qps: float | None = None,
    ):
        self.app_name = app_name or CONFIG["app_name"]
        self._max_concurrent = max_concurrent or CONFIG["max_concurrent"]
        self._spawn_qps = spawn_qps or CONFIG["spawn_qps"]
        # Lazy: don't look up the Modal App until first acquire (avoids
        # forcing Modal auth on import-time).
        self._app: "modal.App | None" = None
        self._app_lock = asyncio.Lock()
        # Cap simultaneous LIVE sandboxes
        self._concurrency = asyncio.Semaphore(self._max_concurrent)
        # Cap spawn rate (token bucket-style: at most spawn_qps creates / sec)
        self._spawn_token = asyncio.Semaphore(1)
        self._last_spawn_t = 0.0
        self._active: dict[str, CodingSandbox] = {}
        self._active_lock = asyncio.Lock()
        atexit.register(self._sync_terminate_all)

    async def _get_app(self) -> "modal.App":
        if self._app is None:
            async with self._app_lock:
                if self._app is None:
                    self._app = await modal.App.lookup.aio(
                        self.app_name, create_if_missing=True
                    )
        return self._app

    async def _await_spawn_slot(self) -> None:
        """Block until at least 1/spawn_qps seconds have passed since the
        previous spawn. Simple leaky-bucket; good enough."""
        async with self._spawn_token:
            min_dt = 1.0 / max(self._spawn_qps, 0.01)
            elapsed = time.monotonic() - self._last_spawn_t
            if elapsed < min_dt:
                await asyncio.sleep(min_dt - elapsed)
            self._last_spawn_t = time.monotonic()

    async def acquire(self, instance_id: str, **create_kwargs: Any) -> CodingSandbox:
        """Spawn a sandbox for `instance_id`. Caller MUST call `release`
        (or use `async with self.session(instance_id)`)."""
        await self._concurrency.acquire()
        try:
            await self._await_spawn_slot()
            app = await self._get_app()
            sandbox = await CodingSandbox.create(instance_id, app=app, **create_kwargs)
            async with self._active_lock:
                self._active[sandbox.id] = sandbox
            return sandbox
        except Exception:
            self._concurrency.release()
            raise

    async def release(self, sandbox: CodingSandbox) -> None:
        async with self._active_lock:
            self._active.pop(sandbox.id, None)
        try:
            await sandbox.cleanup()
        finally:
            self._concurrency.release()

    def session(self, instance_id: str, **create_kwargs: Any) -> "_SandboxSession":
        """Async context manager: `async with pool.session(inst_id) as sb: ...`"""
        return _SandboxSession(self, instance_id, create_kwargs)

    def _sync_terminate_all(self) -> None:
        """atexit hook — best-effort termination of any leaked sandboxes."""
        if not self._active:
            return

        async def _all():
            await asyncio.gather(
                *(sb.cleanup() for sb in list(self._active.values())),
                return_exceptions=True,
            )

        try:
            asyncio.run(_all())
        except Exception:
            pass


class _SandboxSession:
    """Async context manager for a per-rollout sandbox session."""

    def __init__(self, pool: CodingSandboxPool, instance_id: str, kwargs: dict):
        self._pool = pool
        self._instance_id = instance_id
        self._kwargs = kwargs
        self._sandbox: CodingSandbox | None = None

    async def __aenter__(self) -> CodingSandbox:
        self._sandbox = await self._pool.acquire(self._instance_id, **self._kwargs)
        return self._sandbox

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._sandbox is not None:
            await self._pool.release(self._sandbox)


# ---------------------------------------------------------------------------
# Module-level pool singleton (lazy)
# ---------------------------------------------------------------------------

_pool: CodingSandboxPool | None = None


def get_pool() -> CodingSandboxPool:
    global _pool
    if _pool is None:
        _pool = CodingSandboxPool()
    return _pool
