"""Execution-side concerns for the SWE-bench rollout: Sandbox interface,
Modal-backed implementation, tool dispatch, and the sample-tied lifecycle
finalize helper.

Lifecycle model:
  * One sandbox per sample, alive across resumes via sample.metadata["sandbox_id"].
  * Lazy creation — Modal API isn't called until the first tool call.
  * On terminal sample status (COMPLETED/TRUNCATED/FAILED): close (terminate
    Modal container, forget id). On ABORTED: detach (release Python handle,
    leave Modal container alive for the next resume to reattach).
  * No global cleanup hook here — leaked sandboxes self-destruct after the
    Modal wall-clock timeout, or you can run scripts/cleanup_sandboxes.py.

Concurrency: warn-only counter on active sandboxes. No gating. Modal handles
real rate limits; this is just a heuristic to spot leaks during development.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level state — warn counter, no semaphore gating
# ---------------------------------------------------------------------------

_active_count: int = 0
_WARN_THRESHOLD = int(os.environ.get("SLIME_SWEBENCH_WARN_CONCURRENT", "128"))


# ---------------------------------------------------------------------------
# Run-wide config — set once at module load, used for every sandbox in this
# process. Env-overrideable so cluster jobs can tweak without code changes.
# ---------------------------------------------------------------------------

APP_NAME = os.environ.get("SLIME_SWEBENCH_APP", "slime-swebench-sandbox")
SANDBOX_TIMEOUT_S = int(os.environ.get("SLIME_SWEBENCH_TIMEOUT", "1800"))
PER_CMD_TIMEOUT_S = int(os.environ.get("SLIME_SWEBENCH_PER_CMD_TIMEOUT", "120"))
MAX_OUTPUT_BYTES = int(os.environ.get("SLIME_SWEBENCH_MAX_OUTPUT_BYTES", str(64 * 1024)))
CPU = float(os.environ.get("SLIME_SWEBENCH_CPU", "2.0"))
MEMORY_MB = int(os.environ.get("SLIME_SWEBENCH_MEMORY_MB", "8192"))
WORKDIR = "/testbed"  # SWE-bench Epoch AI images check out the repo here


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SandboxCreateError(RuntimeError):
    """Modal failed to spawn a fresh sandbox (image pull, quota, network, ...)."""


class SandboxReattachError(RuntimeError):
    """A parked sandbox referenced by sample.metadata['sandbox_id'] is no longer
    reachable — likely timed out during a long buffer wait, or evicted."""


class SandboxDiedError(RuntimeError):
    """The Modal sandbox stopped running mid-rollout (wall-clock timeout, OOM,
    eviction, external termination). Detected via poll() before issuing the
    next operation."""


# ---------------------------------------------------------------------------
# Sandbox ABC — narrow primitives
# ---------------------------------------------------------------------------


class Sandbox:
    """Narrow execution interface. Subclasses provide a real backend.

    `sandbox_id` is a public mutable attribute: subclasses update it when they
    create or reattach to the underlying container so generate.py can persist
    it to sample.metadata for cross-resume reattach. Initialized to None;
    subclasses set it in their own __init__.

    The undo stack is also initialized here so any Sandbox subclass gets it
    for free, and tool dispatch (run_tool) can rely on it being present.
    """

    def __init__(self) -> None:
        self._undo_stack: dict[str, list[str]] = {}
        self.sandbox_id: str | None = None

    async def exec(self, command: str, *, timeout: int = PER_CMD_TIMEOUT_S) -> tuple[str, int]:
        """Run a command. Return (merged_stdout_stderr, exit_code).
        Defaults to PER_CMD_TIMEOUT_S; override for long-running commands."""
        raise NotImplementedError

    async def read_file(self, path: str) -> str:
        """Read a file's contents as UTF-8 text. Raise FileNotFoundError if missing."""
        raise NotImplementedError

    async def write_file(self, path: str, content: str) -> None:
        """Write UTF-8 text to a file, creating parent dirs as needed."""
        raise NotImplementedError

    async def close(self) -> None:
        """Terminate the underlying container and release any concurrency slot."""
        raise NotImplementedError

    async def detach(self) -> None:
        """Drop the Python handle without terminating the container — for the
        sample-aborted path where we want the container to survive for resume."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SWEBenchSandbox — Modal-backed concrete implementation
# ---------------------------------------------------------------------------


class SWEBenchSandbox(Sandbox):
    """Modal sandbox bound to one SWE-bench-Verified instance.

    Constructed cheaply (sync, no I/O). Modal sandbox is created or reattached
    lazily on the first call into exec/read_file/write_file. Call close() on
    terminal sample status to terminate; detach() on abort to keep alive.
    """

    def __init__(self, instance_id: str, sandbox_id: str | None) -> None:
        super().__init__()
        self.instance_id = instance_id
        # sandbox_id: pre-activation it's the reattach hint (None means
        # "create fresh on first use"). Post-activation it's the live ID.
        # Updated in _ensure_sandbox after create/reattach succeeds.
        self.sandbox_id = sandbox_id
        self._sandbox: Any | None = None

    # ---- lazy create / reattach -------------------------------------------
    async def _ensure_sandbox(self) -> Any:
        # FAST PATH — already activated. poll() catches mid-rollout death cheaply
        # (no container work, just a control-plane status query).
        if self._sandbox is not None:
            rc = await self._sandbox.poll.aio()
            if rc is not None:
                raise SandboxDiedError(
                    f"sandbox {self.sandbox_id} died mid-rollout (exit code {rc}); "
                    f"likely hit wall-clock timeout, OOM, or external termination"
                )
            return self._sandbox

        # SLOW PATH — reattach or create.
        import modal

        if self.sandbox_id:
            # Wrap both from_id and the poll probe — Modal's from_id raises
            # NotFoundError if the container has been fully torn down (vs.
            # exists-but-dead which surfaces via poll returning non-None).
            try:
                handle = await modal.Sandbox.from_id.aio(self.sandbox_id)
                rc = await handle.poll.aio()
            except Exception as e:
                raise SandboxReattachError(
                    f"reattach to {self.sandbox_id} failed: {e!r}"
                ) from e
            if rc is not None:
                raise SandboxReattachError(
                    f"reattach to {self.sandbox_id} failed: container is dead "
                    f"(exit code {rc}, likely timed out during buffer wait)"
                )
            self._sandbox = handle
        else:
            try:
                image_tag = f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{self.instance_id}:latest"
                image = modal.Image.from_registry(image_tag, force_build=False)
                app = await modal.App.lookup.aio(APP_NAME, create_if_missing=True)
                self._sandbox = await modal.Sandbox.create.aio(
                    image=image,
                    app=app,
                    cpu=CPU,
                    memory=MEMORY_MB,
                    timeout=SANDBOX_TIMEOUT_S,
                    workdir=WORKDIR,
                    block_network=False,
                )
            except Exception as e:
                raise SandboxCreateError(
                    f"create for instance {self.instance_id!r} failed: {e!r}"
                ) from e

        # Update sandbox_id with the live object_id (matters for fresh-create:
        # was None going in, now holds the real ID). Idempotent for reattach.
        self.sandbox_id = self._sandbox.object_id

        # Bump the alive-sandbox counter; warn (don't gate) if it crosses the
        # threshold — likely a leak in close/detach bookkeeping somewhere.
        global _active_count
        _active_count += 1
        if _active_count > _WARN_THRESHOLD:
            logger.warning(
                f"active sandbox count {_active_count} exceeds warn threshold "
                f"{_WARN_THRESHOLD} — check for leaks or raise SLIME_SWEBENCH_WARN_CONCURRENT"
            )
        return self._sandbox

    # ---- shell primitive ---------------------------------------------------

    async def exec(self, command: str, *, timeout: int = PER_CMD_TIMEOUT_S) -> tuple[str, int]:
        import modal
        sb = await self._ensure_sandbox()
        try:
            proc = await sb.exec.aio("bash", "-c", command, timeout=timeout)
        except modal.exception.SandboxTimeoutError:
            return "<sandbox timeout while starting command>", -1
        except modal.exception.NotFoundError as e:
            # Sandbox died between our last poll and now (Modal-side latency
            # can lag the real container state). Treat as mid-rollout death.
            raise SandboxDiedError(
                f"sandbox {self.sandbox_id} died mid-rollout: {e!r}"
            ) from e

        buf: list[str] = []
        remaining = MAX_OUTPUT_BYTES

        async def drain(stream: Any) -> None:
            nonlocal remaining
            try:
                async for chunk in stream:
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode("utf-8", errors="replace")
                    if remaining <= 0:
                        continue  # keep draining so the process can exit
                    take = chunk[:remaining]
                    buf.append(take)
                    remaining -= len(take)
            except UnicodeDecodeError:
                buf.append("\n<truncated: non-utf8 output>\n")

        await drain(proc.stdout)
        await drain(proc.stderr)
        try:
            rc = await proc.wait.aio()
        except modal.exception.SandboxTimeoutError:
            return "<command timed out>", -1

        out = "".join(buf)
        if remaining <= 0:
            out += f"\n<truncated; output exceeded {MAX_OUTPUT_BYTES} bytes>\n"
        return out, int(rc or 0)

    # ---- file I/O via Modal's native filesystem namespace ------------------

    async def read_file(self, path: str) -> str:
        sb = await self._ensure_sandbox()
        try:
            return await sb.filesystem.read_text.aio(path)
        except FileNotFoundError:
            raise
        except Exception as e:
            raise FileNotFoundError(f"read_text({path!r}) failed: {e!r}") from e

    async def write_file(self, path: str, content: str) -> None:
        sb = await self._ensure_sandbox()
        try:
            await sb.filesystem.write_text.aio(content, path)
        except Exception as e:
            raise OSError(f"write_text({path!r}) failed: {e!r}") from e

    # ---- lifecycle ---------------------------------------------------------

    async def close(self) -> None:
        # Terminate the Modal container, then tear down our client connection.
        if self._sandbox is not None:
            try:
                await self._sandbox.terminate.aio()
            except Exception as e:
                logger.warning(f"sandbox {self.sandbox_id} terminate failed: {e!r}")
            try:
                await self._sandbox.detach.aio()
            except Exception as e:
                logger.warning(f"sandbox {self.sandbox_id} detach failed: {e!r}")
            self._sandbox = None
            global _active_count
            _active_count -= 1

    async def detach(self) -> None:
        # Tear down our client connection but leave the Modal container alive
        # so the next resume can reattach via Sandbox.from_id(sandbox_id).
        if self._sandbox is not None:
            try:
                await self._sandbox.detach.aio()
            except Exception as e:
                logger.warning(f"sandbox {self.sandbox_id} detach failed: {e!r}")
            self._sandbox = None
            global _active_count
            _active_count -= 1

