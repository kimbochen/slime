#!/usr/bin/env python3
"""Live tests for examples/swe-bench/sandbox.py against real Modal sandboxes.

Tests actually spawn, exec on, read/write files in, and terminate Modal
containers. Each create costs Modal $ and seconds; tests are written to spawn
as few sandboxes as possible.

Skip with SKIP_LIVE_SANDBOX=1 (POSIX skip exit code 77).
Skip automatically if MODAL_CONFIG_PATH is unset and ~/.modal.toml is absent.

Default instance is `astropy__astropy-12907` (small / fast image pull per the
in-repo smoke tests). Override with SLIME_SWEBENCH_TEST_INSTANCE.

Run from anywhere:
    MODAL_CONFIG_PATH=/path/to/.modal.toml \\
      uv run --with modal python examples/swe-bench/tests/test_sandbox.py
"""

import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE_DIR = HERE.parent
sys.path.insert(0, str(EXAMPLE_DIR))


# ---------------------------------------------------------------------------
# Skip gates
# ---------------------------------------------------------------------------


if os.environ.get("SKIP_LIVE_SANDBOX") == "1":
    print("SKIP: SKIP_LIVE_SANDBOX=1")
    sys.exit(77)

if not os.environ.get("MODAL_CONFIG_PATH") and not Path.home().joinpath(".modal.toml").exists():
    print(
        "SKIP: no Modal credentials found. Set MODAL_CONFIG_PATH or place "
        "credentials at ~/.modal.toml."
    )
    sys.exit(77)


from sandbox import (  # noqa: E402
    SWEBenchSandbox,
    SandboxCreateError,
    SandboxDiedError,
    SandboxReattachError,
)


TEST_INSTANCE = os.environ.get("SLIME_SWEBENCH_TEST_INSTANCE", "astropy__astropy-12907")


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Lazy-creation invariants (no Modal API calls)
# ---------------------------------------------------------------------------


def test_construction_does_no_io() -> None:
    """Constructor should be sync and side-effect-free. sandbox_id reflects
    the constructor hint when None / not-yet-activated."""
    sb = SWEBenchSandbox(TEST_INSTANCE, None)
    assert sb._sandbox is None
    assert sb.sandbox_id is None
    sb2 = SWEBenchSandbox(TEST_INSTANCE, "sb-fake-hint")
    assert sb2._sandbox is None
    assert sb2.sandbox_id == "sb-fake-hint"


# ---------------------------------------------------------------------------
# Fresh-create + exec + close
# ---------------------------------------------------------------------------


def test_create_exec_close_lifecycle() -> None:
    async def _go():
        sb = SWEBenchSandbox(TEST_INSTANCE, None)
        try:
            out, rc = await sb.exec("echo hello")
            assert rc == 0, f"echo should succeed, got rc={rc}, out={out!r}"
            assert "hello" in out
            assert sb.sandbox_id is not None
            assert sb.sandbox_id.startswith("sb-"), sb.sandbox_id
        finally:
            await sb.close()
        assert sb._sandbox is None, "close() should null the handle"
    _run(_go())


def test_exec_nonzero_exit_returns_rc() -> None:
    async def _go():
        sb = SWEBenchSandbox(TEST_INSTANCE, None)
        try:
            out, rc = await sb.exec("false")
            assert rc != 0, f"`false` should exit nonzero, got rc={rc}"
        finally:
            await sb.close()
    _run(_go())


# ---------------------------------------------------------------------------
# read_file / write_file via Modal's native filesystem APIs
# ---------------------------------------------------------------------------


def test_write_read_roundtrip() -> None:
    async def _go():
        sb = SWEBenchSandbox(TEST_INSTANCE, None)
        try:
            await sb.write_file("/tmp/slime-test.txt", "hello slime")
            content = await sb.read_file("/tmp/slime-test.txt")
            assert content == "hello slime", f"got {content!r}"
        finally:
            await sb.close()
    _run(_go())


def test_write_creates_parent_dirs() -> None:
    """Modal's filesystem.write_text creates parents automatically."""
    async def _go():
        sb = SWEBenchSandbox(TEST_INSTANCE, None)
        try:
            await sb.write_file("/tmp/slime-deep/nested/file.txt", "ok")
            content = await sb.read_file("/tmp/slime-deep/nested/file.txt")
            assert content == "ok"
        finally:
            await sb.close()
    _run(_go())


def test_read_missing_file_raises_file_not_found() -> None:
    async def _go():
        sb = SWEBenchSandbox(TEST_INSTANCE, None)
        try:
            try:
                await sb.read_file("/no/such/path-xyz")
            except FileNotFoundError:
                return
            raise AssertionError("expected FileNotFoundError")
        finally:
            await sb.close()
    _run(_go())


# ---------------------------------------------------------------------------
# Detach + reattach: state should persist across the boundary
# ---------------------------------------------------------------------------


def test_detach_leaves_container_alive_reattach_sees_state() -> None:
    """Detach must NOT terminate the container; the next SWEBenchSandbox
    constructed with the same sandbox_id should reattach and see prior state."""
    async def _go():
        sb1 = SWEBenchSandbox(TEST_INSTANCE, None)
        original_id: str | None = None
        try:
            await sb1.write_file("/tmp/slime-detach-witness", "marker-value")
            original_id = sb1.sandbox_id
            assert original_id is not None
            await sb1.detach()
            assert sb1._sandbox is None, "detach should null Python handle"

            sb2 = SWEBenchSandbox(TEST_INSTANCE, original_id)
            try:
                content = await sb2.read_file("/tmp/slime-detach-witness")
                assert content == "marker-value", f"state lost across detach+reattach: got {content!r}"
                assert sb2.sandbox_id == original_id
            finally:
                await sb2.close()
        finally:
            # Belt-and-suspenders: if sb1 still holds the handle (test failed
            # before detach), close it. If sb2 closed, sb1's container is gone too.
            if sb1._sandbox is not None:
                await sb1.close()
    _run(_go())


# ---------------------------------------------------------------------------
# Error paths: create failure (bogus instance) + reattach failure (dead id)
# ---------------------------------------------------------------------------


def test_create_with_bogus_instance_raises_create_error() -> None:
    """Image tag for a non-existent instance should fail with SandboxCreateError."""
    async def _go():
        sb = SWEBenchSandbox("bogus__instance-does-not-exist-zzz", None)
        try:
            try:
                await sb.exec("true")
            except SandboxCreateError:
                return
            raise AssertionError("expected SandboxCreateError")
        finally:
            if sb._sandbox is not None:
                await sb.close()
    _run(_go())


def test_reattach_to_dead_sandbox_raises_lifecycle_error() -> None:
    """After close(), the container is terminated. Trying to use it should
    fail with a sandbox lifecycle error. Either:
      * SandboxReattachError — from_id or poll() catches the dead state
      * SandboxDiedError — poll() lies (Modal latency) and the first real
        operation (exec) discovers the dead container

    Both are valid signals that the sandbox is gone. The test accepts either.
    """
    async def _go():
        sb1 = SWEBenchSandbox(TEST_INSTANCE, None)
        await sb1.exec("echo init")
        dead_id = sb1.sandbox_id
        assert dead_id is not None
        await sb1.close()                        # actually terminates the container

        sb2 = SWEBenchSandbox(TEST_INSTANCE, dead_id)
        try:
            try:
                await sb2.exec("echo")
            except (SandboxReattachError, SandboxDiedError):
                return
            raise AssertionError("expected SandboxReattachError or SandboxDiedError")
        finally:
            if sb2._sandbox is not None:
                await sb2.close()
    _run(_go())


def main() -> int:
    test_fns = [
        test_construction_does_no_io,
        test_create_exec_close_lifecycle,
        test_exec_nonzero_exit_returns_rc,
        test_write_read_roundtrip,
        test_write_creates_parent_dirs,
        test_read_missing_file_raises_file_not_found,
        test_detach_leaves_container_alive_reattach_sees_state,
        test_create_with_bogus_instance_raises_create_error,
        test_reattach_to_dead_sandbox_raises_lifecycle_error,
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
