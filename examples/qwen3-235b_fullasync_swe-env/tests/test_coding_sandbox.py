"""
test_coding_sandbox.py
======================

Hermetic unit tests for coding_sandbox.py. Stubs the `modal` module so the
tests run without Modal auth / network access. Covers:

  • _instance_image_tag — registry/tag formatting
  • CommandResult.ok — exit_code + timeout logic
  • CommandResult.short_repr — long-output truncation
  • _read_stream_capped — byte cap, str/bytes chunks, UnicodeDecodeError
  • CodingSandboxPool — concurrency semaphore + spawn rate limit + active set

Run:
  uv run python examples/qwen3-235b_fullasync_swe-env/tests/test_coding_sandbox.py
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import time
import types
from pathlib import Path


# ---------------------------------------------------------------------------
# Stub `modal` before importing coding_sandbox
# ---------------------------------------------------------------------------


class _FakeModalSandbox:
    _next_id = 0

    def __init__(self, **kwargs):
        type(self)._next_id += 1
        self.object_id = f"sb-fake-{type(self)._next_id}"
        self.kwargs = kwargs
        self._terminated = False
        # Pluggable per-instance exec behavior — tests can override.
        self.exec_responses: dict[str, dict] = {}

    async def _aterminate(self):
        self._terminated = True

    @property
    def terminate(self):
        async def _t():
            self._terminated = True
        return type("_T", (), {"aio": staticmethod(_t)})()

    @property
    def exec(self):
        # Used by CodingSandbox.run_command. Not exercised in these tests
        # directly — the run_command path needs an integration test.
        async def _exec(*args, **kw):
            raise RuntimeError("fake exec not configured")
        return type("_E", (), {"aio": staticmethod(_exec)})()


class _FakeCreateNS:
    """modal.Sandbox.create — async classmethod-like wrapper."""
    @staticmethod
    async def aio(image, app, cpu, memory, timeout, workdir, block_network):
        return _FakeModalSandbox(
            image=image, app=app, cpu=cpu, memory=memory,
            timeout=timeout, workdir=workdir, block_network=block_network,
        )


class _FakeSandboxCls:
    create = _FakeCreateNS()


class _FakeAppLookupNS:
    @staticmethod
    async def aio(app_name, create_if_missing=True):
        return types.SimpleNamespace(name=app_name)


class _FakeAppCls:
    lookup = _FakeAppLookupNS()


class _FakeImageFromRegistryNS:
    @staticmethod
    def __call__(tag, force_build=False):
        return types.SimpleNamespace(tag=tag, force_build=force_build)


class _FakeImageCls:
    @staticmethod
    def from_registry(tag, force_build=False):
        return types.SimpleNamespace(tag=tag, force_build=force_build)


class _FakeException:
    class SandboxTimeoutError(Exception):
        pass


def _install_modal_stub():
    """Install a minimal `modal` module stub in sys.modules."""
    if "modal" in sys.modules:
        del sys.modules["modal"]
    modal_mod = types.ModuleType("modal")
    modal_mod.Sandbox = _FakeSandboxCls
    modal_mod.App = _FakeAppCls
    modal_mod.Image = _FakeImageCls
    modal_mod.exception = _FakeException
    sys.modules["modal"] = modal_mod
    return modal_mod


def _load_sandbox_module():
    """Install stub, then (re)load coding_sandbox so it picks up the stub."""
    _install_modal_stub()
    # Drop cached coding_sandbox so it re-imports modal-the-stub.
    for k in list(sys.modules):
        if k == "coding_sandbox":
            del sys.modules[k]
    # tests/ lives next to lib/; resolve relative to this file so the test
    # doesn't bake in an absolute path.
    lib_dir = str((Path(__file__).resolve().parent.parent / "lib"))
    if lib_dir in sys.path:
        sys.path.remove(lib_dir)
    sys.path.insert(0, lib_dir)
    return importlib.import_module("coding_sandbox")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_instance_image_tag_default_registry():
    cs = _load_sandbox_module()
    tag = cs._instance_image_tag("django__django-11019")
    # Default registry from CONFIG (env-overridable). Just check shape.
    assert tag.endswith("/swe-bench.eval.x86_64.django__django-11019:latest"), tag


def test_instance_image_tag_custom_registry():
    cs = _load_sandbox_module()
    tag = cs._instance_image_tag(
        "scikit-learn__scikit-learn-10297",
        registry="registry.example.com/me",
    )
    assert tag == (
        "registry.example.com/me/swe-bench.eval.x86_64."
        "scikit-learn__scikit-learn-10297:latest"
    )


def test_command_result_ok_happy():
    cs = _load_sandbox_module()
    r = cs.CommandResult(
        stdout="ok\n", stderr="", exit_code=0, timed_out=False, duration_s=0.1
    )
    assert r.ok is True


def test_command_result_ok_nonzero_exit():
    cs = _load_sandbox_module()
    r = cs.CommandResult(stdout="", stderr="oops", exit_code=1,
                          timed_out=False, duration_s=0.0)
    assert r.ok is False


def test_command_result_ok_timeout_overrides_exit_zero():
    """A command can return exit_code=0 but timed_out=True (e.g., the sandbox
    killed it after the wall clock); ok must be False."""
    cs = _load_sandbox_module()
    r = cs.CommandResult(stdout="", stderr="", exit_code=0,
                          timed_out=True, duration_s=120.0)
    assert r.ok is False


def test_command_result_short_repr_no_truncation_for_short():
    cs = _load_sandbox_module()
    r = cs.CommandResult(
        stdout="a\nb\nc", stderr="", exit_code=0, timed_out=False, duration_s=0.1
    )
    out = r.short_repr(max_lines=20)
    assert "a\nb\nc" in out
    assert "elided" not in out
    assert "exit_code: 0" in out


def test_command_result_short_repr_truncates_long():
    cs = _load_sandbox_module()
    lines = "\n".join(f"line{i}" for i in range(100))
    r = cs.CommandResult(stdout=lines, stderr="", exit_code=0,
                          timed_out=False, duration_s=0.1)
    out = r.short_repr(max_lines=10)
    # First half + sentinel + last half
    assert "line0" in out
    assert "line99" in out
    assert "elided" in out
    # Should NOT contain the middle line
    assert "line50" not in out


def test_command_result_short_repr_timeout_banner():
    cs = _load_sandbox_module()
    r = cs.CommandResult(stdout="", stderr="", exit_code=-1,
                          timed_out=True, duration_s=120.0)
    out = r.short_repr()
    assert "TIMEOUT" in out
    assert "120.0s" in out


def test_command_result_short_repr_omits_empty_streams():
    cs = _load_sandbox_module()
    r = cs.CommandResult(stdout="", stderr="", exit_code=0,
                          timed_out=False, duration_s=0.0)
    out = r.short_repr()
    assert "stdout" not in out
    assert "stderr" not in out


# --- _read_stream_capped ----------------------------------------------------


async def _aiter(items):
    for x in items:
        yield x


def test_read_stream_capped_under_cap():
    cs = _load_sandbox_module()
    stream = _aiter([b"hello ", b"world"])
    out = asyncio.run(cs._read_stream_capped(stream, max_bytes=1024))
    assert out == "hello world"


def test_read_stream_capped_at_cap():
    cs = _load_sandbox_module()
    stream = _aiter([b"a" * 100, b"b" * 100])
    out = asyncio.run(cs._read_stream_capped(stream, max_bytes=150))
    # Should take 100 a's + 50 b's, then drain (silently) the rest.
    assert out == ("a" * 100) + ("b" * 50)


def test_read_stream_capped_zero_cap():
    cs = _load_sandbox_module()
    stream = _aiter([b"abc", b"def"])
    out = asyncio.run(cs._read_stream_capped(stream, max_bytes=0))
    assert out == ""


def test_read_stream_capped_str_chunks_pass_through():
    """Modal's text-mode stream yields str chunks. Verify they get re-encoded."""
    cs = _load_sandbox_module()
    stream = _aiter(["hello ", "world"])
    out = asyncio.run(cs._read_stream_capped(stream, max_bytes=1024))
    assert out == "hello world"


def test_read_stream_capped_unicode_error_appended_sentinel():
    """If the underlying stream raises UnicodeDecodeError mid-iteration
    (modal sometimes does on non-utf8 output), the helper must catch it,
    return what we have, and append a sentinel."""
    cs = _load_sandbox_module()

    async def _bad_stream():
        yield b"good prefix"
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad byte")

    out = asyncio.run(cs._read_stream_capped(_bad_stream(), max_bytes=1024))
    assert "good prefix" in out
    assert "truncated" in out
    assert "non-utf8" in out


# --- CodingSandboxPool ------------------------------------------------------


def test_pool_concurrency_cap():
    """Verify the pool's semaphore caps concurrent acquires. With max=2 and
    3 acquirers, only 2 should be in flight at once."""
    cs = _load_sandbox_module()
    pool = cs.CodingSandboxPool(
        app_name="test", max_concurrent=2, spawn_qps=1000.0
    )

    inflight = 0
    peak = 0
    lock = asyncio.Lock()
    enter_event = asyncio.Event()  # set when ANY worker enters
    release_event = asyncio.Event()  # held — we release sandboxes manually

    async def worker():
        nonlocal inflight, peak
        sb = await pool.acquire("django__django-11019")
        async with lock:
            inflight += 1
            peak = max(peak, inflight)
        enter_event.set()
        await release_event.wait()
        async with lock:
            inflight -= 1
        # Stub out cleanup since the fake sandbox has no real cleanup.
        sb.cleanup = lambda: _completed_future()
        await pool.release(sb)

    async def _completed_future():
        return None

    async def main():
        tasks = [asyncio.create_task(worker()) for _ in range(3)]
        # Give workers a chance to race for the semaphore.
        await asyncio.sleep(0.05)
        release_event.set()
        await asyncio.gather(*tasks)

    asyncio.run(main())
    assert peak == 2, f"expected peak concurrency 2, got {peak}"


def test_pool_releases_semaphore_on_create_failure():
    """If CodingSandbox.create raises, the semaphore must be released so the
    pool doesn't get permanently jammed."""
    cs = _load_sandbox_module()

    async def _failing_create(instance_id, **kw):
        raise RuntimeError("simulated modal failure")

    pool = cs.CodingSandboxPool(
        app_name="test", max_concurrent=1, spawn_qps=1000.0
    )
    # Replace create to force a failure.
    cs.CodingSandbox.create = _failing_create

    async def main():
        for _ in range(3):
            try:
                await pool.acquire("django__django-11019")
            except RuntimeError:
                pass
        # If the semaphore leaked, this would deadlock; we time it out.

    asyncio.run(asyncio.wait_for(main(), timeout=2.0))


def test_pool_active_set_tracks_sandbox():
    """An acquired sandbox lands in the pool's _active dict; release pops it."""
    cs = _load_sandbox_module()

    # Restore real create (test above may have monkeypatched it away).
    cs = _load_sandbox_module()

    pool = cs.CodingSandboxPool(
        app_name="test", max_concurrent=4, spawn_qps=1000.0
    )

    async def main():
        sb = await pool.acquire("django__django-11019")
        assert sb.id in pool._active
        # Fake the cleanup since our stub sandbox has no real one.
        async def _noop():
            return None
        sb.cleanup = _noop
        await pool.release(sb)
        assert sb.id not in pool._active

    asyncio.run(main())


def test_pool_spawn_rate_limiter_enforces_min_interval():
    """With spawn_qps=10, two consecutive spawns must be ≥ 0.1s apart."""
    cs = _load_sandbox_module()
    pool = cs.CodingSandboxPool(
        app_name="test", max_concurrent=8, spawn_qps=10.0
    )

    async def main():
        async def _noop():
            return None
        t0 = time.monotonic()
        sb1 = await pool.acquire("a")
        sb1.cleanup = _noop
        sb2 = await pool.acquire("b")
        sb2.cleanup = _noop
        dt = time.monotonic() - t0
        # With 2 acquires at qps=10, min wallclock is ~0.1s for the second
        # spawn (the first goes through immediately).
        assert dt >= 0.09, f"rate limiter should enforce ≥0.09s, got {dt:.3f}"
        await pool.release(sb1)
        await pool.release(sb2)

    asyncio.run(main())


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


TESTS = [
    test_instance_image_tag_default_registry,
    test_instance_image_tag_custom_registry,
    test_command_result_ok_happy,
    test_command_result_ok_nonzero_exit,
    test_command_result_ok_timeout_overrides_exit_zero,
    test_command_result_short_repr_no_truncation_for_short,
    test_command_result_short_repr_truncates_long,
    test_command_result_short_repr_timeout_banner,
    test_command_result_short_repr_omits_empty_streams,
    test_read_stream_capped_under_cap,
    test_read_stream_capped_at_cap,
    test_read_stream_capped_zero_cap,
    test_read_stream_capped_str_chunks_pass_through,
    test_read_stream_capped_unicode_error_appended_sentinel,
    test_pool_concurrency_cap,
    test_pool_releases_semaphore_on_create_failure,
    test_pool_active_set_tracks_sandbox,
    test_pool_spawn_rate_limiter_enforces_min_interval,
]


def main():
    n_pass = n_fail = 0
    fails = []
    for t in TESTS:
        name = t.__name__
        try:
            t()
            print(f"  PASS  {name}")
            n_pass += 1
        except Exception as e:
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            fails.append(f"{name}: {e}")
            n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed")
    if fails:
        print("\nFailures:")
        for f in fails:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
