"""
Smoke test for modal_tool_sandbox.py.

CPU-only — does not require any GPU or slime training. Validates:
  1. Pool of N Modal sandboxes can be spun up.
  2. Concurrent execute_code() calls succeed and return correct output.
  3. The pool releases healthy sandboxes back into the queue.
  4. Sandbox network is actually blocked (block_network=True).
  5. Per-exec timeout fires when code runs too long.

Run:
  MODAL_CONFIG_PATH=/home/sa-shared/kimbo/slime/.modal.toml \
    SLIME_MODAL_POOL_SIZE=4 \
    python examples/glm45-air_full-async_retool-modal/test_modal_sandbox.py
"""

import asyncio
import os
import sys
import time

# Make the example dir importable when run as a script from repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force a small pool for the smoke test.
os.environ.setdefault("SLIME_MODAL_POOL_SIZE", "4")
os.environ.setdefault("SLIME_MODAL_PER_EXEC_TIMEOUT", "10")
os.environ.setdefault("SLIME_MODAL_APP", "infx-slime-retool-sandbox-test")

import modal_tool_sandbox as mts  # noqa: E402


async def test_basic_math():
    print("[1/4] basic math via tool_registry.execute_tool ...")
    out = await mts.tool_registry.execute_tool(
        "code_interpreter", {"code": "print(2 ** 100)"}
    )
    expected = "1267650600228229401496703205376"
    assert expected in out, f"expected {expected} in output, got: {out!r}"
    print("    OK")


async def test_concurrency():
    print("[2/4] running 8 concurrent execs over a 4-sandbox pool ...")
    started = time.time()
    codes = [f"print({i} ** 10)" for i in range(8)]
    outs = await asyncio.gather(
        *[mts.tool_registry.execute_tool("code_interpreter", {"code": c}) for c in codes]
    )
    for i, out in enumerate(outs):
        assert str(i ** 10) in out, f"sample {i} bad: {out!r}"
    print(f"    OK (8 calls in {time.time() - started:.1f}s)")


async def test_network_blocked():
    print("[3/4] confirming network is blocked inside sandbox ...")
    code = (
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://example.com', timeout=3).read(10)\n"
        "    print('NETWORK_OPEN')\n"
        "except Exception as e:\n"
        "    print('NETWORK_BLOCKED:', type(e).__name__)\n"
    )
    out = await mts.tool_registry.execute_tool("code_interpreter", {"code": code})
    assert "NETWORK_BLOCKED" in out, f"network not blocked: {out!r}"
    print("    OK")


async def test_per_exec_timeout():
    print("[4/4] confirming per-exec timeout fires ...")
    # SLIME_MODAL_PER_EXEC_TIMEOUT=10 above.
    code = "import time; time.sleep(15); print('SHOULD_NOT_PRINT')"
    out = await mts.tool_registry.execute_tool("code_interpreter", {"code": code})
    assert "timed out" in out.lower(), f"expected timeout, got: {out!r}"
    print("    OK")


async def main():
    if not os.environ.get("MODAL_CONFIG_PATH"):
        print("WARNING: MODAL_CONFIG_PATH unset — Modal SDK will fall back to ~/.modal.toml")

    await test_basic_math()
    await test_concurrency()
    await test_network_blocked()
    await test_per_exec_timeout()
    print("\nAll Modal sandbox smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
