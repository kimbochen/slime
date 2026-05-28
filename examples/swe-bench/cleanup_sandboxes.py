#!/usr/bin/env python3
"""Manually terminate all alive Modal sandboxes in the SWE-bench app.

Run after a training run dies, after a crash, or to free Modal quota when
parked sandboxes have accumulated. Sandboxes left parked (sample aborted but
not yet resumed when training ended) are not auto-cleaned — they self-destruct
after the per-sandbox wall-clock timeout (default 1800s), but that costs $.

Usage:
    uv run --with modal python examples/swe-bench/scripts/cleanup_sandboxes.py

Env:
    SLIME_SWEBENCH_APP   Modal app name to search (default: infx-slime-swebench-sandbox)
    MODAL_CONFIG_PATH    Path to your modal.toml credentials
"""

import asyncio
import os
import sys

import modal


APP_NAME = os.environ.get("SLIME_SWEBENCH_APP", "slime-swebench-sandbox")


async def main() -> int:
    try:
        app = await modal.App.lookup.aio(APP_NAME)
    except modal.exception.NotFoundError:
        print(f"app {APP_NAME!r} not found — nothing to clean up")
        return 0

    # Modal's Sandbox.list takes app_id (string), not app (App object), and
    # returns an AsyncGenerator. Collect them first so we can show a count
    # before terminating.
    sandboxes = []
    async for sb in modal.Sandbox.list.aio(app_id=app.app_id):
        sandboxes.append(sb)
    print(f"found {len(sandboxes)} alive sandboxes in app {APP_NAME!r}")
    if not sandboxes:
        return 0

    failures = 0
    for sb in sandboxes:
        try:
            await sb.terminate.aio()
            print(f"  terminated {sb.object_id}")
        except Exception as e:
            failures += 1
            print(f"  failed     {sb.object_id}: {e!r}")

    print(f"\n{len(sandboxes) - failures}/{len(sandboxes)} terminated")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
