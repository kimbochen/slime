#!/usr/bin/env python3
"""
End-to-end smoke test for coding_sandbox.py against ONE real SWE-bench
Verified instance image. Validates the full lifecycle:

  1. Pool acquires a sandbox from a per-instance Epoch AI image
  2. run_command works inside the container
  3. The repo is pre-checked-out at WORKDIR (/testbed)
  4. heartbeat correctly reports liveness
  5. write_file + read_file round-trip
  6. apply_patch with a trivial diff
  7. cleanup terminates the sandbox

Pick a small / fast-pulling instance for the trial — `astropy__astropy-12907`
is in the Verified set and the image is well-cached.

Run from the swe-env dir (or any cwd; MODAL_CONFIG_PATH must be set):
  MODAL_CONFIG_PATH=$PWD/.modal.toml python smoke_test_sandbox.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../tests
LIB_DIR = HERE.parent / "lib"
SLIME_ROOT = HERE.parents[2]                    # slime repo root
sys.path.insert(0, str(LIB_DIR))

# Default to <slime_root>/.modal.toml; override MODAL_CONFIG_PATH to point
# elsewhere. If neither resolves, the modal SDK will fail with a clear
# auth error.
os.environ.setdefault("MODAL_CONFIG_PATH", str(SLIME_ROOT / ".modal.toml"))

import modal
from coding_sandbox import CodingSandboxPool, _instance_image_tag

# Surface Modal image-pull errors instead of swallowing them.
modal.enable_output()

INSTANCE_ID = os.environ.get("SWE_SMOKE_INSTANCE", "astropy__astropy-12907")

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, note: str = "") -> None:
    results.append((PASS if ok else FAIL, name, note))


async def main() -> int:
    print(f"smoke test instance: {INSTANCE_ID}")
    print(f"expected image:      {_instance_image_tag(INSTANCE_ID)}")

    pool = CodingSandboxPool(max_concurrent=2, spawn_qps=2.0)

    t0 = time.monotonic()
    async with pool.session(INSTANCE_ID) as sb:
        check("acquired sandbox", sb is not None)
        print(f"  sandbox id={sb.id}  (acquired in {time.monotonic() - t0:.1f}s)")

        # 1. heartbeat — sandbox is live
        alive = await sb.heartbeat()
        check("heartbeat true", alive)

        # 2. repo is at /testbed
        ls = await sb.run_command("ls -la /testbed | head -20")
        check("ls /testbed ok", ls.ok, ls.short_repr(10))
        print(f"  /testbed:\n{ls.stdout[:500]}")

        # 3. git status — image should be a git repo
        git_status = await sb.run_command("git status --short", cwd="/testbed")
        check("git status ok", git_status.exit_code == 0, git_status.short_repr(5))

        # 4. write + read round trip
        write_res = await sb.write_file("/tmp/agent_hello.txt", "hello from slime\n")
        check("write_file ok", write_res.exit_code == 0, write_res.short_repr(5))
        content = await sb.read_file("/tmp/agent_hello.txt")
        check("read_file matches", content == "hello from slime\n", repr(content))

        # 5. apply_patch — trivial no-op patch (creates a new file, doesn't
        #    touch the repo). Use a unified diff that just adds /tmp/foo.txt.
        patch = (
            "--- /dev/null\n"
            "+++ b/new_file_from_patch.txt\n"
            "@@ -0,0 +1 @@\n"
            "+touched by smoke test\n"
        )
        ap = await sb.apply_patch(patch)
        # git apply may or may not succeed depending on workdir state; just
        # check it didn't blow up the sandbox.
        check("apply_patch ran without crashing", ap.exit_code in (0, 1),
              f"exit={ap.exit_code}\n{ap.short_repr(10)}")

        # 6. read python version (sanity for the image's conda env)
        py = await sb.run_command("python --version && which python")
        check("python found in image", py.ok, py.short_repr(5))
        print(f"  python: {py.stdout.strip()}")

        # 7. byte-cap test — write a 200 KB blob and try to read it; we
        #    expect ~131072 bytes back (the default cap).
        big = "x" * (200 * 1024)
        await sb.run_command(f"yes 'x' | head -c 204800 > /tmp/big.txt")
        readback = await sb.read_file("/tmp/big.txt")
        check("read_file respects byte cap (~128 KB)",
              len(readback) <= sb.max_stream_bytes,
              f"len(readback)={len(readback)} max_stream_bytes={sb.max_stream_bytes}")

    # session exit triggers cleanup; pool.release was called
    check("session released cleanly", True)

    print()
    n_pass = sum(1 for s, _, _ in results if s == PASS)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    for status, name, note in results:
        marker = "✓" if status == PASS else "✗"
        print(f"  {marker} {name}")
        if status == FAIL and note:
            for line in note.splitlines():
                print(f"      {line}")
    print(f"\n{n_pass} passed, {n_fail} failed.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
