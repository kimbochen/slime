"""SWE-bench reward function for slime's --custom-rm-path hook.

Spawns a FRESH Modal sandbox (not the rollout's), applies the SWE-bench test
patch and the model's proposed patch, runs the instance's eval script, and
returns a binary 0.0/1.0 score based on whether all gold tests passed.

Fresh sandbox vs reuse-rollout: fresh is honest about scoring the patch in a
clean environment. Reusing the rollout's sandbox risks false positives from
debugging commands the model ran during exploration (touching files, setting
env vars, etc.) that aren't part of its proposed fix.

Inputs read from sample:
  sample.metadata["instance_id"]   — drives image lookup
  sample.metadata["test_patch"]    — diff that introduces FAIL_TO_PASS tests
  sample.metadata["model_patch"]   — diff of model's edits (captured by generate.py
                                      before sandbox close; absent on FAILED rollouts)

Score semantics:
  1.0  → /eval.sh exited 0 (all FAIL_TO_PASS + PASS_TO_PASS tests passed)
  0.0  → anything else: failed sample, missing model_patch, patch-apply error,
         /eval.sh non-zero exit, sandbox spawn/eval timeout, etc.

Failure reasons are surfaced via sample.metadata["reward_fail_reason"] so the
metrics hook can aggregate / inspect them.

Wire via:
  --custom-rm-path reward.compute_reward
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from sandbox import (
    SWEBenchSandbox,
    SandboxCreateError,
    SandboxDiedError,
    SandboxReattachError,
)


# Eval can take a while: SWE-bench's official harness caps at 1200s per
# instance. Slightly looser by default; override via env if needed.
EVAL_TIMEOUT_S = int(os.environ.get("SLIME_SWEBENCH_EVAL_TIMEOUT", "1500"))

# Epoch AI images put pytest in a conda env named `testbed`. Activate it
# before running tests. Override via env if a different image layout is used.
CONDA_ENV = os.environ.get("SLIME_SWEBENCH_CONDA_ENV", "testbed")
CONDA_ACTIVATE = os.environ.get(
    "SLIME_SWEBENCH_CONDA_ACTIVATE", "/opt/miniconda3/bin/activate"
)


# Bucket categories that mirror metrics._REWARD_FAIL_BUCKETS — used in the
# per-sample fail-reason JSONL so downstream analysis can join on this string
# without re-doing the substring bucketing.
_REASON_CATEGORY: tuple[tuple[str, str], ...] = (
    ("missing instance_id",     "other"),
    ("no model_patch",          "no_model_patch"),
    ("test_patch apply failed", "test_patch_failed"),
    ("model_patch apply failed","model_patch_failed"),
    ("pytest exit",             "pytest_exit_nonzero"),
    ("no test ids",             "other"),
    ("reward sandbox failed",   "sandbox_error"),
)


def _categorize(reason: str) -> str:
    for substr, cat in _REASON_CATEGORY:
        if substr in reason:
            return cat
    return "other"


def _set_fail_reason(sample: Any, reason: str) -> None:
    """Record fail reason on sample.metadata AND append it to the per-sample
    reward-fail JSONL (if SLIME_REWARD_FAIL_JSONL is set).

    Why: metrics.py only persists *bucketed counts* (e.g. model_patch_failed_frac).
    The raw reason — which tells us *why* a bucket fired (e.g. which file
    rejected a patch) — would otherwise be discarded. The JSONL gives us the
    raw strings for post-hoc analysis without bloating the noisy training log.

    POSIX O_APPEND makes plain open('a').write atomic for writes < PIPE_BUF
    (4KB on Linux), so concurrent rollout workers can share one file without
    locking. We truncate `reason` to 500 chars upstream, so entries stay <1KB.
    """
    sample.metadata["reward_fail_reason"] = reason
    path = os.environ.get("SLIME_REWARD_FAIL_JSONL")
    if not path:
        return
    try:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "instance_id": sample.metadata.get("instance_id", "?"),
            "category": _categorize(reason),
            "reason": reason,
        }
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        # JSONL emission is best-effort; never let it break reward scoring.
        pass


async def compute_reward(args: Any, sample: Any, **kwargs: Any) -> float:
    """Slime --custom-rm-path entry point.

    Returns 1.0 if the model's patch makes all SWE-bench gold tests pass, else
    0.0. Records a fail_reason in sample.metadata on non-trivial failures so
    metrics can distinguish "model wrote a wrong patch" from "patch didn't
    apply" from "infrastructure failure."
    """
    instance_id = sample.metadata.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        _set_fail_reason(sample, "missing instance_id in metadata")
        return 0.0

    # Rollout never produced edits (sample failed, model emitted finish-only,
    # or generate.py couldn't capture the diff). Tests will fail without the
    # fix — skip the eval and return 0.0 to save Modal cost.
    model_patch = sample.metadata.get("model_patch", "")
    if not model_patch.strip():
        _set_fail_reason(sample, f"no model_patch (status={sample.status.name})")
        return 0.0

    test_patch = sample.metadata.get("test_patch", "")

    # Fresh sandbox for honest eval — no rollout-side contamination.
    sandbox = SWEBenchSandbox(instance_id, None)
    try:
        # 1. Apply the test patch (introduces FAIL_TO_PASS tests not in base_commit).
        if test_patch.strip():
            await sandbox.write_file("/tmp/test_patch.diff", test_patch)
            out, rc = await sandbox.exec(
                "cd /testbed && git apply --whitespace=nowarn /tmp/test_patch.diff"
            )
            if rc != 0:
                _set_fail_reason(
                    sample, f"test_patch apply failed (rc={rc}): {out[:500]}"
                )
                return 0.0

        # 2. Apply the model's patch.
        await sandbox.write_file("/tmp/model_patch.diff", model_patch)
        out, rc = await sandbox.exec(
            "cd /testbed && git apply --whitespace=nowarn /tmp/model_patch.diff"
        )
        if rc != 0:
            _set_fail_reason(
                sample, f"model_patch apply failed (rc={rc}): {out[:500]}"
            )
            return 0.0

        # 3. Run FAIL_TO_PASS and PASS_TO_PASS via pytest. Exit 0 iff every
        # specified test passed. (We don't distinguish per-test results here
        # — binary scoring matches the official SWE-bench metric.)
        fail_to_pass = sample.metadata.get("fail_to_pass", []) or []
        pass_to_pass = sample.metadata.get("pass_to_pass", []) or []
        test_ids = list(fail_to_pass) + list(pass_to_pass)
        if not test_ids:
            _set_fail_reason(sample, "no test ids in metadata")
            return 0.0

        # Write test list to a file to avoid shell-argv length limits — some
        # SWE-bench instances have 1000+ tests.
        await sandbox.write_file("/tmp/tests.txt", "\n".join(test_ids))
        eval_cmd = (
            f"source {CONDA_ACTIVATE} {CONDA_ENV} && "
            f"cd /testbed && "
            f"xargs -a /tmp/tests.txt python -m pytest --tb=no -q --no-header"
        )
        out, rc = await sandbox.exec(eval_cmd, timeout=EVAL_TIMEOUT_S)
        if rc == 0:
            return 1.0
        _set_fail_reason(
            sample, f"pytest exit {rc}; tail of output: ...{out[-500:]}"
        )
        return 0.0

    except (SandboxCreateError, SandboxReattachError, SandboxDiedError) as e:
        # Modal infrastructure failure during eval — can't score this sample.
        _set_fail_reason(sample, f"reward sandbox failed: {e!r}")
        return 0.0
    finally:
        # Always tear down — reward sandboxes are one-shot, no resume use case.
        try:
            await sandbox.close()
        except Exception:
            pass
