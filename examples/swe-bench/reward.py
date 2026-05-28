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

import os
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


async def compute_reward(args: Any, sample: Any, **kwargs: Any) -> float:
    """Slime --custom-rm-path entry point.

    Returns 1.0 if the model's patch makes all SWE-bench gold tests pass, else
    0.0. Records a fail_reason in sample.metadata on non-trivial failures so
    metrics can distinguish "model wrote a wrong patch" from "patch didn't
    apply" from "infrastructure failure."
    """
    instance_id = sample.metadata.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        sample.metadata["reward_fail_reason"] = "missing instance_id in metadata"
        return 0.0

    # Rollout never produced edits (sample failed, model emitted finish-only,
    # or generate.py couldn't capture the diff). Tests will fail without the
    # fix — skip the eval and return 0.0 to save Modal cost.
    model_patch = sample.metadata.get("model_patch", "")
    if not model_patch.strip():
        sample.metadata["reward_fail_reason"] = (
            f"no model_patch (status={sample.status.name})"
        )
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
                sample.metadata["reward_fail_reason"] = (
                    f"test_patch apply failed (rc={rc}): {out[:500]}"
                )
                return 0.0

        # 2. Apply the model's patch.
        await sandbox.write_file("/tmp/model_patch.diff", model_patch)
        out, rc = await sandbox.exec(
            "cd /testbed && git apply --whitespace=nowarn /tmp/model_patch.diff"
        )
        if rc != 0:
            sample.metadata["reward_fail_reason"] = (
                f"model_patch apply failed (rc={rc}): {out[:500]}"
            )
            return 0.0

        # 3. Run FAIL_TO_PASS and PASS_TO_PASS via pytest. Exit 0 iff every
        # specified test passed. (We don't distinguish per-test results here
        # — binary scoring matches the official SWE-bench metric.)
        fail_to_pass = sample.metadata.get("fail_to_pass", []) or []
        pass_to_pass = sample.metadata.get("pass_to_pass", []) or []
        test_ids = list(fail_to_pass) + list(pass_to_pass)
        if not test_ids:
            sample.metadata["reward_fail_reason"] = "no test ids in metadata"
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
        sample.metadata["reward_fail_reason"] = (
            f"pytest exit {rc}; tail of output: ...{out[-500:]}"
        )
        return 0.0

    except (SandboxCreateError, SandboxReattachError, SandboxDiedError) as e:
        # Modal infrastructure failure during eval — can't score this sample.
        sample.metadata["reward_fail_reason"] = f"reward sandbox failed: {e!r}"
        return 0.0
    finally:
        # Always tear down — reward sandboxes are one-shot, no resume use case.
        try:
            await sandbox.close()
        except Exception:
            pass
