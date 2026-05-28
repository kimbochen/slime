#!/usr/bin/env python3
"""Tests for examples/swe-bench/reward.py.

Two flavors:
  * Cheap unit tests (no Modal) for the early-return failure paths
    (missing instance_id, missing model_patch).
  * Live tests gated on Modal credentials that actually spawn a sandbox,
    apply patches, and run /eval.sh. Uses the dataset's own gold patch as the
    "model patch" so we can assert score == 1.0 for a correct fix.

Skip live tests with SKIP_LIVE_SANDBOX=1. The cheap tests always run.

Run from anywhere:
    MODAL_CONFIG_PATH=/path/to/.modal.toml \\
      uv run --with modal --with datasets python examples/swe-bench/tests/test_reward.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../tests
EXAMPLE_DIR = HERE.parent                       # .../swe-bench
SLIME_ROOT = EXAMPLE_DIR.parents[1]             # repo root (for `import slime`)
sys.path.insert(0, str(EXAMPLE_DIR))
sys.path.insert(0, str(SLIME_ROOT))


# ---------------------------------------------------------------------------
# Always-on cheap tests (no Modal)
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def _make_sample(metadata: dict, status_name: str = "COMPLETED"):
    """Construct a Sample with the given metadata and status."""
    from slime.utils.types import Sample
    s = Sample()
    s.metadata = dict(metadata)
    s.status = getattr(Sample.Status, status_name)
    return s


def test_no_instance_id_returns_zero_and_sets_reason() -> None:
    import reward
    sample = _make_sample({})
    score = _run(reward.compute_reward(None, sample))
    assert score == 0.0
    assert "instance_id" in sample.metadata.get("reward_fail_reason", "")


def test_empty_model_patch_returns_zero_with_status_in_reason() -> None:
    import reward
    sample = _make_sample(
        {"instance_id": "django__django-12345", "model_patch": ""},
        status_name="COMPLETED",
    )
    score = _run(reward.compute_reward(None, sample))
    assert score == 0.0
    reason = sample.metadata.get("reward_fail_reason", "")
    assert "no model_patch" in reason
    assert "COMPLETED" in reason


def test_missing_model_patch_key_returns_zero() -> None:
    import reward
    sample = _make_sample(
        {"instance_id": "django__django-12345"},   # no model_patch key
        status_name="TRUNCATED",
    )
    score = _run(reward.compute_reward(None, sample))
    assert score == 0.0
    assert "no model_patch" in sample.metadata.get("reward_fail_reason", "")


# ---------------------------------------------------------------------------
# Live tests (gated)
# ---------------------------------------------------------------------------


_LIVE_SKIP_REASON = None
if os.environ.get("SKIP_LIVE_SANDBOX") == "1":
    _LIVE_SKIP_REASON = "SKIP_LIVE_SANDBOX=1"
elif (not os.environ.get("MODAL_CONFIG_PATH")
      and not Path.home().joinpath(".modal.toml").exists()):
    _LIVE_SKIP_REASON = "no Modal credentials"


TEST_INSTANCE = os.environ.get("SLIME_SWEBENCH_TEST_INSTANCE", "astropy__astropy-12907")


def _load_instance_from_dataset(instance_id: str) -> dict:
    """Pull the named instance from SWE-bench Verified and return a metadata
    dict shaped like what gen_prompt_data.py produces."""
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    row = next((r for r in ds if r["instance_id"] == instance_id), None)
    if row is None:
        raise RuntimeError(f"instance {instance_id!r} not found in SWE-bench Verified")
    return {
        "instance_id": row["instance_id"],
        "test_patch": row["test_patch"],
        "fail_to_pass": json.loads(row["FAIL_TO_PASS"]),
        "pass_to_pass": json.loads(row["PASS_TO_PASS"]),
        "gold_patch": row["patch"],   # we'll use this as the "model_patch"
    }


def test_gold_patch_scores_one_live() -> None:
    """Apply the dataset's own gold patch as if it were the model's patch.
    A correct fix should make ALL tests pass → reward = 1.0."""
    if _LIVE_SKIP_REASON:
        print(f"SKIP: {_LIVE_SKIP_REASON}")
        return
    import reward
    md = _load_instance_from_dataset(TEST_INSTANCE)
    sample = _make_sample({
        "instance_id": md["instance_id"],
        "test_patch": md["test_patch"],
        "fail_to_pass": md["fail_to_pass"],
        "pass_to_pass": md["pass_to_pass"],
        "model_patch": md["gold_patch"],
    })
    score = _run(reward.compute_reward(None, sample))
    fail_reason = sample.metadata.get("reward_fail_reason", "<none>")
    assert score == 1.0, (
        f"gold patch should score 1.0, got {score}; "
        f"reward_fail_reason: {fail_reason}"
    )


def test_invalid_patch_scores_zero_live() -> None:
    """An intentionally malformed unified diff should fail to apply → 0.0
    with a reward_fail_reason mentioning the apply failure."""
    if _LIVE_SKIP_REASON:
        print(f"SKIP: {_LIVE_SKIP_REASON}")
        return
    import reward
    md = _load_instance_from_dataset(TEST_INSTANCE)
    sample = _make_sample({
        "instance_id": md["instance_id"],
        "test_patch": md["test_patch"],
        "fail_to_pass": md["fail_to_pass"],
        "pass_to_pass": md["pass_to_pass"],
        "model_patch": "this is not a valid unified diff, just garbage\n",
    })
    score = _run(reward.compute_reward(None, sample))
    assert score == 0.0
    reason = sample.metadata.get("reward_fail_reason", "")
    assert "model_patch apply failed" in reason, reason


def test_empty_patch_after_test_patch_scores_zero_live() -> None:
    """Empty model_patch = early-return path, doesn't even touch Modal.
    Verify the early return triggers BEFORE the live eval would have run.
    """
    # This is actually a cheap test in disguise — early return saves the
    # Modal cost. We include it in the live group as a sanity check that
    # the cost-saving early return works when wired against real metadata.
    if _LIVE_SKIP_REASON:
        print(f"SKIP: {_LIVE_SKIP_REASON}")
        return
    import reward
    md = _load_instance_from_dataset(TEST_INSTANCE)
    sample = _make_sample({
        "instance_id": md["instance_id"],
        "test_patch": md["test_patch"],
        "model_patch": "",         # no edits
    })
    score = _run(reward.compute_reward(None, sample))
    assert score == 0.0
    assert "no model_patch" in sample.metadata.get("reward_fail_reason", "")


def main() -> int:
    test_fns = [
        # cheap
        test_no_instance_id_returns_zero_and_sets_reason,
        test_empty_model_patch_returns_zero_with_status_in_reason,
        test_missing_model_patch_key_returns_zero,
        # live
        test_gold_patch_scores_one_live,
        test_invalid_patch_scores_zero_live,
        test_empty_patch_after_test_patch_scores_zero_live,
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
