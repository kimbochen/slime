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


def test_fail_reason_jsonl_appends_when_env_set(tmp_path=None) -> None:
    """When SLIME_REWARD_FAIL_JSONL is set, every fail reason is appended as
    one JSON record with {ts, instance_id, category, reason}."""
    import tempfile
    import reward

    with tempfile.TemporaryDirectory() as td:
        jsonl_path = os.path.join(td, "reward-fails.jsonl")
        os.environ["SLIME_REWARD_FAIL_JSONL"] = jsonl_path
        try:
            # Trigger two distinct fail paths.
            s1 = _make_sample({})                                      # missing instance_id
            _run(reward.compute_reward(None, s1))
            s2 = _make_sample(
                {"instance_id": "django__django-12345", "model_patch": ""},
                status_name="COMPLETED",
            )
            _run(reward.compute_reward(None, s2))

            assert os.path.exists(jsonl_path), "JSONL was not written"
            lines = [l for l in open(jsonl_path).read().splitlines() if l.strip()]
            assert len(lines) == 2, f"expected 2 records, got {len(lines)}: {lines}"

            r1 = json.loads(lines[0])
            assert r1["category"] == "other"          # "missing instance_id" → "other"
            assert "instance_id" in r1["reason"]
            assert r1["instance_id"] == "?"           # no instance_id available
            assert "ts" in r1

            r2 = json.loads(lines[1])
            assert r2["category"] == "no_model_patch"
            assert "no model_patch" in r2["reason"]
            assert r2["instance_id"] == "django__django-12345"
        finally:
            os.environ.pop("SLIME_REWARD_FAIL_JSONL", None)


def test_fail_reason_jsonl_noop_when_env_unset() -> None:
    """When SLIME_REWARD_FAIL_JSONL is unset, reward.py writes nothing.
    Guarantees backward compat for callers who don't opt in."""
    import reward
    # Make sure env var is not present.
    os.environ.pop("SLIME_REWARD_FAIL_JSONL", None)
    sample = _make_sample({})
    score = _run(reward.compute_reward(None, sample))
    assert score == 0.0
    # Metadata mutation still happens — only the JSONL side effect is gated.
    assert "instance_id" in sample.metadata.get("reward_fail_reason", "")


def test_django_classes_extracts_class_paths_via_rpartition() -> None:
    """SWE-bench django test IDs are "<name> (module.Class)" where <name> is
    EITHER the method name OR the method's docstring first line. Docstrings
    can themselves contain " (" (e.g. "setUp() is reraised"), so we MUST use
    rpartition on the LAST " (" to find the class. We then drop the name
    entirely and just return the unique class paths."""
    import reward
    ids = [
        # Plain method-name format.
        "test_override_file_upload_permissions (test_utils.tests.OverrideSettingsTests)",
        "test_callable_path (model_fields.test_filepathfield.FilePathFieldTests)",
        # Docstring-as-name with internal "(" — the rpartition case.
        "An exception is setUp() is reraised after disable() is called. (test_utils.tests.SomeClass)",
        "assertRaisesMessage shouldn't interpret RE special chars. (test_utils.tests.SomeClass)",
        # Duplicate (same class as above) — should collapse to one entry.
        "test_other_method (test_utils.tests.SomeClass)",
    ]
    got = reward._django_classes(ids)
    assert got == sorted({
        "test_utils.tests.OverrideSettingsTests",
        "model_fields.test_filepathfield.FilePathFieldTests",
        "test_utils.tests.SomeClass",
    }), f"got {got}"


def test_django_classes_skips_docstring_only_entries() -> None:
    """Some django PASS_TO_PASS entries are JUST a docstring with no
    "(module.Class)" suffix at all. They contain shell metacharacters like
    '(' that would break bash. We MUST drop these to avoid the bash parse
    error overwhelming the whole eval. (Confirmed in django__django-10914.)"""
    import reward
    ids = [
        "test_legit (test_utils.tests.SomeClass)",     # keep — converts to class
        "module.path.Class.test_method",                # keep — already dotted
        "An exception is setUp() is reraised after disable() is called.",   # SKIP
        "assertRaisesMessage shouldn't interpret RE special chars.",        # SKIP
        "testing/test_assertrewrite.py::TestX::test_y",                     # SKIP (pytest-style, not dotted)
    ]
    got = reward._django_classes(ids)
    assert got == sorted({
        "test_utils.tests.SomeClass",
        "module.path.Class.test_method",
    }), f"got {got}"


def test_well_formed_pytest_id_rejects_unbalanced_brackets() -> None:
    """SWE-bench's data extraction sometimes truncates parametrized test IDs
    at whitespace inside the parameter, leaving an unbalanced '['. Passing
    these to pytest yields 'not found' → exit 4 → false 0.0. Filter them out."""
    import reward
    assert reward._well_formed_pytest_id("test_x")                       is True
    assert reward._well_formed_pytest_id("test_x[trivial]")              is True
    assert reward._well_formed_pytest_id("test_x[a-b]")                  is True
    assert reward._well_formed_pytest_id("test_x[escaped")               is False   # truncated
    assert reward._well_formed_pytest_id("test_x[multi-line")            is False
    assert reward._well_formed_pytest_id("a::test_x[nested[inner]]")     is True


def test_eval_cmd_uses_xargs_with_newline_delimiter_for_default_repo() -> None:
    """Non-django repos must use `xargs -d '\\n'` (newline-only delimiter,
    no shell quote interpretation) so parametrized test IDs with quotes,
    brackets, or whitespace survive intact. Default xargs splits on
    whitespace and treats quotes specially — that mangles pytest IDs."""
    import reward
    test_ids = [
        "tests/test_foo.py::test_bar",
        "tests/test_baz.py::test_with_params[a 'b' c]",   # the xargs-killer
    ]
    cmd, content = reward._make_eval_cmd("pytest-dev/pytest", test_ids)
    assert "xargs -d '\\n'" in cmd, \
        f"must use newline-delimiter xargs to disable quote interp: {cmd}"
    assert "-a /tmp/tests.txt" in cmd, \
        f"must read test IDs from file (argv length limit): {cmd}"
    assert "python -m pytest" in cmd
    assert content == "\n".join(test_ids), \
        "tests file content must be exact newline-joined test IDs (no shell escaping)"


def test_eval_cmd_uses_runtests_for_django() -> None:
    """Django repos use runtests.py with CLASS-level paths (not method-level).
    See _django_classes for why we lose per-method granularity."""
    import reward
    test_ids = [
        "test_override_file_upload_permissions (test_utils.tests.OverrideSettingsTests)",
        "test_callable_path (model_fields.test_filepathfield.FilePathFieldTests)",
    ]
    cmd, content = reward._make_eval_cmd("django/django", test_ids)
    assert "runtests.py" in cmd, f"django must use runtests.py: {cmd}"
    assert "pytest" not in cmd, f"django must not use pytest: {cmd}"
    assert "--settings=test_sqlite" in cmd
    # Class paths (NOT method paths) should be in the command.
    assert "test_utils.tests.OverrideSettingsTests" in cmd
    assert "model_fields.test_filepathfield.FilePathFieldTests" in cmd
    # Method names should NOT appear — class-level eval drops them.
    assert "test_override_file_upload_permissions" not in cmd, \
        "must not pass method-level paths to runtests.py"
    # Django path doesn't use the args file.
    assert content is None


def test_eval_cmd_filters_malformed_pytest_ids() -> None:
    """SWE-bench data sometimes has truncated test IDs (unbalanced '['). We
    drop them before passing to pytest so the eval runs against the well-
    formed subset instead of getting a 'not found' → exit 4 → false 0.0."""
    import reward
    test_ids = [
        "tests/test_x.py::test_a[good]",       # keep
        "tests/test_x.py::test_b[broken",      # drop — truncated
        "tests/test_x.py::test_c",             # keep
        "tests/test_x.py::test_d[also broken", # drop
    ]
    cmd, content = reward._make_eval_cmd("pytest-dev/pytest", test_ids)
    assert content is not None
    lines = content.splitlines()
    assert "tests/test_x.py::test_a[good]" in lines
    assert "tests/test_x.py::test_c" in lines
    assert "tests/test_x.py::test_b[broken" not in lines
    assert "tests/test_x.py::test_d[also broken" not in lines
    assert len(lines) == 2


def test_categorize_buckets() -> None:
    """The bucketer in reward.py mirrors metrics._REWARD_FAIL_BUCKETS so that
    a future join across the two JSONLs is keyed on the same category names."""
    import reward
    cases = {
        "missing instance_id in metadata": "other",
        "no model_patch (status=COMPLETED)": "no_model_patch",
        "test_patch apply failed (rc=1): error: ...": "test_patch_failed",
        "model_patch apply failed (rc=1): error: patch failed at line 42":
            "model_patch_failed",
        "pytest exit 1; tail of output: ...":           "pytest_exit_nonzero",
        "no test ids in metadata":                       "other",
        "reward sandbox failed: SandboxCreateError(...)": "sandbox_error",
        "something totally unexpected":                  "other",
    }
    for reason, expected in cases.items():
        assert reward._categorize(reason) == expected, \
            f"{reason!r} → {reward._categorize(reason)!r}, expected {expected!r}"


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
        test_fail_reason_jsonl_appends_when_env_set,
        test_fail_reason_jsonl_noop_when_env_unset,
        test_categorize_buckets,
        test_django_classes_extracts_class_paths_via_rpartition,
        test_django_classes_skips_docstring_only_entries,
        test_well_formed_pytest_id_rejects_unbalanced_brackets,
        test_eval_cmd_uses_xargs_with_newline_delimiter_for_default_repo,
        test_eval_cmd_uses_runtests_for_django,
        test_eval_cmd_filters_malformed_pytest_ids,
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
