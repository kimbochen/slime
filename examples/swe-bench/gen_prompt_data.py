#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets>=2.18"]
# ///
"""
gen_prompt_data.py
==================

Build a slime-compatible JSONL of SWE-bench instances for use with the
swe-bench example's `generate.py` + `reward.py` pipeline.

Each output row carries:
  - prompt:   the issue body (what the agent sees)
  - label:    the gold patch (reward function compares against this)
  - metadata: instance_id + test specs + base commit + version
  - score:    placeholder for slime's --reward-key (filled in by reward.py)

The `instance_id` in metadata is what `sandbox.SWEBenchSandbox` keys on to
pull the per-instance Epoch AI Docker image. The test specs let the reward
function know which tests to run without needing the SWE-bench dataset
loaded at reward time.

Default produces a 10-instance bring-up slice (2 each from 5 repos) so a
new launch can be validated quickly. Use `--n-instances 500` for the full
SWE-bench Verified set.

Usage:
  ./gen_prompt_data.py                                  # 10-instance bring-up slice
  ./gen_prompt_data.py --n-instances 500 --out ...      # full set
  ./gen_prompt_data.py --per-repo 4 --n-instances 40    # 4 per repo, 10 repos
  ./gen_prompt_data.py --source lite                    # SWE-bench Lite (300 instances)

Output JSONL shape, one row per line:
  {
    "prompt":   "<issue body>",
    "label":    "<gold unified-diff patch>",
    "metadata": {
        "instance_id":              "django__django-12345",
        "repo":                     "django/django",
        "base_commit":              "abc1234...",
        "environment_setup_commit": "def5678...",
        "version":                  "4.0",
        "difficulty":               "..." | null,
        "fail_to_pass":             [list of test ids that should pass after the fix],
        "pass_to_pass":             [list of test ids that should still pass],
        "test_patch":               "<diff that introduces the FAIL_TO_PASS tests>"
    },
    "score": 0.0
  }
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


SOURCES = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite":     "princeton-nlp/SWE-bench_Lite",
}


def _row_to_jsonl_dict(row: dict) -> dict:
    """Build the slime JSONL row from a SWE-bench dataset row.

    FAIL_TO_PASS and PASS_TO_PASS are stored as JSON-encoded strings in the
    dataset; parse them into Python lists so reward.py can use them directly.
    """
    fail_to_pass = json.loads(row["FAIL_TO_PASS"]) if isinstance(row["FAIL_TO_PASS"], str) else row["FAIL_TO_PASS"]
    pass_to_pass = json.loads(row["PASS_TO_PASS"]) if isinstance(row["PASS_TO_PASS"], str) else row["PASS_TO_PASS"]

    return {
        "prompt": row["problem_statement"],
        "label": row["patch"],  # gold patch — reward.py diffs against this
        "metadata": {
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "environment_setup_commit": row.get("environment_setup_commit"),
            "version": row.get("version"),
            "difficulty": row.get("difficulty"),
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
            "test_patch": row.get("test_patch", ""),
        },
        "score": 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--out",
        default="mnt/data/swebench_verified/sample.jsonl",
        help="Output JSONL path, relative to cwd (default: %(default)s).",
    )
    ap.add_argument(
        "--n-instances",
        type=int,
        default=10,
        help="Total instances to emit (default: %(default)s). "
             "Pass 0 to emit every instance from the source.",
    )
    ap.add_argument(
        "--per-repo",
        type=int,
        default=2,
        help="Max instances to take from any single repo (default: %(default)s); "
             "set to 0 to disable per-repo balancing.",
    )
    ap.add_argument(
        "--source",
        choices=list(SOURCES.keys()),
        default="verified",
        help="Which SWE-bench split to pull from. 'lite' (300 instances) is "
             "filtered to simpler patches; 'verified' (500 instances) is the "
             "human-validated curation. Both are subsets of the original "
             "SWE-bench, so Epoch AI's per-instance Docker images cover both.",
    )
    args = ap.parse_args()

    dataset_path = SOURCES[args.source]
    ds = load_dataset(dataset_path, split="test")
    print(f"Loaded {len(ds)} instances from {dataset_path}")

    seen: dict[str, int] = {}
    picks = []
    for row in ds:
        if args.per_repo > 0 and seen.get(row["repo"], 0) >= args.per_repo:
            continue
        picks.append(row)
        seen[row["repo"]] = seen.get(row["repo"], 0) + 1
        if args.n_instances > 0 and len(picks) >= args.n_instances:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in picks:
            f.write(json.dumps(_row_to_jsonl_dict(r)) + "\n")
    print(f"Wrote {len(picks)} instances to {out_path}")
    for r in picks:
        print(f"  {r['instance_id']}")


if __name__ == "__main__":
    main()
