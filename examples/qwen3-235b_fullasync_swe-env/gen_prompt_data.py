#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets>=2.18"]
# ///
"""
gen_prompt_data.py
==================

Build a slime-compatible JSONL of SWE-bench Verified instances. Each line
contains the issue body as the agent's prompt and the instance_id as the
label, which `coding_sandbox.py` uses to pull the per-instance Docker image.

Default produces a 10-instance bring-up slice (2 each from 5 repos) so a
new launch can be validated quickly. Use `--n-instances 500` for the full
SWE-bench Verified set.

Usage:
  ./gen_prompt_data.py                                    # 10-instance bring-up slice
  ./gen_prompt_data.py --n-instances 500 --out ...        # full set
  ./gen_prompt_data.py --per-repo 4 --n-instances 40 ...  # 4 per repo, 10 repos

Output JSONL shape, one row per line:
  {"prompt": "<issue body>", "label": "<instance_id>", "score": 0.0}

`score` is a placeholder slime expects when `--reward-key score` is set;
the actual reward is computed by `generate_with_codingagent.reward_func`.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out",
        default="mnt/data/swebench_verified/sample.jsonl",
        help="Output JSONL path, relative to cwd (default: %(default)s)",
    )
    ap.add_argument(
        "--n-instances",
        type=int,
        default=10,
        help="Total instances to emit (default: %(default)s)",
    )
    ap.add_argument(
        "--per-repo",
        type=int,
        default=2,
        help="Max instances to take from any single repo (default: %(default)s); "
             "set to 0 to disable balancing.",
    )
    args = ap.parse_args()

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    print(f"Loaded {len(ds)} Verified instances from princeton-nlp/SWE-bench_Verified")

    seen: dict[str, int] = {}
    picks = []
    for row in ds:
        if args.per_repo > 0 and seen.get(row["repo"], 0) >= args.per_repo:
            continue
        picks.append(row)
        seen[row["repo"]] = seen.get(row["repo"], 0) + 1
        if len(picks) >= args.n_instances:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in picks:
            f.write(json.dumps({
                "prompt": r["problem_statement"],
                "label": r["instance_id"],
                "score": 0.0,
            }) + "\n")
    print(f"Wrote {len(picks)} instances to {out_path}")
    for r in picks:
        print(f"  {r['instance_id']}")


if __name__ == "__main__":
    main()
