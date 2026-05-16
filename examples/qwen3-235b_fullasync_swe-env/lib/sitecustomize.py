"""
sitecustomize.py — Python auto-imports this at interpreter startup if it
appears on sys.path (which `lib/` does, because every experiment's
run_swe.sh puts `${LIB_DIR}` first in PYTHONPATH).

Triggers patch_log_utils which monkey-patches slime.utils.logging_utils.log
to also append a JSON line to SLIME_METRICS_JSONL. This is the ONE entry
point that runs in every Python process — rollout AND trainer — without
needing to modify slime itself.

If a different sitecustomize.py exists earlier on sys.path, it shadows
this one. To force our patch in that case, an experiment can add
`import patch_log_utils` to its run_swe.sh entry point.
"""
import sys


try:
    import patch_log_utils  # noqa: F401 — imported for the monkey-patch side effect
except ImportError:
    # patch_log_utils (or slime itself) isn't on sys.path yet — that's
    # fine; the patch can be triggered later by an explicit import.
    pass
except Exception as e:
    # Don't break interpreter startup if anything else goes wrong.
    print(f"[sitecustomize] patch_log_utils failed to load: {e}", file=sys.stderr)
