"""Python auto-imports sitecustomize at interpreter startup if it appears
on sys.path. By including examples/swe-bench/telemetry/ in PYTHONPATH, this
file fires in every Python process — rollout AND trainer Ray actors —
without modifying slime.

It triggers log_patch which monkey-patches slime.utils.logging_utils.log to
also append to SLIME_METRICS_JSONL. That single entry point captures every
wandb metric across rollout, trainer, and eval — without needing each
component to know about the JSONL.

If a different sitecustomize.py earlier on sys.path shadows ours, the
fallback is to add `import log_patch` directly to the launch script.
"""
import sys


try:
    import log_patch  # noqa: F401 — imported for the monkey-patch side effect
except ImportError:
    # log_patch (or slime itself) isn't on sys.path yet — fine; the patch
    # can be triggered later by an explicit import.
    pass
except Exception as e:
    # Never break interpreter startup, no matter what goes wrong.
    print(f"[sitecustomize] log_patch failed to load: {e}", file=sys.stderr)
