"""Monkey-patches slime.utils.logging_utils.log to ALSO append a JSON line
to SLIME_METRICS_JSONL (in addition to wandb / tensorboard side effects).

Activated by setting SLIME_METRICS_JSONL=/path/to/file.jsonl in the runtime
env. Unset → no-op pass-through. Slime behaves identically when not enabled.

Why this lives in a standalone module rather than inside metrics.py:

  metrics.py is only imported by the rollout Ray actor (via
  --custom-rollout-log-function-path). The trainer actor never imports it.
  Without a process-agnostic patch loader, the JSONL would only catch
  rollout-side log calls and miss the trainer-side ones from:
    - slime/backends/megatron_utils/data.py:213   (rollout/raw_reward, advantages, kl...)
    - slime/backends/megatron_utils/model.py:655  (train/loss, grad_norm, pg_loss...)

  The sibling sitecustomize.py auto-imports this module at interpreter
  startup, so every Python process (rollout + trainer) applies the patch
  without modifying slime itself.

Concurrency: multiple Ray actor processes may write to the same file.
POSIX O_APPEND is atomic for writes under PIPE_BUF (typically 4 KB on
Linux); each record is one short JSON line, well under that limit. The
threading.Lock guards intra-process races within a single actor.
"""

from __future__ import annotations

import functools
import json
import os
import threading
import time


_jsonl_lock = threading.Lock()
_patched = False


def _patch() -> None:
    """Idempotent — safe to call multiple times."""
    global _patched
    if _patched:
        return

    try:
        from slime.utils import logging_utils
    except ImportError:
        # slime not yet on sys.path (e.g., sitecustomize ran before PYTHONPATH
        # was fully expanded). Caller can retry later via explicit import.
        return

    orig_log = logging_utils.log

    @functools.wraps(orig_log)
    def _log_with_jsonl(args, metrics, step_key):
        # Preserve the original side effects (wandb, tensorboard, ...).
        orig_log(args, metrics, step_key)

        path = os.environ.get("SLIME_METRICS_JSONL")
        if not path:
            return

        record = {
            "step_key": step_key,
            "step": metrics.get(step_key),
            "timestamp": time.time(),
            **{k: v for k, v in metrics.items() if k != step_key},
        }
        with _jsonl_lock:
            # Defensive: create parent dir if missing. Handles paths set in
            # environments where the dir wasn't pre-created (e.g., a fresh
            # cluster, or a path that only exists in one of host/container).
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")

    logging_utils.log = _log_with_jsonl
    _patched = True


# Auto-apply on import.
_patch()
