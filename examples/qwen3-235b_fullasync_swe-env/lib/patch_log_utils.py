"""
patch_log_utils.py
=====================

Monkey-patches `slime.utils.logging_utils.log` to ALSO append a JSON line
to SLIME_METRICS_JSONL (in addition to the original wandb / tensorboard
side effects).

Why this lives in its own module instead of in `rollout_metrics.py`:

  `rollout_metrics` is only imported by the **rollout** Ray actor (via
  slime's `--custom-generate-function-path`). The **trainer** actor never
  imports it. Without a separate patch loader, the JSONL only catches
  rollout-side `logging_utils.log` calls (rollout-step custom logger +
  eval logger), and misses the trainer-side ones from:
    - slime/backends/megatron_utils/data.py:213  (rollout/raw_reward, rewards, advantages, kl, ...)
    - slime/backends/megatron_utils/model.py:655 (train/loss, grad_norm, pg_loss, ...)

  By moving the patch into a module that gets auto-imported by Python's
  site machinery (via the sibling `sitecustomize.py`), every Python
  process — including the trainer — applies the same patch on startup.

Activated by setting SLIME_METRICS_JSONL=/path/to/file.jsonl in the
runtime env. No-op pass-through when unset.
"""
from __future__ import annotations

import functools
import json
import os
import threading
import time


_jsonl_lock = threading.Lock()
_patched = False


def _patch():
    """Idempotent — safe to call multiple times."""
    global _patched
    if _patched:
        return

    try:
        from slime.utils import logging_utils
    except ImportError:
        # slime not yet on sys.path (e.g., when sitecustomize runs before
        # PYTHONPATH is fully expanded). Caller can retry later.
        return

    orig_log = logging_utils.log

    @functools.wraps(orig_log)
    def _logging_with_jsonl(args, metrics, step_key):
        # Preserve the original side effects (wandb / tensorboard / etc.)
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
        # POSIX O_APPEND is atomic for sub-PIPE_BUF writes; the lock is
        # defensive against intra-process races inside the same actor.
        with _jsonl_lock:
            with open(path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")

    logging_utils.log = _logging_with_jsonl
    _patched = True


# Auto-apply on import.
_patch()
