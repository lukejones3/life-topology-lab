"""Shared identifiers for the private embedding store.

The public laboratory does not build or publish subject-level embeddings, but
analysis modules still need to address tables produced by the private pipeline.
"""

from __future__ import annotations

import hashlib
import re


def _table_name(model_name: str) -> str:
    """Return the stable LanceDB table name used by the private pipeline."""
    readable = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")[-40:]
    digest = hashlib.sha1(model_name.encode()).hexdigest()[:8]
    return f"daily_state_{readable}_{digest}"
