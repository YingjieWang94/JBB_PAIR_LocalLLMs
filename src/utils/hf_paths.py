from __future__ import annotations

import os
from pathlib import Path


def resolve_local_model(model_id_or_path: str) -> str:
    """
    Resolve HF repo IDs to your on-disk snapshot layout:
      org/name -> $MODEL_ROOT/org__name
    If an explicit path is provided and exists, return it unchanged.
    """
    p = Path(model_id_or_path)
    if p.exists():
        return str(p)

    model_root = os.environ.get("MODEL_ROOT")
    if not model_root:
        return model_id_or_path

    if "/" in model_id_or_path:
        org, name = model_id_or_path.split("/", 1)
        candidate = Path(model_root) / f"{org}__{name}"
        if candidate.exists():
            return str(candidate)

    return model_id_or_path


def local_only_enabled() -> bool:
    """
    Treat offline flags as authoritative.
    """
    v = os.environ.get("TRANSFORMERS_OFFLINE", "")
    return v.lower() in ("1", "true", "yes", "y")
