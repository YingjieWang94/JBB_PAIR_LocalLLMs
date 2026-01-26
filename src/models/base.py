from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModelOutput:
    text: str
    raw: Optional[Dict[str, Any]] = None


class TextClassifier:
    """Binary or multi-class classifier interface."""
    def classify(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError
