from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable
import orjson
import time
import uuid


def new_run_id(prefix: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:6]}"


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("wb")

    def write(self, record: Dict):
        self.f.write(orjson.dumps(record))
        self.f.write(b"\n")

    def close(self):
        self.f.close()
