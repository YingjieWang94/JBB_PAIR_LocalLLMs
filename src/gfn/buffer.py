# src/gfn/buffer.py
import json
from typing import Dict, Any, Iterable


class TransitionBuffer:
    def __init__(self):
        self.data = []

    def add(self, record: Dict[str, Any]):
        self.data.append(record)

    def extend(self, records: Iterable[Dict[str, Any]]):
        for r in records:
            self.add(r)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for r in self.data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    @staticmethod
    def load(path: str):
        buf = TransitionBuffer()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                buf.add(json.loads(line))
        return buf
