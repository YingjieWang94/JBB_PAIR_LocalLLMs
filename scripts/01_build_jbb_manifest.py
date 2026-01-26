from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import orjson
from datasets import load_dataset


def _norm_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def main() -> None:
    out_root = Path(r"C:\Projects\jbb_pair")
    out_path = out_root / "data" / "processed" / "jbb_manifest.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: config name required
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")

    n_written = 0
    with out_path.open("wb") as f:
        for split_name, split in ds.items():
            # Here split_name is literally 'harmful' or 'benign'
            subset = split_name.lower().strip()

            for row in split:
                row = dict(row)

                rec: Dict[str, Any] = {
                    "dataset": "JailbreakBench/JBB-Behaviors",
                    "config": "behaviors",
                    "hf_split": split_name,
                    "subset": subset,  # harmful | benign

                    # JBB columns (as observed)
                    "index": row.get("Index", None),
                    "goal": _norm_str(row.get("Goal")),
                    "target_reference": _norm_str(row.get("Target")),
                    "behavior_id": _norm_str(row.get("Behavior")),   # behavior name (stable enough)
                    "category": _norm_str(row.get("Category")),
                    "source": _norm_str(row.get("Source")),

                    # keep raw for traceability
                    "raw": row,
                }

                f.write(orjson.dumps(rec))
                f.write(b"\n")
                n_written += 1

    print(f"Wrote {n_written} records -> {out_path}")


if __name__ == "__main__":
    main()
