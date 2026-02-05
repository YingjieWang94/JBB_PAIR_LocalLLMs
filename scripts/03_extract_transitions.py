# scripts/03_extract_transitions.py
import json
import argparse
from pathlib import Path


def main(args):
    out = open(args.out, "w", encoding="utf-8")

    for path in Path(args.in_dir).glob("*.jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)

                if "behavior_id" not in row:
                    continue

                transitions = row.get("turns", [])
                best = 0.0

                for t, tr in enumerate(transitions):
                    score = tr.get("score", 0.0)
                    best = max(best, score)

                    out.write(json.dumps({
                        "behavior_id": row["behavior_id"],
                        "t": t,
                        "score": score,
                        "best_score": best,
                        "done_reason": row.get("done_reason", "unknown"),
                    }) + "\n")

    out.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
