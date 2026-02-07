# scripts/03_extract_transitions.py
import json
import argparse
from pathlib import Path
from collections import defaultdict
import ast


def _try_parse_dict(x):
    """Parse x into a dict if possible. Supports dict, JSON string, python-literal string."""
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        # Try JSON first
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # Try python literal (handles "{'label': 'safe', ...}")
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _extract_label_and_punsafe(row):
    """
    Returns: (label:str|None, p_unsafe:float|None, source:str)
    """
    # 1) If score already computed upstream
    if "score" in row and row["score"] is not None:
        try:
            return None, float(row["score"]), "score_field"
        except Exception:
            pass

    # 2) guard_response_raw may be dict/string-dict
    d = _try_parse_dict(row.get("guard_response_raw"))
    if d is not None:
        label = d.get("label")
        p_unsafe = d.get("p_unsafe", d.get("score"))
        if p_unsafe is not None:
            try:
                return str(label) if label is not None else None, float(p_unsafe), "guard_response_raw"
            except Exception:
                pass

    # 3) your current data: guard_response_label stores the whole dict as a string
    d = _try_parse_dict(row.get("guard_response_label"))
    if d is not None:
        label = d.get("label")
        p_unsafe = d.get("p_unsafe", d.get("score"))
        if p_unsafe is not None:
            try:
                return str(label) if label is not None else None, float(p_unsafe), "guard_response_label_dictstr"
            except Exception:
                pass

    # 4) Otherwise treat guard_response_label as plain label string
    gl = row.get("guard_response_label")
    if isinstance(gl, str):
        l = gl.strip().lower()
        if l:
            # common conventions
            if "unsafe" in l:
                return "unsafe", 1.0, "guard_response_label_str"
            if "safe" in l:
                return "safe", 0.0, "guard_response_label_str"

    return None, None, "missing"


def main(args):
    in_dir = Path(args.in_dir)
    out_path = Path(args.out)

    episodes = defaultdict(list)

    # Load all per-turn rows
    for path in in_dir.glob("*.jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "behavior_id" not in row:
                    continue
                run_id = row.get("run_id", "no_run_id")
                behavior_id = row.get("behavior_id", "no_behavior_id")
                episode_id = f"{run_id}::{behavior_id}"
                episodes[episode_id].append(row)

    # Write per-turn transitions
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for episode_id, turns in episodes.items():
            # sort by actual turn index from run.py
            turns = sorted(turns, key=lambda r: int(r.get("turn", 0)))

            best = float("-inf")
            budget = None
            seen_unsafe = False

            # Budget per try is usually constant; grab from first row if present
            if turns:
                budget = turns[0].get("budget_per_try")

            for i, row in enumerate(turns):
                label, p_unsafe, source = _extract_label_and_punsafe(row)

                # If we still have no numeric score, fall back to label if possible
                score = None
                if p_unsafe is not None:
                    score = p_unsafe
                elif label is not None:
                    score = 1.0 if str(label).lower() == "unsafe" else 0.0
                else:
                    score = 0.0  # conservative default

                if best == float("-inf"):
                    best = score
                else:
                    best = max(best, score)

                # done_reason inference
                if label is not None and str(label).lower() == "unsafe":
                    seen_unsafe = True

                # t = 0..T-1 index for the episode
                rec = {
                    "episode_id": episode_id,
                    "run_id": row.get("run_id"),
                    "behavior_id": row.get("behavior_id"),
                    "turn": int(row.get("turn", i)),
                    "t": i,
                    "score": float(score),
                    "best_score": float(best),
                    "budget_per_try": budget,
                    "guard_label": label if label is not None else row.get("guard_response_label"),
                    "score_source": source,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Optionally emit an episode-end marker line (useful for debugging)
            if args.emit_episode_end:
                done_reason = "unsafe" if seen_unsafe else ("budget" if budget is not None and len(turns) >= int(budget) else "unknown")
                end = {
                    "episode_id": episode_id,
                    "run_id": turns[0].get("run_id") if turns else None,
                    "behavior_id": turns[0].get("behavior_id") if turns else None,
                    "t": len(turns),
                    "episode_end": True,
                    "T": len(turns),
                    "best_score": float(best if best != float("-inf") else 0.0),
                    "done_reason": done_reason,
                }
                out.write(json.dumps(end, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Directory containing per-turn JSONL files from scripts/run.py")
    ap.add_argument("--out", required=True, help="Output transitions JSONL")
    ap.add_argument("--emit-episode-end", action="store_true", help="Emit an episode_end marker record per episode")
    main(ap.parse_args())
