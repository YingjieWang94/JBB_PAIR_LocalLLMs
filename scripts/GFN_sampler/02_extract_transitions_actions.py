#!/usr/bin/env python3
"""
Extract transitions WITH actions (prompt_id) for GFN sampler training.

Reads per-turn JSONL files produced by scripts/run.py and maps attacker_prompt -> prompt_id
using the prompt banks produced by 01_build_prompt_banks.py.

Outputs a single transitions JSONL containing:
- per-turn records with:
    episode_id, run_id, behavior_id, turn, t,
    score, best_score,
    prompt_id (action), guard_label, score_source
- plus an episode_end marker record after each episode (optional but recommended).

Interrupt/resume safe:
- writes output incrementally to a temp file and atomically renames at the end
- can resume by skipping already-processed input files (manifest)
"""
from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict
import ast
import hashlib
import os
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST_NAME = "_extract_manifest.json"

def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def _try_parse_dict(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None

def _extract_label_and_punsafe(row):
    if "score" in row and row["score"] is not None:
        try:
            return None, float(row["score"]), "score_field"
        except Exception:
            pass
    d = _try_parse_dict(row.get("guard_response_raw"))
    if d is not None:
        label = d.get("label")
        p_unsafe = d.get("p_unsafe", d.get("score"))
        if p_unsafe is not None:
            try:
                return str(label) if label is not None else None, float(p_unsafe), "guard_response_raw"
            except Exception:
                pass
    d = _try_parse_dict(row.get("guard_response_label"))
    if d is not None:
        label = d.get("label")
        p_unsafe = d.get("p_unsafe", d.get("score"))
        if p_unsafe is not None:
            try:
                return str(label) if label is not None else None, float(p_unsafe), "guard_response_label_dictstr"
            except Exception:
                pass
    gl = row.get("guard_response_label")
    if isinstance(gl, str):
        l = gl.strip().lower()
        if l:
            if "unsafe" in l:
                return "unsafe", 1.0, "guard_response_label_str"
            if "safe" in l:
                return "safe", 0.0, "guard_response_label_str"
    return None, None, "missing"

def atomic_rename(tmp: Path, final: Path) -> None:
    tmp.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, final)

def load_vocab(banks_dir: Path) -> dict[str,int]:
    vocab_path = banks_dir / "prompt_vocab.jsonl"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing {vocab_path}. Run 01_build_prompt_banks.py first.")
    h2pid = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            h2pid[r["hash"]] = int(r["prompt_id"])
    return h2pid

def load_manifest(out_path: Path) -> dict:
    mp = out_path.parent / MANIFEST_NAME
    if mp.exists():
        with open(mp, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed_files": {}}

def save_manifest(out_path: Path, manifest: dict) -> None:
    mp = out_path.parent / MANIFEST_NAME
    tmp = mp.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, mp)

def iter_input_files(in_dir: Path):
    for p in sorted(in_dir.glob("*.jsonl")):
        if p.is_file():
            yield p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Directory containing per-turn JSONL files from scripts/run.py")
    ap.add_argument("--banks-dir", required=True, help="Directory produced by 01_build_prompt_banks.py")
    ap.add_argument("--out", required=True, help="Output transitions JSONL")
    ap.add_argument("--emit-episode-end", action="store_true", help="Emit an episode_end record per episode")
    ap.add_argument("--resume", action="store_true", help="Resume using manifest (skip already processed files)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    banks_dir = Path(args.banks_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h2pid = load_vocab(banks_dir)

    manifest = load_manifest(out_path) if args.resume else {"processed_files": {}}

    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp") if not args.resume else out_path.with_suffix(out_path.suffix + ".part")
    mode = "a" if args.resume and tmp_out.exists() else "w"

    # We extract episodes per file to keep memory low.
    with open(tmp_out, mode, encoding="utf-8") as out:
        processed_files = 0
        for fp in iter_input_files(in_dir):
            st = fp.stat()
            key = str(fp.resolve())
            sig = f"{st.st_size}:{int(st.st_mtime)}"
            if args.resume and manifest["processed_files"].get(key) == sig:
                continue

            episodes = defaultdict(list)
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if "behavior_id" not in row:
                        continue
                    run_id = row.get("run_id", "no_run_id")
                    behavior_id = row.get("behavior_id", "no_behavior_id")
                    episode_id = f"{run_id}::{behavior_id}"
                    episodes[episode_id].append(row)

            # write episodes from this file
            for episode_id, turns in episodes.items():
                turns = sorted(turns, key=lambda r: int(r.get("turn", 0)))
                best = float("-inf")
                seen_unsafe = False
                budget = turns[0].get("budget_per_try") if turns else None

                for i, row in enumerate(turns):
                    label, p_unsafe, source = _extract_label_and_punsafe(row)
                    score = float(p_unsafe) if p_unsafe is not None else (1.0 if (label and str(label).lower()=="unsafe") else 0.0)
                    best = score if best == float("-inf") else max(best, score)
                    if label is not None and str(label).lower() == "unsafe":
                        seen_unsafe = True

                    prompt = row.get("attacker_prompt", "")
                    prompt_id = None
                    if isinstance(prompt, str) and prompt.strip():
                        h = stable_hash(prompt.strip())
                        prompt_id = h2pid.get(h)

                    rec = {
                        "episode_id": episode_id,
                        "run_id": row.get("run_id"),
                        "behavior_id": row.get("behavior_id"),
                        "turn": int(row.get("turn", i)),
                        "t": i,
                        "score": float(score),
                        "best_score": float(best if best != float("-inf") else 0.0),
                        "budget_per_try": budget,
                        "prompt_id": prompt_id,
                        "guard_label": label if label is not None else row.get("guard_response_label"),
                        "score_source": source,
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")

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

            manifest["processed_files"][key] = sig
            processed_files += 1
            if processed_files % 20 == 0:
                out.flush()
                os.fsync(out.fileno())
                save_manifest(out_path, manifest)
                print(f"[PROGRESS] processed_files={processed_files}")

        out.flush()
        os.fsync(out.fileno())

    # finalize
    if not args.resume:
        atomic_rename(tmp_out, out_path)
    else:
        # on resume, keep .part file; user can rename manually or rerun without --resume to finalize.
        # If we see completion (all files processed), we can atomically move to final name.
        # We'll attempt that now.
        # Note: if input dir changes later, rerun will append.
        all_done = True
        for fp in iter_input_files(in_dir):
            st = fp.stat()
            key = str(fp.resolve())
            sig = f"{st.st_size}:{int(st.st_mtime)}"
            if manifest["processed_files"].get(key) != sig:
                all_done = False
                break
        if all_done:
            atomic_rename(tmp_out, out_path)
            (out_path.parent / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[OK] finalized {out_path}")
        else:
            save_manifest(out_path, manifest)
            print(f"[OK] wrote partial {tmp_out} (resume later)")

if __name__ == "__main__":
    main()
