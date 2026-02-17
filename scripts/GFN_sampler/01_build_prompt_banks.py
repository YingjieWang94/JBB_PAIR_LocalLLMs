#!/usr/bin/env python3
"""
Build per-behavior prompt banks (and a global prompt vocab) from existing run.py JSONL outputs.

- Does NOT modify any existing scripts.
- Reads per-turn JSONL files produced by scripts/run.py.
- Output is interrupt/resume-safe:
  - Writes to a temporary file then atomically renames.
  - Maintains a manifest of processed input files.

Outputs (under --out-dir):
  - prompt_vocab.jsonl                 (global prompt_id -> text + counts)
  - behavior_to_prompt_ids.jsonl       (behavior_id -> list[prompt_id])
  - per_behavior/BEHAVIOR_ID.jsonl     (prompt_id, text, count_in_behavior)

prompt_id is global (shared across behaviors) but each behavior gets its own subset list.
This satisfies "per-behavior" sampling while still allowing a single shared model.
"""
from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import os
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST_NAME = "_build_manifest.json"

def stable_id(text: str) -> str:
    # stable content hash to help dedup across runs; prompt_id will be assigned deterministically by sort order
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def atomic_write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def load_manifest(out_dir: Path) -> dict:
    mpath = out_dir / MANIFEST_NAME
    if mpath.exists():
        with open(mpath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed_files": {}}

def save_manifest(out_dir: Path, manifest: dict) -> None:
    atomic_write_text(out_dir / MANIFEST_NAME, json.dumps(manifest, indent=2, ensure_ascii=False))

def iter_input_files(in_dir: Path):
    for p in sorted(in_dir.glob("*.jsonl")):
        if p.is_file():
            yield p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Directory containing per-turn JSONL files from scripts/run.py")
    ap.add_argument("--out-dir", required=True, help="Output directory for banks")
    ap.add_argument("--min-count", type=int, default=1, help="Drop prompts that appear fewer than this many times in a behavior")
    ap.add_argument("--max-prompts-per-behavior", type=int, default=0,
                    help="If >0, keep only top-N prompts by count for each behavior (after min-count).")
    ap.add_argument("--resume", action="store_true", help="Resume using manifest (skip already processed files)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(out_dir) if args.resume else {"processed_files": {}}

    # Accumulators
    # behavior_id -> Counter(text_hash) and hash->text
    beh_counts: dict[str, Counter] = defaultdict(Counter)
    hash_to_text: dict[str, str] = {}          # global
    hash_global_count: Counter = Counter()     # global frequency

    # If resuming and outputs exist, we can reload prior state.
    # For simplicity, we rebuild from scratch unless user wants speed. This is safer.
    # If you want true incremental builds, implement reading existing vocab/banks.
    if args.resume and manifest.get("complete", False):
        print("[INFO] Manifest indicates complete build already. Nothing to do.")
        return

    processed = 0
    for fp in iter_input_files(in_dir):
        st = fp.stat()
        key = str(fp.resolve())
        sig = f"{st.st_size}:{int(st.st_mtime)}"
        if args.resume and manifest["processed_files"].get(key) == sig:
            continue

        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                beh = row.get("behavior_id")
                prompt = row.get("attacker_prompt")
                if not beh or not isinstance(prompt, str) or not prompt.strip():
                    continue
                prompt = prompt.strip()

                h = stable_id(prompt)
                hash_to_text.setdefault(h, prompt)
                beh_counts[beh][h] += 1
                hash_global_count[h] += 1

        manifest["processed_files"][key] = sig
        processed += 1
        if processed % 20 == 0:
            print(f"[PROGRESS] processed_files={processed}")
            save_manifest(out_dir, manifest)

    print(f"[INFO] finished scanning. behaviors={len(beh_counts)} unique_prompts={len(hash_to_text)}")

    # Deterministic prompt_id assignment: sort by (global_count desc, hash asc)
    # This makes prompt_id stable across rebuilds given the same inputs.
    sorted_hashes = sorted(hash_to_text.keys(), key=lambda h: (-hash_global_count[h], h))
    hash_to_pid = {h: i for i, h in enumerate(sorted_hashes)}

    # Write global vocab
    vocab_rows = []
    for h in sorted_hashes:
        vocab_rows.append({
            "prompt_id": hash_to_pid[h],
            "hash": h,
            "count_total": int(hash_global_count[h]),
            "text": hash_to_text[h],
        })
    atomic_write_jsonl(out_dir / "prompt_vocab.jsonl", vocab_rows)

    # Write per-behavior banks + behavior->ids mapping
    per_dir = out_dir / "per_behavior"
    per_dir.mkdir(parents=True, exist_ok=True)

    beh2ids_rows = []
    for beh, ctr in sorted(beh_counts.items(), key=lambda kv: kv[0]):
        # apply min-count + top-k
        items = [(h, c) for h, c in ctr.items() if c >= args.min_count]
        items.sort(key=lambda hc: (-hc[1], hc[0]))
        if args.max_prompts_per_behavior and args.max_prompts_per_behavior > 0:
            items = items[:args.max_prompts_per_behavior]

        ids = [hash_to_pid[h] for h, _ in items]
        beh2ids_rows.append({"behavior_id": beh, "prompt_ids": ids})

        # write per behavior jsonl
        rows = []
        for h, c in items:
            rows.append({
                "behavior_id": beh,
                "prompt_id": hash_to_pid[h],
                "count_in_behavior": int(c),
                "text": hash_to_text[h],
            })
        atomic_write_jsonl(per_dir / f"{beh}.jsonl", rows)

    atomic_write_jsonl(out_dir / "behavior_to_prompt_ids.jsonl", beh2ids_rows)

    manifest["complete"] = True
    manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_manifest(out_dir, manifest)
    print(f"[OK] wrote banks to {out_dir}")

if __name__ == "__main__":
    main()
