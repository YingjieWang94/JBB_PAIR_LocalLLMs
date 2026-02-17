#!/usr/bin/env python3
"""
Sample attacker prompts from a trained GFN sampler checkpoint (per behavior).

This is a utility to sanity-check the learned policy before integrating into run.py.
It prints the top-k prompt_ids and (optionally) their texts.

Usage:
python scripts/GFN_sampler/04_sample_prompts.py \
  --ckpt /path/to/run_dir/ckpt_latest.pt \
  --banks-dir /path/to/banks \
  --behavior-id <id> --t 0 --best-score 0.0 --score 0.0 --budget-left 19 --topk 20
"""
from __future__ import annotations

import sys, json, argparse, math
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.GFN_sampler.03_train_gfn_sampler import DiscreteSamplerGFN  # reuse class

def load_vocab_text(banks_dir: Path):
    p = banks_dir / "prompt_vocab.jsonl"
    pid2text = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            pid2text[int(r["prompt_id"])] = r["text"]
    return pid2text

def load_beh2prompts(banks_dir: Path):
    p = banks_dir / "behavior_to_prompt_ids.jsonl"
    out = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            out[str(r["behavior_id"])] = [int(x) for x in r.get("prompt_ids", [])]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--banks-dir", required=True)
    ap.add_argument("--behavior-id", required=True)
    ap.add_argument("--t", type=float, default=0.0)
    ap.add_argument("--best-score", type=float, default=0.0)
    ap.add_argument("--score", type=float, default=0.0)
    ap.add_argument("--budget-left", type=float, default=0.0)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--show-text", action="store_true")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    args = ap.parse_args()

    banks_dir = Path(args.banks_dir)
    beh2prompts = load_beh2prompts(banks_dir)
    behaviors = sorted(beh2prompts.keys())
    beh2idx = {b:i for i,b in enumerate(behaviors)}

    if args.behavior_id not in beh2idx:
        raise SystemExit(f"behavior_id not found in banks: {args.behavior_id}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    n_beh = len(behaviors)
    # infer n_prompts from prompt_emb weight
    n_prompts = ckpt["model"]["prompt_emb.weight"].shape[0]

    # infer dims
    beh_dim = ckpt["model"]["beh_emb.weight"].shape[1]
    prompt_dim = ckpt["model"]["prompt_emb.weight"].shape[1]
    # infer hidden from mlp.0 weight
    hidden = ckpt["model"]["mlp.0.weight"].shape[0]
    state_dim = ckpt["model"]["mlp.0.weight"].shape[1] - beh_dim

    model = DiscreteSamplerGFN(n_beh, n_prompts, beh_dim, prompt_dim, state_dim, hidden)
    model.load_state_dict(ckpt["model"])

    device = torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    pid2text = load_vocab_text(banks_dir) if args.show_text else {}

    allowed = beh2prompts[args.behavior_id]
    beh_idx = beh2idx[args.behavior_id]
    state_num = torch.tensor([[args.t, args.best_score, args.score, args.budget_left]], dtype=torch.float32, device=device)
    beh_t = torch.tensor([beh_idx], dtype=torch.long, device=device)

    # score all allowed prompts in chunks
    scores = []
    with torch.no_grad():
        s_emb = model.state_embed(beh_t, state_num)  # [1,D]
        chunk = 2048
        for i in range(0, len(allowed), chunk):
            ids = torch.tensor([allowed[i:i+chunk]], dtype=torch.long, device=device)
            logits = model.logits_for_actions(s_emb, ids)[0]  # [K]
            scores.append(logits.detach().cpu())
    logits_all = torch.cat(scores, dim=0)
    probs = F.softmax(logits_all, dim=0)
    topk = min(args.topk, probs.numel())
    vals, idxs = torch.topk(probs, k=topk)
    for rank, (p, j) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        pid = allowed[j]
        if args.show_text:
            text = pid2text.get(pid, "")[:200].replace("\n","\\n")
            print(f"{rank:02d}. pid={pid} p={p:.6f} text={text}")
        else:
            print(f"{rank:02d}. pid={pid} p={p:.6f}")

if __name__ == "__main__":
    main()
