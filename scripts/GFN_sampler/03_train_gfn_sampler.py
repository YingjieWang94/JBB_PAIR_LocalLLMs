#!/usr/bin/env python3
"""
Train a discrete GFN sampler (Trajectory Balance) that selects attacker prompts
from a per-behavior prompt bank.

Key properties:
- Does NOT require modifying scripts/run.py outputs; it learns from logged attacker_prompt strings.
- Uses prompt_id as discrete action.
- Uses per-behavior action set via behavior_to_prompt_ids.jsonl.
- Streaming training over transitions JSONL, with robust checkpointing and resume.

Model:
- Behavior embedding + numeric state features -> state embedding
- Prompt embedding lookup -> prompt embedding
- Logit(s,a) = dot(state_emb, prompt_emb) + bias_prompt

Training:
- TB objective on terminal reward R = exp(beta * best_score) (same as src/gfn/reward.py)
- For log P_F(tau), we approximate per-step log pi(a|s) using sampled softmax:
  For each step, we construct a candidate set {positive + negatives sampled from same behavior's prompt_ids}.
  This is efficient even when the full action set is large.

Resume:
- Checkpoints:
    run_dir/ckpt_latest.pt
    run_dir/ckpt_stepXXXX.pt (optional periodic)
- State:
    run_dir/state.json stores file byte offset, episode counter, global step.
- Safe writes: write tmp then rename.

Usage example:
python scripts/GFN_sampler/03_train_gfn_sampler.py \
  --transitions /path/to/transitions.jsonl \
  --banks-dir /path/to/banks \
  --run-dir /path/to/out_run \
  --device cuda --beta 5.0 --negatives 128 --batch-episodes 32
"""
from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# reuse reward definition to stay consistent
from src.gfn.reward import terminal_reward

def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def load_behavior_to_prompts(banks_dir: Path) -> Dict[str, List[int]]:
    p = banks_dir / "behavior_to_prompt_ids.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run 01_build_prompt_banks.py first.")
    out: Dict[str, List[int]] = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[str(r["behavior_id"])] = [int(x) for x in r.get("prompt_ids", [])]
    return out

def load_vocab_size(banks_dir: Path) -> int:
    p = banks_dir / "prompt_vocab.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run 01_build_prompt_banks.py first.")
    mx = -1
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            mx = max(mx, int(r["prompt_id"]))
    return mx + 1

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class DiscreteSamplerGFN(nn.Module):
    """
    Forward policy π(a|s) parameterization via bilinear scoring:
      state_emb = MLP([beh_emb || state_numeric])
      logit = <state_emb, prompt_emb[a]> + bias[a]
    """
    def __init__(self, n_behaviors: int, n_prompts: int, beh_dim: int, prompt_dim: int, state_dim: int, hidden: int):
        super().__init__()
        self.beh_emb = nn.Embedding(n_behaviors, beh_dim)
        self.prompt_emb = nn.Embedding(n_prompts, prompt_dim)
        self.prompt_bias = nn.Embedding(n_prompts, 1)
        self.mlp = nn.Sequential(
            nn.Linear(beh_dim + state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, prompt_dim),
        )
        self.logZ = nn.Parameter(torch.tensor(0.0))

    def state_embed(self, beh_idx: torch.Tensor, state_num: torch.Tensor) -> torch.Tensor:
        b = self.beh_emb(beh_idx)  # [B, beh_dim]
        x = torch.cat([b, state_num], dim=-1)
        return self.mlp(x)         # [B, prompt_dim]

    def logits_for_actions(self, state_emb: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        # state_emb: [B, D], action_ids: [B, K]
        pe = self.prompt_emb(action_ids)          # [B, K, D]
        pb = self.prompt_bias(action_ids).squeeze(-1)  # [B, K]
        # dot product
        logits = (pe * state_emb.unsqueeze(1)).sum(-1) + pb
        return logits

@dataclass
class Episode:
    behavior_id: str
    beh_idx: int
    steps: List[Tuple[List[float], int]]  # (state_num, action_prompt_id)
    best_score: float

def parse_state(rec: dict) -> List[float]:
    # Minimal numeric state; keep it stable
    t = float(rec.get("t", 0))
    best = float(rec.get("best_score", 0.0))
    score = float(rec.get("score", 0.0))
    budget = rec.get("budget_per_try", None)
    if budget is None:
        budget_left = 0.0
    else:
        budget_left = float(max(0, int(budget) - int(rec.get("t", 0)) - 1))
    return [t, best, score, budget_left]

def iter_episodes(transitions_path: Path, *, start_offset: int = 0):
    """
    Streaming episode iterator from a transitions JSONL that contains episode_end markers.
    Returns tuples: (Episode dict-like, new_offset)
    """
    with open(transitions_path, "rb") as f:
        if start_offset:
            f.seek(start_offset)
        buf = []
        current_episode_id = None
        current_behavior_id = None
        best_score = 0.0
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                rec = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            if rec.get("episode_end"):
                # yield buffered episode
                if current_behavior_id and buf:
                    yield {
                        "behavior_id": current_behavior_id,
                        "steps": buf,
                        "best_score": float(rec.get("best_score", best_score)),
                    }, f.tell()
                buf = []
                current_episode_id = None
                current_behavior_id = None
                best_score = 0.0
                continue

            ep = rec.get("episode_id")
            beh = rec.get("behavior_id")
            pid = rec.get("prompt_id", None)
            if pid is None:
                continue  # skip missing prompt_id
            if current_episode_id is None:
                current_episode_id = ep
                current_behavior_id = beh
            # if no end markers (or malformed), detect episode switch
            if ep != current_episode_id:
                if current_behavior_id and buf:
                    yield {"behavior_id": current_behavior_id, "steps": buf, "best_score": best_score}, pos
                buf = []
                current_episode_id = ep
                current_behavior_id = beh
                best_score = 0.0

            state_num = parse_state(rec)
            buf.append((state_num, int(pid)))
            best_score = max(best_score, float(rec.get("best_score", 0.0)))

        # tail
        if current_behavior_id and buf:
            yield {"behavior_id": current_behavior_id, "steps": buf, "best_score": best_score}, f.tell()

def sample_negatives(allowed: List[int], pos: int, k: int) -> List[int]:
    if k <= 0:
        return []
    if not allowed:
        return []
    # if action set is tiny, sample with replacement but avoid pos when possible
    negs = []
    for _ in range(k):
        for _tries in range(10):
            x = random.choice(allowed)
            if x != pos or len(allowed) == 1:
                negs.append(x)
                break
        else:
            negs.append(pos)
    return negs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", required=True, help="Transitions JSONL from 02_extract_transitions_actions.py (with --emit-episode-end)")
    ap.add_argument("--banks-dir", required=True, help="Banks dir from 01_build_prompt_banks.py")
    ap.add_argument("--run-dir", required=True, help="Output run dir for checkpoints/logs")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta", type=float, default=5.0, help="Reward beta (exp(beta * best_score))")
    ap.add_argument("--negatives", type=int, default=128, help="Number of negatives per step (sampled from same behavior)")
    ap.add_argument("--batch-episodes", type=int, default=16, help="Episodes per optimizer step")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--beh-dim", type=int, default=64)
    ap.add_argument("--prompt-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--save-every", type=int, default=200, help="Save ckpt_latest every N optimizer steps")
    ap.add_argument("--save-steps", type=int, default=0, help="If >0, also save ckpt_stepXXXX every N steps")
    ap.add_argument("--max-optim-steps", type=int, default=0, help="If >0, stop after this many optimizer steps")
    ap.add_argument("--resume", action="store_true", help="Resume from run-dir/ckpt_latest.pt if exists")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    transitions_path = Path(args.transitions)
    banks_dir = Path(args.banks_dir)

    set_seed(args.seed)

    beh2prompts = load_behavior_to_prompts(banks_dir)
    # behavior indexing
    behaviors = sorted(beh2prompts.keys())
    beh2idx = {b:i for i,b in enumerate(behaviors)}
    n_beh = len(behaviors)

    n_prompts = load_vocab_size(banks_dir)

    state_dim = 4  # [t, best, score, budget_left]
    model = DiscreteSamplerGFN(
        n_behaviors=n_beh,
        n_prompts=n_prompts,
        beh_dim=args.beh_dim,
        prompt_dim=args.prompt_dim,
        state_dim=state_dim,
        hidden=args.hidden,
    )
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_latest = run_dir / "ckpt_latest.pt"
    state_path = run_dir / "state.json"

    start_offset = 0
    global_step = 0
    episodes_seen = 0

    if args.resume and ckpt_latest.exists():
        ckpt = torch.load(ckpt_latest, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        global_step = int(ckpt.get("global_step", 0))
        episodes_seen = int(ckpt.get("episodes_seen", 0))
        start_offset = int(ckpt.get("file_offset", 0))
        # restore RNG (optional)
        if "rng_state" in ckpt:
            random.setstate(ckpt["rng_state"]["py"])
            torch.random.set_rng_state(ckpt["rng_state"]["torch"])
            if torch.cuda.is_available() and ckpt["rng_state"].get("cuda") is not None:
                torch.cuda.random.set_rng_state_all(ckpt["rng_state"]["cuda"])
        print(f"[RESUME] step={global_step} episodes_seen={episodes_seen} offset={start_offset}")

    # write config once
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        atomic_write_json(cfg_path, vars(args))

    def save_ckpt(file_offset: int, *, step_name: Optional[str]=None):
        rng = {
            "py": random.getstate(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.random.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        payload = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "global_step": global_step,
            "episodes_seen": episodes_seen,
            "file_offset": int(file_offset),
            "rng_state": rng,
            "behaviors": behaviors,
        }
        tmp = run_dir / "ckpt_latest.pt.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, ckpt_latest)

        if step_name:
            step_path = run_dir / f"ckpt_{step_name}.pt"
            tmp2 = step_path.with_suffix(".pt.tmp")
            torch.save(payload, tmp2)
            os.replace(tmp2, step_path)

        atomic_write_json(state_path, {
            "global_step": global_step,
            "episodes_seen": episodes_seen,
            "file_offset": int(file_offset),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    # training loop (stream episodes)
    batch: List[Episode] = []
    last_offset = start_offset

    for ep_dict, new_offset in iter_episodes(transitions_path, start_offset=start_offset):
        last_offset = new_offset
        beh = str(ep_dict["behavior_id"])
        if beh not in beh2idx:
            continue
        beh_idx = beh2idx[beh]

        steps = []
        for state_num, pid in ep_dict["steps"]:
            # ensure pid is in behavior's action set; otherwise skip
            if pid not in set(beh2prompts[beh]):
                continue
            steps.append((state_num, pid))
        if not steps:
            continue

        batch.append(Episode(behavior_id=beh, beh_idx=beh_idx, steps=steps, best_score=float(ep_dict["best_score"])))

        if len(batch) < args.batch_episodes:
            continue

        # one optimizer step over batch episodes
        opt.zero_grad(set_to_none=True)
        losses = []
        for ep in batch:
            # terminal reward
            R = terminal_reward(ep.best_score, beta=args.beta)
            logR = math.log(max(R, 1e-12))

            # accumulate log_pf over steps
            log_pf = 0.0
            for state_num, pos_pid in ep.steps:
                allowed = beh2prompts[ep.behavior_id]
                negs = sample_negatives(allowed, pos_pid, args.negatives)
                cand = [pos_pid] + negs
                # build tensors
                state_t = torch.tensor(state_num, dtype=torch.float32, device=device).unsqueeze(0)  # [1, S]
                beh_t = torch.tensor([ep.beh_idx], dtype=torch.long, device=device)                   # [1]
                cand_t = torch.tensor([cand], dtype=torch.long, device=device)                        # [1, K]

                s_emb = model.state_embed(beh_t, state_t)                                            # [1, D]
                logits = model.logits_for_actions(s_emb, cand_t)                                     # [1, K]
                logp = F.log_softmax(logits, dim=-1)[0, 0]                                           # positive is at index 0
                log_pf = log_pf + logp

            # TB loss: (logZ + log_pf - logR)^2
            tb = (model.logZ + log_pf - logR) ** 2
            losses.append(tb)

        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        global_step += 1
        episodes_seen += len(batch)

        if global_step % 10 == 0:
            print(f"[STEP {global_step}] loss={float(loss.item()):.4f} logZ={float(model.logZ.item()):.3f} episodes_seen={episodes_seen}")

        if global_step % args.save_every == 0:
            save_ckpt(last_offset)

        if args.save_steps and args.save_steps > 0 and global_step % args.save_steps == 0:
            save_ckpt(last_offset, step_name=f"step{global_step:06d}")

        batch = []

        if args.max_optim_steps and args.max_optim_steps > 0 and global_step >= args.max_optim_steps:
            break

    # final save
    save_ckpt(last_offset, step_name="final")
    print(f"[OK] training finished. ckpt={ckpt_latest}")

if __name__ == "__main__":
    main()
