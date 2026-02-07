# scripts/04_train_gfn.py
import argparse
import json
from collections import defaultdict
import random

import torch
from torch.optim import Adam
from src.gfn.policy import SimpleGFNPolicy


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_episodes(transitions_path: str):
    """
    Input: JSONL produced by patched 03_extract_transitions.py
    Output: dict episode_id -> list of transition dicts (sorted by t)
    """
    episodes = defaultdict(list)
    with open(transitions_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("episode_end"):
                continue
            eid = r["episode_id"]
            episodes[eid].append(r)

    # sort by t
    for eid in list(episodes.keys()):
        episodes[eid] = sorted(episodes[eid], key=lambda x: int(x["t"]))
        if len(episodes[eid]) == 0:
            del episodes[eid]
    return episodes


def make_features(tr):
    # Features: [t, score, best_score]
    return torch.tensor([float(tr["t"]), float(tr["score"]), float(tr["best_score"])], dtype=torch.float32)


def main(args):
    device = args.device
    set_seed(args.seed)

    episodes = load_episodes(args.transitions)
    eids = list(episodes.keys())
    if not eids:
        raise RuntimeError("No episodes loaded. Did you run patched 03_extract_transitions.py on run.py outputs?")

    model = SimpleGFNPolicy(input_dim=3, hidden_dim=args.hidden_dim).to(device)

    # Train logZ as a parameter (common in TB)
    log_z = torch.zeros(1, requires_grad=True, device=device)

    opt = Adam(model.parameters(), lr=args.lr)
    opt_z = Adam([log_z], lr=args.lr)

    def episode_log_pf(ep):
        # Sum model outputs across transitions in the episode
        xs = torch.stack([make_features(tr) for tr in ep], dim=0).to(device)
        return model(xs).sum()

    def episode_log_r(ep):
        # terminal reward based on terminal best_score
        best = float(ep[-1]["best_score"])
        # logR = beta * best
        return torch.tensor(args.beta * best, dtype=torch.float32, device=device)

    best_loss = float("inf")
    plateau = 0

    for step in range(1, args.max_updates + 1):
        # sample minibatch of episodes
        batch_eids = random.sample(eids, k=min(args.batch_size, len(eids)))

        losses = []
        for eid in batch_eids:
            ep = episodes[eid]
            log_pf = episode_log_pf(ep)
            log_r = episode_log_r(ep)
            losses.append((log_z + log_pf - log_r) ** 2)

        loss = torch.stack(losses).mean()

        opt.zero_grad()
        opt_z.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt_z.step()

        if step % args.log_every == 0:
            cur = float(loss.item())
            print(f"[GFN-TB] step={step} loss={cur:.6f} logZ={float(log_z.item()):.4f} episodes={len(eids)}")

            # simple early stop
            if cur < best_loss * 0.995:
                best_loss = cur
                plateau = 0
            else:
                plateau += 1
            if plateau >= args.plateau_patience:
                print(f"[GFN-TB] Early stop at step {step} (plateau)")
                break

    # Save both model and logZ
    payload = {
        "model_state": model.state_dict(),
        "log_z": float(log_z.detach().cpu().item()),
        "args": vars(args),
    }
    torch.save(payload, args.out)
    print("saved ->", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=5.0, help="logR = beta * best_score")
    ap.add_argument("--max-updates", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=64, help="Minibatch over episodes")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--plateau-patience", type=int, default=10)
    main(ap.parse_args())
