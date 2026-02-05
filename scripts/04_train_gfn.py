# scripts/04_train_gfn.py
import argparse
import torch
import json
from src.gfn.policy import SimpleGFNPolicy
from src.gfn.train_tb import train_gfn


def input_fn(batch):
    feats = []
    for b in batch:
        feats.append([b["t"], b["score"], b["best_score"]])
    return torch.tensor(feats, dtype=torch.float32)


def main(args):
    transitions = []
    with open(args.transitions, "r") as f:
        for line in f:
            transitions.append(json.loads(line))

    model = SimpleGFNPolicy(input_dim=3)

    train_gfn(
        model,
        transitions,
        input_fn=input_fn,
        max_updates=args.max_updates,
    )

    torch.save(model.state_dict(), args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-updates", type=int, default=100000)
    main(ap.parse_args())
