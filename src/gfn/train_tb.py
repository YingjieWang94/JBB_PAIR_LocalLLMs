# src/gfn/train_tb.py
import torch
from torch.optim import Adam
from typing import List, Dict
from .reward import terminal_reward


def trajectory_balance_loss(
    log_pf: torch.Tensor,
    log_z: torch.Tensor,
    reward: torch.Tensor,
):
    """
    TB loss: (log Z + sum log p_F - log R)^2
    """
    return ((log_z + log_pf - torch.log(reward + 1e-8)) ** 2).mean()


def train_gfn(
    model,
    transitions: List[Dict],
    input_fn,
    lr=1e-4,
    max_updates=100_000,
    eval_every=2000,
    plateau_patience=6,
    device="cpu",
):
    model.to(device)
    opt = Adam(model.parameters(), lr=lr)
    log_z = torch.zeros(1, requires_grad=True, device=device)
    opt_z = Adam([log_z], lr=lr)

    best_loss = float("inf")
    plateau = 0

    for step in range(1, max_updates + 1):
        batch = transitions
        x = input_fn(batch).to(device)
        log_pf = model(x).sum()

        best_scores = torch.tensor(
            [b["best_score"] for b in batch], device=device
        )
        rewards = torch.exp(5.0 * best_scores)

        loss = trajectory_balance_loss(log_pf, log_z, rewards)

        opt.zero_grad()
        opt_z.zero_grad()
        loss.backward()
        opt.step()
        opt_z.step()

        if step % eval_every == 0:
            cur = loss.item()
            if cur < best_loss * 0.99:
                best_loss = cur
                plateau = 0
            else:
                plateau += 1

            if plateau >= plateau_patience:
                print(f"[GFN] Early stop at step {step}")
                break
