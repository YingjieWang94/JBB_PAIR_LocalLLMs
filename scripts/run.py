from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.io import JsonlWriter, new_run_id
from src.judges.llama3 import LlamaGuard3
from src.models.hf_chat import generate_completion
from src.prompts.escalation import (
    build_escalation_judge_messages,
    parse_escalation_json,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"


def load_hf_causal_lm_resident(
    model_id: str,
    hf_token: Optional[str],
    dtype: torch.dtype,
    device: str,
):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        use_fast=True,
        trust_remote_code=True,
    )
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    mdl.eval()
    return tok, mdl, time.time() - t0


def apply_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.profile == "server":
        args.device = args.device or "cuda"
        args.guard_device = args.guard_device or "cuda"
        args.escalation_device = args.escalation_device or "cuda"
        args.max_behaviors = args.max_behaviors if args.max_behaviors is not None else 20
        args.budget_per_try = args.budget_per_try if args.budget_per_try is not None else 10
    return args


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()

    parser.add_argument("--profile", type=str, default="server")
    parser.add_argument("--subset", type=str, default="harmful")

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--guard-device", type=str, default=None)
    parser.add_argument("--escalation-device", type=str, default=None)

    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--max-behaviors", type=int, default=None)
    parser.add_argument("--budget-per-try", type=int, default=None)
    parser.add_argument("--use-cache", type=lambda x: x.lower() == "true", default=True)

    # Sharding
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-idx", type=int, default=0)

    # Models
    parser.add_argument("--target-model-id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--attacker-model-id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--guard-model-id", type=str, default="meta-llama/Llama-Guard-3-8B")

    parser.add_argument("--enable-escalation", action="store_true")
    parser.add_argument("--escalation-model-id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--escalation-temp", type=float, default=0.0)
    parser.add_argument("--escalation-top-p", type=float, default=1.0)

    args = parser.parse_args()
    args = apply_profile_defaults(args)

    if args.num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= args.shard_idx < args.num_shards):
        raise ValueError("shard_idx must be in [0, num_shards)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    hf_token = os.getenv("HF_TOKEN")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print("[LOAD] target:", args.target_model_id)
    tok_t, mdl_t, _ = load_hf_causal_lm_resident(args.target_model_id, hf_token, dtype, args.device)

    print("[LOAD] attacker:", args.attacker_model_id)
    tok_a, mdl_a, _ = load_hf_causal_lm_resident(args.attacker_model_id, hf_token, dtype, args.device)

    print("[LOAD] guard:", args.guard_model_id)
    guard = LlamaGuard3(
        model_id=args.guard_model_id,
        device=args.guard_device,
        dtype=dtype,
        token=hf_token,
    )

    tok_e = mdl_e = None
    if args.enable_escalation:
        print("[LOAD] escalation:", args.escalation_model_id)
        tok_e, mdl_e, _ = load_hf_causal_lm_resident(
            args.escalation_model_id,
            hf_token,
            dtype,
            args.escalation_device,
        )

    behaviors_path = DATA_DIR / "processed" / f"{args.subset}.jsonl"
    behaviors = [json.loads(l) for l in open(behaviors_path, "r", encoding="utf-8")]

    max_beh = min(len(behaviors), args.max_behaviors) if args.max_behaviors else len(behaviors)
    shard_indices = [i for i in range(max_beh) if i % args.num_shards == args.shard_idx]

    if not shard_indices:
        print(f"[SHARD] {args.shard_idx}/{args.num_shards}: nothing to do")
        return

    run_id = new_run_id()
    ts = time.strftime("%Y%m%d_%H%M%S")
    shard_tag = f"shard{args.shard_idx}-of{args.num_shards}"

    out_path = GENERATED_DIR / f"pair_{args.profile}_{shard_tag}_{ts}_{run_id}.jsonl"
    ckpt_path = GENERATED_DIR / f"pair_{args.profile}_{shard_tag}_{ts}_{run_id}.ckpt.json"

    writer = JsonlWriter(out_path)
    ckpt = {"completed": []}

    print(f"[RUN] shard {args.shard_idx}/{args.num_shards}: {len(shard_indices)} behaviors")

    for b_idx in shard_indices:
        beh = behaviors[b_idx]
        goal = beh["goal"]

        for step in range(args.budget_per_try):
            attacker_prompt = goal
            _, target_resp, _ = generate_completion(tok_t, mdl_t, attacker_prompt)

            guard_res = guard.classify(target_resp)
            unsafe = guard_res.get("label") == "unsafe"

            esc_json = None
            if unsafe and args.enable_escalation:
                msgs = build_escalation_judge_messages(goal, attacker_prompt, target_resp)
                _, esc_text, _ = generate_completion(
                    tok_e,
                    mdl_e,
                    msgs,
                    temperature=args.escalation_temp,
                    top_p=args.escalation_top_p,
                )
                esc_json = parse_escalation_json(esc_text)

            record = {
                "behavior_id": beh["behavior_id"],
                "goal": goal,
                "response": target_resp,
                "unsafe": unsafe,
                "escalation": esc_json,
            }
            writer.write(record)

        ckpt["completed"].append(b_idx)
        with open(ckpt_path, "w") as f:
            json.dump(ckpt, f)

    writer.close()
    print("Generated files:")
    print(out_path)
    print(ckpt_path)


if __name__ == "__main__":
    main()
