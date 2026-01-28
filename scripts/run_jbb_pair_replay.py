from __future__ import annotations

# --- make `import src...` work no matter how the script is launched ---
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import gc
import inspect
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv

# You will need: pip install pyyaml jailbreakbench
import yaml
import jailbreakbench as jbb
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.judges.llama3 import LlamaGuard3
from src.models.hf_chat import generate_completion
from src.utils.io import JsonlWriter, new_run_id

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# -------------------------
# helpers
# -------------------------
def parse_dtype(name: str) -> torch.dtype:
    name = (name or "").strip().lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def cuda_cleanup_strong() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def unload_model_strong(tok: Any, mdl: Any) -> None:
    try:
        del tok
    except Exception:
        pass
    try:
        del mdl
    except Exception:
        pass
    cuda_cleanup_strong()


def load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def init_guard_compat(model_id: str, device: str, hf_token: Optional[str], dtype: torch.dtype) -> Any:
    sig = inspect.signature(LlamaGuard3.__init__)
    params = sig.parameters

    kwargs: Dict[str, Any] = {}
    if "model_id" in params:
        kwargs["model_id"] = model_id
    elif "model" in params:
        kwargs["model"] = model_id

    if "device" in params:
        kwargs["device"] = device
    if "dtype" in params:
        kwargs["dtype"] = dtype

    if hf_token:
        if "hf_token" in params:
            kwargs["hf_token"] = hf_token
        elif "token" in params:
            kwargs["token"] = hf_token

    return LlamaGuard3(**kwargs)


def guard_classify_compat(guard: Any, prompt: str, response: str) -> Tuple[str, Any]:
    sig = inspect.signature(guard.classify)
    n_nonself = sum(1 for p in sig.parameters.values() if p.name != "self")

    if n_nonself <= 1:
        payload = f"[USER PROMPT]\n{prompt}\n\n[ASSISTANT RESPONSE]\n{response}"
        res = guard.classify(payload)
    else:
        res = guard.classify(prompt, response)

    if isinstance(res, tuple):
        if len(res) >= 2:
            return str(res[0]), res[1]
        if len(res) == 1:
            return str(res[0]), None
        return "unknown", None
    return str(res), None


def load_hf_causal_lm_resident(
    model_id: str,
    dtype: torch.dtype,
    device: str,
    hf_token: Optional[str],
    *,
    use_cache: bool,
    max_gpu_mem_util: float,
    offload_dir: str,
) -> Tuple[Any, Any, float]:
    t0 = time.perf_counter()

    tok = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        use_fast=True,
        trust_remote_code=True,
    )

    device_map = None
    max_memory = None
    offload_folder = None

    if device == "cuda" and torch.cuda.is_available():
        device_map = "balanced_low_0"
        total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cap_gib = max(1, int(total_gib * float(max_gpu_mem_util)))
        max_memory = {0: f"{cap_gib}GiB", "cpu": "256GiB"}
        offload_folder = offload_dir
        Path(offload_folder).mkdir(parents=True, exist_ok=True)

    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=dtype,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=offload_folder,
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    mdl.eval()
    try:
        mdl.config.use_cache = bool(use_cache)
        if hasattr(mdl, "generation_config") and mdl.generation_config is not None:
            mdl.generation_config.use_cache = bool(use_cache)
    except Exception:
        pass

    return tok, mdl, time.perf_counter() - t0


def build_target_messages(user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
    ]


def iter_manifest_subset(manifest_path: Path, split: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # support either key
            s = obj.get("subset") or obj.get("hf_split")
            if split in ("harmful", "benign") and s != split:
                continue
            rows.append(obj)
    return rows


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    art_cfg = cfg.get("artifact", {})
    models_cfg = cfg.get("models", {})
    gen_cfg = (cfg.get("generation", {}) or {}).get("target", {})
    sys_cfg = cfg.get("system", {})

    seed = int(run_cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)

    split = str(run_cfg.get("split", "harmful"))
    max_behaviors = int(run_cfg.get("max_behaviors", 0))

    shard_cfg = run_cfg.get("shard", {}) or {}
    num_shards = int(shard_cfg.get("num_shards", 1))
    shard_idx = int(shard_cfg.get("shard_idx", 0))

    manifest_path = Path(data_cfg.get("manifest_jsonl", "")).expanduser()
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest_jsonl not found: {manifest_path}")

    out_root = Path(run_cfg.get("out_root", str(ROOT))).expanduser()
    out_dir = out_root / "data" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = new_run_id(prefix="jbb_pair_replay")
    out_path = out_dir / f"{run_id}.jsonl"

    # HF token optional
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    device = str(sys_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype = parse_dtype(str(sys_cfg.get("dtype", "fp16")))
    use_cache = bool(sys_cfg.get("use_cache", True))
    max_gpu_mem_util = float(sys_cfg.get("max_gpu_mem_util", 0.9))
    offload_dir = str(sys_cfg.get("offload_dir", str(out_root / ".hf_offload")))
    fail_fast = bool(sys_cfg.get("fail_fast", False))

    target_model_id = str(models_cfg["target_model_id"])
    guard_model_id = str(models_cfg["guard_model_id"])

    # 1) load JBB behaviors (for labels/metadata)
    behaviors = iter_manifest_subset(manifest_path, split=split)
    if max_behaviors and max_behaviors > 0:
        behaviors = behaviors[:max_behaviors]

    total = len(behaviors)
    if num_shards <= 0:
        raise ValueError("num_shards must be >= 1")
    shard_size = (total + num_shards - 1) // num_shards
    start = shard_idx * shard_size
    end = min(total, start + shard_size)
    shard = behaviors[start:end]

    # 2) load official PAIR artifact prompts for the requested model
    provider = art_cfg.get("provider", "jailbreakbench")
    if provider != "jailbreakbench":
        raise ValueError("Only artifact.provider=jailbreakbench is supported in this script.")
    method = str(art_cfg.get("method", "PAIR"))
    model_name = str(art_cfg.get("model_name", "vicuna-13b-v1.5"))

    artifact = jbb.read_artifact(method=method, model_name=model_name)
    jailbreak_prompts = artifact.jailbreaks  # list[str]
    if len(jailbreak_prompts) < total:
        raise RuntimeError(
            f"Artifact has {len(jailbreak_prompts)} jailbreaks but manifest has {total} behaviors."
        )

    print(f"[RUN] {run_cfg.get('name','(unnamed)')}  run_id={run_id}")
    print(f"split={split} total={total} shard={shard_idx}/{num_shards} shard_n={len(shard)}")
    print(f"artifact: provider={provider} method={method} model_name={model_name}")
    print(f"target_model_id={target_model_id}")
    print(f"guard_model_id={guard_model_id}")
    print(f"out={out_path}")

    # 3) init guard (often keep on CPU to save VRAM)
    guard_device = "cpu" if device == "cuda" else device
    guard = init_guard_compat(guard_model_id, guard_device, hf_token, dtype)

    # 4) load target once (recommended; much faster than per-turn reload)
    tok_t = mdl_t = None
    try:
        tok_t, mdl_t, dt_load = load_hf_causal_lm_resident(
            target_model_id,
            dtype=dtype,
            device=device,
            hf_token=hf_token,
            use_cache=use_cache,
            max_gpu_mem_util=max_gpu_mem_util,
            offload_dir=offload_dir,
        )
        print(f"Loaded target in {dt_load:.2f}s")

        writer = JsonlWriter(out_path)

        for i, beh in enumerate(shard):
            # Map to global index (important to align with artifact ordering)
            global_idx = start + i
            jb_prompt = jailbreak_prompts[global_idx]

            goal = beh.get("goal") or beh.get("prompt") or ""
            behavior_id = beh.get("behavior_id") or beh.get("id") or ""
            category = beh.get("category") or ""
            subset = beh.get("subset") or beh.get("hf_split") or split

            timings: Dict[str, Any] = {}
            errors: Dict[str, Any] = {}

            # target generation
            try:
                msgs_t = build_target_messages(jb_prompt)
                t0 = time.perf_counter()
                _, t_comp, meta_t = generate_completion(
                    tok_t,
                    mdl_t,
                    msgs_t,
                    temperature=float(gen_cfg.get("temperature", 0.7)),
                    top_p=float(gen_cfg.get("top_p", 0.9)),
                    max_new_tokens=int(gen_cfg.get("max_new_tokens", 512)),
                )
                timings["t_gen_target"] = time.perf_counter() - t0
                target_response = (t_comp or "").strip()
            except Exception as e:
                errors["target"] = repr(e)
                target_response = ""
                meta_t = {}
                if fail_fast:
                    raise

            # guard
            guard_label = "unknown"
            guard_raw = None
            try:
                t0 = time.perf_counter()
                guard_label, guard_raw = guard_classify_compat(guard, jb_prompt, target_response)
                timings["t_guard"] = time.perf_counter() - t0
            except Exception as e:
                errors["guard"] = repr(e)
                if fail_fast:
                    raise

            record = {
                "run_id": run_id,
                "subset": subset,
                "category": category,
                "behavior_id": behavior_id,
                "behavior_index": global_idx,
                "goal": goal,
                "jbb_method": method,
                "jbb_model_name": model_name,
                "jailbreak_prompt": jb_prompt,
                "target_model_id": target_model_id,
                "target_response": target_response,
                "guard_model_id": guard_model_id,
                "guard_response_label": guard_label,
                "guard_response_raw": guard_raw,
                "timings": timings,
                "errors": errors,
                "raw": {"target_meta": meta_t},
            }
            writer.write(record)

            if (i + 1) % 10 == 0 or (i + 1) == len(shard):
                print(f"[PROGRESS] {i+1}/{len(shard)} guard={guard_label} err={bool(errors)}")

        writer.close()

    finally:
        if tok_t is not None:
            unload_model_strong(tok_t, mdl_t)

    print("[DONE]")


if __name__ == "__main__":
    main()
