from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os
import torch
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.hf_paths import resolve_local_model, local_only_enabled


@dataclass
class GenConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 256
    repetition_penalty: float = 1.0


class HFChatModel:
    """
    Minimal HF chat model wrapper supporting:
      - gated models via HF_TOKEN env var
      - chat templates when available
      - fallback to a simple concatenated format otherwise
    """

    def __init__(
        self,
        model_id_or_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        hf_token: Optional[str] = None,
    ) -> None:
        self.model_id_or_path = model_id_or_path
        self.dtype = dtype
        self.device = device
        self.hf_token = hf_token

        # Respect "local-only" mode if enabled by your project utilities.
        model_resolved = resolve_local_model(model_id_or_path)
        local_files_only = local_only_enabled()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_resolved,
            local_files_only=local_files_only,
            token=hf_token or os.environ.get("HF_TOKEN"),
            use_fast=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_resolved,
            local_files_only=local_files_only,
            token=hf_token or os.environ.get("HF_TOKEN"),
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )

        self.model.eval()

    def _render_messages(self, messages: List[Dict[str, str]]) -> str:
        # Render using chat template if possible
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        # Fallback: simple concatenation
        chunks: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            chunks.append(f"[{role.upper()}]\n{content}")
        chunks.append("[ASSISTANT]\n")
        return "\n\n".join(chunks)

    @torch.no_grad()
    def generate(self, messages: List[Dict[str, str]], gen: GenConfig) -> Dict[str, Any]:
        prompt = self._render_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        uses_device_map = hasattr(self.model, "hf_device_map") and isinstance(getattr(self.model, "hf_device_map"), dict)
        if not uses_device_map:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            # For sharded/offloaded models (accelerate device_map/offload):
            # place inputs on the same device as the embedding table to avoid CPU/CUDA mismatches.
            try:
                emb_device = self.model.get_input_embeddings().weight.device
            except Exception:
                emb_device = next(self.model.parameters()).device

            inputs = {k: v.to(emb_device) for k, v in inputs.items()}

            warnings.filterwarnings(
                "ignore",
                message=r"You are calling \.generate\(\) with the `input_ids` being on a device type different than your model's device\.",
                category=UserWarning,
            )

        do_sample = gen.temperature is not None and gen.temperature > 1e-6

        out = self.model.generate(
            **inputs,
            max_new_tokens=gen.max_new_tokens,
            do_sample=do_sample,
            temperature=gen.temperature if do_sample else None,
            top_p=gen.top_p if do_sample else None,
            repetition_penalty=gen.repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        comp = self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        return {
            "prompt": prompt,
            "completion": comp,
            "meta": {},
        }


@torch.no_grad()
def generate_completion(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
    repetition_penalty: float = 1.0,
) -> tuple[str, str, Dict[str, Any]]:
    # Render prompt using chat template when possible
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = None
    else:
        prompt = None

    if prompt is None:
        chunks: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            chunks.append(f"[{role.upper()}]\n{content}")
        chunks.append("[ASSISTANT]\n")
        prompt = "\n\n".join(chunks)

    inputs = tokenizer(prompt, return_tensors="pt")
    uses_device_map = hasattr(model, "hf_device_map") and isinstance(getattr(model, "hf_device_map"), dict)
    if not uses_device_map:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        # For sharded/offloaded models (accelerate device_map/offload):
        # place inputs on the same device as the embedding table to avoid CPU/CUDA mismatches.
        try:
            emb_device = model.get_input_embeddings().weight.device
        except Exception:
            emb_device = next(model.parameters()).device

        inputs = {k: v.to(emb_device) for k, v in inputs.items()}

        warnings.filterwarnings(
            "ignore",
            message=r"You are calling \.generate\(\) with the `input_ids` being on a device type different than your model's device\.",
            category=UserWarning,
        )

    do_sample = temperature is not None and temperature > 1e-6

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )

    comp = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    meta: Dict[str, Any] = {}
    return prompt, comp, meta
