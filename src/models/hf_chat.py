from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        model_id: str,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
        token: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device

        if token is None:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")




        if dtype is None:
            dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_available()) else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            token=token,
            trust_remote_code=trust_remote_code,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if (device == "cuda" and torch.cuda.is_available()) else None,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        # Some Llama-like models need a pad token defined
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _render_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Prefer tokenizer chat template if present; otherwise fall back.
        messages: [{"role":"system|user|assistant", "content": "..."}]
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Fallback format
        chunks = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            chunks.append(f"{role}: {content}")
        chunks.append("ASSISTANT:")
        return "\n".join(chunks)

    @torch.no_grad()
    def generate(self, messages: List[Dict[str, str]], gen: GenConfig) -> Dict[str, Any]:
        prompt = self._render_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = gen.temperature is not None and gen.temperature > 1e-6

        out = self.model.generate(
            **inputs,
            max_new_tokens=gen.max_new_tokens,
            do_sample=do_sample,
            temperature=gen.temperature if do_sample else None,
            top_p=gen.top_p if do_sample else None,
            repetition_penalty=gen.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Try to strip the prompt prefix if it matches
        completion = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()

        return {
            "prompt": prompt,
            "text": completion,
        }

# ---------------------------------------------------------------------
# Backwards-compatible helper (used by scripts/run.py in older versions)
# ---------------------------------------------------------------------
@torch.no_grad()
def generate_completion(
    tokenizer,
    model,
    prompt_or_messages,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
    repetition_penalty: float = 1.0,
) -> tuple[str, str, dict]:
    """
    Compatibility shim for older runner code.

    Accepts either:
      - prompt_or_messages: str
      - prompt_or_messages: List[{"role": "...", "content": "..."}]

    Returns: (rendered_prompt, completion_text, meta)
    """

    # Render chat messages if a list is provided
    if isinstance(prompt_or_messages, str):
        prompt = prompt_or_messages
    else:
        messages = prompt_or_messages
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                # Fallback format
                chunks = []
                for m in messages:
                    role = m.get("role", "user").upper()
                    content = m.get("content", "")
                    chunks.append(f"{role}: {content}")
                chunks.append("ASSISTANT:")
                prompt = "\n".join(chunks)
        else:
            chunks = []
            for m in messages:
                role = m.get("role", "user").upper()
                content = m.get("content", "")
                chunks.append(f"{role}: {content}")
            chunks.append("ASSISTANT:")
            prompt = "\n".join(chunks)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature is not None and temperature > 1e-6

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    completion = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()

    meta = {
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }
    return prompt, completion, meta
