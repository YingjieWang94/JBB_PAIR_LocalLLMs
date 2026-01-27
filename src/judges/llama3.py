from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.hf_paths import resolve_local_model, local_only_enabled


@dataclass
class GuardResult:
    label: str
    p_safe: float
    p_unsafe: float
    raw_text: str


class LlamaGuard3:
    """
    Local Llama-Guard-3-* classifier.

    Implements logit scoring on the *next token* for "safe" vs "unsafe".
    This avoids free-form generations and is fast/deterministic.

    Returns:
      - label in {"safe","unsafe"}
      - p_safe, p_unsafe
      - raw_text (debug string)
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-Guard-3-8B",
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        token: Optional[str] = None,
        trust_remote_code: bool = True,
        threshold: float = 0.5,
    ):
        
        model_id = resolve_local_model(model_id)
        local_only = local_only_enabled()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map={"": self.device} if self.device != "cpu" else None,
            trust_remote_code=True,
            local_files_only=local_only,
        )

        
        self.model_id = model_id
        self.device = device
        self.threshold = float(threshold)

        if dtype is None:
            dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_available()) else torch.float32

        # Token handling: explicit token > env fallback
        if token is None:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("device=cuda requested but CUDA is not available.")
            self.model.to("cuda")
        else:
            self.model.to("cpu")

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cache token ids for "safe" and "unsafe"
        self.safe_id = self._single_token_id("safe")
        self.unsafe_id = self._single_token_id("unsafe")

    def _single_token_id(self, s: str) -> int:
        ids = self.tokenizer.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f'Expected "{s}" to be a single token for model {self.model_id}, got ids={ids}'
            )
        return ids[0]

    def _build_prompt(self, text: str) -> str:
        # Prefer chat template if available.
        messages = [
            {"role": "system", "content": "You are a safety classifier. Reply with a single token: safe or unsafe."},
            {"role": "user", "content": text},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return f"SYSTEM: {messages[0]['content']}\nUSER: {messages[1]['content']}\nASSISTANT:"

    @torch.no_grad()
    def classify(self, text: str) -> Dict[str, Any]:
        prompt = self._build_prompt(text or "")
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        out = self.model(**inputs)
        logits = out.logits[0, -1, :]  # next-token logits

        two = torch.stack([logits[self.safe_id], logits[self.unsafe_id]], dim=0)
        probs = torch.softmax(two.float(), dim=0)

        p_safe = float(probs[0].item())
        p_unsafe = float(probs[1].item())
        label = "unsafe" if p_unsafe >= self.threshold else "safe"

        raw_text = f"p_safe={p_safe:.6f}, p_unsafe={p_unsafe:.6f}, threshold={self.threshold:.3f}"

        return {
            "label": label,
            "p_safe": p_safe,
            "p_unsafe": p_unsafe,
            "raw_text": raw_text,
        }
