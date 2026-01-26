from __future__ import annotations

import math
from typing import Any, Dict, Optional

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LlamaGuard3:
    """
    Compatibility wrapper for an always-on safety guard.

    Default model:
      ibm-granite/granite-guardian-3.1-2b

    GPU-only load:
      - no device_map
      - no offload
      - model.to("cuda") when device="cuda"
    """

    def __init__(
        self,
        model_id: str = "ibm-granite/granite-guardian-3.1-2b",
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
    ):
        self.model_id = model_id
        self.device = device

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

        if dtype is None:
            if device == "cuda" and torch.cuda.is_available():
                dtype = torch.bfloat16
            else:
                dtype = torch.float32

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
                raise RuntimeError("device='cuda' requested but CUDA is not available.")
            self.model.to("cuda")

        # Granite Guardian binary labels
        self.safe_token = "No"
        self.unsafe_token = "Yes"

        safe_ids = self.tokenizer.encode(self.safe_token, add_special_tokens=False)
        unsafe_ids = self.tokenizer.encode(self.unsafe_token, add_special_tokens=False)
        if len(safe_ids) != 1 or len(unsafe_ids) != 1:
            raise ValueError(f"Expected single-token labels. Got No={safe_ids}, Yes={unsafe_ids}.")

        self.safe_id = safe_ids[0]
        self.unsafe_id = unsafe_ids[0]

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _prompt(self, text: str) -> str:
        return (
            "You are a risk detection model. Determine whether the following content is risky/harmful.\n"
            "Answer with exactly one token: Yes (risky) or No (not risky).\n\n"
            f"CONTENT:\n{text}\n\nAnswer:"
        )

    @staticmethod
    def _softmax2(a: float, b: float) -> tuple[float, float]:
        m = max(a, b)
        ea = math.exp(a - m)
        eb = math.exp(b - m)
        z = ea + eb
        return ea / z, eb / z

    @torch.no_grad()
    def classify(self, text: str) -> Dict[str, Any]:
        prompt = self._prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        out = self.model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        logits = out.scores[0][0]  # [vocab]
        logit_no = float(logits[self.safe_id].item())
        logit_yes = float(logits[self.unsafe_id].item())

        p_no, p_yes = self._softmax2(logit_no, logit_yes)
        label = "unsafe" if p_yes > p_no else "safe"

        return {
            "label": label,
            "p_safe": p_no,
            "p_unsafe": p_yes,
            "raw_text": f"No={p_no:.4f}, Yes={p_yes:.4f}",
        }
