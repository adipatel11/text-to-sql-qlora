"""Generation backends behind one interface.

A backend takes a list of chat-message lists and returns completion strings.
Because eval and the latency rig only depend on this interface, the exact same
code measures every point in the quantization x serving matrix:

  * HFBackend     - local transformers (fp16 / bitsandbytes 4-bit / 8-bit).
                    Use for the quick baseline and for the bnb-NF4 numbers.
  * OpenAIBackend - any OpenAI-compatible /chat/completions server. This is
                    how we hit vLLM (GPTQ/AWQ/fp16), llama.cpp (GGUF), TGI, etc.
                    Start the server separately, point --base-url at it.

Greedy decoding (temperature 0) is the default so execution accuracy is
deterministic and reproducible.
"""
from __future__ import annotations

import os
from typing import Optional


class HFBackend:
    """Local generation with HuggingFace transformers."""

    def __init__(
        self,
        model: str,
        adapter: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        device: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding is required for correct batched generation.
        self.tokenizer.padding_side = "left"

        quant_config = None
        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=getattr(torch, dtype),
            device_map=device,
            quantization_config=quant_config,
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(self, messages_list: list[list[dict]], batch_size: int = 8) -> list[str]:
        import torch

        outputs: list[str] = []
        for start in range(0, len(messages_list), batch_size):
            batch = messages_list[start : start + batch_size]
            prompts = [
                self.tokenizer.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in batch
            ]
            enc = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            # Strip the prompt tokens; decode only the completion.
            gen = gen[:, enc["input_ids"].shape[1] :]
            outputs.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
        return outputs


class OpenAIBackend:
    """Hit any OpenAI-compatible /chat/completions endpoint (vLLM, llama.cpp...)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        max_new_tokens: int = 256,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout

    def _one(self, messages: list[dict]) -> str:
        import requests

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": self.max_new_tokens,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def generate(self, messages_list: list[list[dict]], workers: int = 8) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(self._one, messages_list))


def build_backend(args) -> "HFBackend | OpenAIBackend":
    """Construct a backend from parsed argparse args (see run_eval / latency)."""
    if args.backend == "hf":
        return HFBackend(
            model=args.model,
            adapter=getattr(args, "adapter", None),
            load_in_4bit=getattr(args, "load_in_4bit", False),
            load_in_8bit=getattr(args, "load_in_8bit", False),
            dtype=getattr(args, "dtype", "bfloat16"),
            max_new_tokens=args.max_new_tokens,
        )
    if args.backend == "openai":
        return OpenAIBackend(
            base_url=args.base_url,
            model=args.model,
            max_new_tokens=args.max_new_tokens,
        )
    raise ValueError(f"unknown backend: {args.backend}")
