"""GPTQ 4-bit quantization of the merged model (for vLLM GPU serving).

GPTQ does layer-wise weight quantization that minimizes output error against a
calibration set. The output dir is a standard GPTQ checkpoint that vLLM serves
natively:

    vllm serve out/gptq-4bit --served-model-name qwen-sql --quantization gptq

Then score it with the usual OpenAI backend (see README serving matrix).

Run on a CUDA GPU. Uses `auto-gptq` (pip install auto-gptq optimum); the
maintained successor `gptqmodel` exposes a near-identical flow if you prefer it.

Usage:
    python -m textsql.export.quantize_gptq \
        --model out/merged-fp16 \
        --calib-data data/processed/train.jsonl \
        --out out/gptq-4bit
"""
from __future__ import annotations

import argparse
from pathlib import Path

from textsql.export.calibration import build_calibration_texts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Merged fp16 model dir or HF name")
    ap.add_argument("--calib-data", required=True, type=Path,
                    help="JSONL split to draw calibration prompts from (e.g. train)")
    ap.add_argument("--out", required=True, type=Path, help="Output dir for GPTQ model")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--desc-act", action="store_true",
                    help="Activation-order quantization: better accuracy, slower.")
    ap.add_argument("--n-samples", type=int, default=256, help="Calibration samples")
    ap.add_argument("--max-len", type=int, default=2048, help="Calibration seq length")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    texts = build_calibration_texts(
        args.calib_data, tokenizer, n_samples=args.n_samples, seed=args.seed
    )
    # auto-gptq wants tokenized examples ({input_ids, attention_mask} tensors).
    examples = [
        dict(tokenizer(t, return_tensors="pt", truncation=True, max_length=args.max_len))
        for t in texts
    ]
    print(f"Calibrating with {len(examples)} samples from {args.calib_data}")

    quantize_config = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=args.desc_act,
    )
    print(f"Loading {args.model} and quantizing to {args.bits}-bit GPTQ "
          f"(group_size={args.group_size}, desc_act={args.desc_act})...")
    model = AutoGPTQForCausalLM.from_pretrained(args.model, quantize_config)
    model.quantize(examples)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(args.out), use_safetensors=True)
    tokenizer.save_pretrained(args.out)
    print(f"Done. GPTQ model in {args.out}")
    print(f"Serve: vllm serve {args.out} --quantization gptq "
          f"--served-model-name qwen-sql --port 8000")


if __name__ == "__main__":
    main()
