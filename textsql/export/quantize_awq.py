"""AWQ 4-bit quantization of the merged model (for vLLM GPU serving).

AWQ (Activation-aware Weight Quantization) protects the most salient weight
channels using activation statistics from a calibration set, and typically
serves faster than GPTQ on vLLM. The output dir is served natively:

    vllm serve out/awq-4bit --served-model-name qwen-sql --quantization awq

Run on a CUDA GPU. Uses `autoawq` (pip install autoawq).

Usage:
    python -m textsql.export.quantize_awq \
        --model out/merged-fp16 \
        --calib-data data/processed/train.jsonl \
        --out out/awq-4bit
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
    ap.add_argument("--out", required=True, type=Path, help="Output dir for AWQ model")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--n-samples", type=int, default=256, help="Calibration samples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    texts = build_calibration_texts(
        args.calib_data, tokenizer, n_samples=args.n_samples, seed=args.seed
    )
    print(f"Calibrating with {len(texts)} samples from {args.calib_data}")

    quant_config = {
        "zero_point": True,
        "q_group_size": args.group_size,
        "w_bit": args.bits,
        "version": "GEMM",
    }
    print(f"Loading {args.model} and quantizing to {args.bits}-bit AWQ "
          f"(group_size={args.group_size})...")
    model = AutoAWQForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True)
    model.quantize(tokenizer, quant_config=quant_config, calib_data=texts)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(args.out))
    tokenizer.save_pretrained(args.out)
    print(f"Done. AWQ model in {args.out}")
    print(f"Serve: vllm serve {args.out} --quantization awq "
          f"--served-model-name qwen-sql --port 8000")


if __name__ == "__main__":
    main()
