"""Merge a QLoRA adapter into the base weights -> a standalone fp16 model.

Everything downstream in the serving matrix consumes a plain, full-precision
HuggingFace checkpoint, not a base-model + adapter pair:

  * GPTQ / AWQ quantizers re-quantize from full-precision weights.
  * llama.cpp's GGUF converter expects a single merged checkpoint.
  * vLLM can serve the merged fp16 model directly as the baseline.

Critical detail: the base model is loaded in **full precision** (fp16/bf16),
*not* 4-bit. Merging a LoRA delta into already-quantized NF4 weights would bake
in the quantization error; we want a clean fp16 model that each method then
quantizes itself. (The training-time NF4 was only to fit the optimizer on a
small GPU.) Merging is pure matmul, so it runs fine on CPU.

Usage:
    python -m textsql.export.merge_lora \
        --base Qwen/Qwen2.5-Coder-3B-Instruct \
        --adapter out/qlora-spider \
        --out out/merged-fp16
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="Base model name or path")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir (from training)")
    ap.add_argument("--out", required=True, type=Path, help="Output dir for merged model")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"],
                    help="Merge/save dtype. float16 is the safest for broad GPU + "
                         "llama.cpp support; use bfloat16 to match training exactly.")
    ap.add_argument("--device", default="cpu",
                    help="Device to merge on. cpu avoids GPU OOM and is plenty fast "
                         "for a one-time merge; use 'auto' to use the GPU.")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    print(f"Loading base model {args.base} in {args.dtype} (full precision)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map=args.device,
    )

    print(f"Attaching adapter {args.adapter} and merging...")
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()  # fold LoRA deltas into the base weights

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model -> {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)

    # Save the tokenizer the model will actually be served with. Prefer the
    # adapter dir (training may have added pad/special tokens), else the base.
    tok_src = args.adapter if (Path(args.adapter) / "tokenizer_config.json").exists() else args.base
    AutoTokenizer.from_pretrained(tok_src).save_pretrained(args.out)

    print(f"Done. Merged fp16 model + tokenizer in {args.out}")
    print("Next: quantize it (textsql.export.quantize_gptq / quantize_awq / "
          "scripts/export_gguf.sh) or serve it directly with vLLM.")


if __name__ == "__main__":
    main()
