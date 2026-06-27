"""QLoRA fine-tuning for text-to-SQL.

4-bit NF4 base weights (bitsandbytes) + LoRA adapters trained on the
completion only (the prompt tokens are masked with -100 so the model isn't
penalized for the schema/question it was given).

Uses transformers.Trainer directly + a tiny custom collator rather than TRL,
so it is robust across library versions and easy to read.

Config comes from a YAML file; any field can be overridden on the CLI, e.g.
    python -m textsql.train.qlora_train --config configs/qlora_qwen3b.yaml \
        --set lora.r=32 train.learning_rate=1e-4

Requires a CUDA GPU (bitsandbytes 4-bit). On a 16-24 GB card a 3B model fits.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def _set_nested(d: dict, dotted: str, value: str) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    # Best-effort type coercion from the CLI string.
    for cast in (int, float):
        try:
            value = cast(value)  # type: ignore[assignment]
            break
        except (ValueError, TypeError):
            continue
    else:
        if value in ("true", "false"):
            value = value == "true"  # type: ignore[assignment]
    cur[keys[-1]] = value


def load_config(path: str, overrides: list[str] | None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        _set_nested(cfg, key, val)
    return cfg


# --------------------------------------------------------------------------- #
# Dataset: render chat template, mask the prompt tokens
# --------------------------------------------------------------------------- #
def build_dataset(jsonl_path: str, tokenizer, max_len: int):
    from datasets import load_dataset

    ds = load_dataset("json", data_files=jsonl_path, split="train")

    def encode(example):
        messages = example["messages"]
        # The completion is the gold SQL as an assistant turn.
        full = messages + [{"role": "assistant", "content": example["gold_sql"]}]
        # Render to *text*, then tokenize to a plain list[int]. Going straight to
        # tokenize=True is not portable: on some transformers builds it returns a
        # BatchEncoding, and `[:max_len]` then slices its internal Encoding
        # objects instead of token ids -- which blow up only later when datasets
        # tries to serialize them to Arrow ("could not convert Encoding...").
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        full_text = tokenizer.apply_chat_template(
            full, add_generation_prompt=False, tokenize=False
        )
        # The template already emits the special tokens as text, so don't add more.
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        full_ids = full_ids[:max_len]
        labels = list(full_ids)
        # Mask everything that belongs to the prompt.
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        return {"input_ids": full_ids, "labels": labels}

    return ds.map(encode, remove_columns=ds.column_names)


@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, features):
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            ids = f["input_ids"]
            lab = f["labels"]
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", dest="overrides", default=[],
                    help="dotted overrides, e.g. lora.r=32")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    print(json.dumps(cfg, indent=2))

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    model_name = cfg["model"]["name"]
    max_len = cfg["data"]["max_seq_len"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=getattr(torch, cfg["model"].get("compute_dtype", "bfloat16")),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto",
        torch_dtype=getattr(torch, cfg["model"].get("compute_dtype", "bfloat16")),
    )
    model.config.use_cache = False  # required with gradient checkpointing
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["train"].get("gradient_checkpointing", True)
    )

    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = build_dataset(cfg["data"]["train_file"], tokenizer, max_len)
    print(f"Training examples: {len(train_ds)}")

    targs = TrainingArguments(
        output_dir=cfg["train"]["output_dir"],
        per_device_train_batch_size=cfg["train"]["batch_size"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=float(cfg["train"]["learning_rate"]),
        num_train_epochs=cfg["train"].get("epochs", 3),
        max_steps=cfg["train"].get("max_steps", -1),
        lr_scheduler_type=cfg["train"].get("lr_scheduler", "cosine"),
        warmup_ratio=cfg["train"].get("warmup_ratio", 0.03),
        logging_steps=cfg["train"].get("logging_steps", 10),
        save_steps=cfg["train"].get("save_steps", 200),
        save_total_limit=cfg["train"].get("save_total_limit", 2),
        bf16=cfg["model"].get("compute_dtype", "bfloat16") == "bfloat16",
        fp16=cfg["model"].get("compute_dtype") == "float16",
        gradient_checkpointing=cfg["train"].get("gradient_checkpointing", True),
        optim=cfg["train"].get("optim", "paged_adamw_8bit"),
        report_to=cfg["train"].get("report_to", "none"),
        seed=cfg.get("seed", 42),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    trainer.train()

    out = cfg["train"]["output_dir"]
    trainer.model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"Saved LoRA adapter + tokenizer to {out}")


if __name__ == "__main__":
    main()
