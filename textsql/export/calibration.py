"""Calibration samples for post-training quantization (GPTQ / AWQ).

GPTQ and AWQ both need a small set of representative forward passes to estimate
activation statistics and choose quantization scales. Quality is best when the
calibration text matches the *serving* distribution, so we draw real Spider
prompts and render them with the exact pinned chat template the model was
trained and is evaluated on (via `textsql.prompts`). Both quantizers consume
the same texts, so GPTQ and AWQ are calibrated identically and the comparison
is apples-to-apples.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from textsql.prompts import build_messages


def build_calibration_texts(
    data_path: str | Path,
    tokenizer,
    n_samples: int = 256,
    seed: int = 42,
) -> list[str]:
    """Return `n_samples` rendered prompt strings drawn from a JSONL split.

    Each string is the full chat-templated prompt (system + user) with the
    assistant generation prompt appended -- i.e. exactly what the model sees at
    inference time, minus the completion. Sampling is seeded so calibration is
    reproducible across runs and across the two quantizers.
    """
    rows: list[dict] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No examples found in {data_path}")

    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n_samples]

    texts: list[str] = []
    for r in rows:
        messages = r.get("messages") or build_messages(r["schema"], r["question"])
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return texts
