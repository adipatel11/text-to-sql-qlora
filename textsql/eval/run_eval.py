"""Generate predictions for a dataset and score execution accuracy.

Works against any backend (local HF or an OpenAI-compatible server), so the
identical command measures the base model, the fine-tuned fp16 model, and
every quantized/served variant.

Examples:
    # Local HF baseline (base model, greedy)
    python -m textsql.eval.run_eval --backend hf \
        --model Qwen/Qwen2.5-Coder-3B-Instruct \
        --data data/processed/dev.jsonl --limit 200 \
        --out-dir results/base_hf

    # Fine-tuned adapter, 4-bit
    python -m textsql.eval.run_eval --backend hf \
        --model Qwen/Qwen2.5-Coder-3B-Instruct --adapter out/qlora-spider \
        --load-in-4bit --data data/processed/dev.jsonl --out-dir results/ft_nf4

    # Against a running vLLM / llama.cpp server
    python -m textsql.eval.run_eval --backend openai \
        --base-url http://localhost:8000/v1 --model qwen-sql \
        --data data/processed/dev.jsonl --out-dir results/vllm_awq
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from textsql.backends import build_backend
from textsql.eval.execution import execution_match
from textsql.prompts import build_messages, extract_sql


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["hf", "openai"], default="hf")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (hf backend)")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8, help="hf backend")
    ap.add_argument("--workers", type=int, default=8, help="openai backend")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-query exec timeout")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    rows = load_jsonl(args.data, args.limit)
    print(f"Loaded {len(rows)} examples from {args.data}")

    # Rebuild messages from schema+question if not stored (keeps prompts pinned).
    messages_list = [
        r.get("messages") or build_messages(r["schema"], r["question"]) for r in rows
    ]

    backend = build_backend(args)
    print(f"Backend: {args.backend} ({args.model})  -- generating...")
    t0 = time.time()
    if args.backend == "hf":
        raw = backend.generate(messages_list, batch_size=args.batch_size)
    else:
        raw = backend.generate(messages_list, workers=args.workers)
    gen_secs = time.time() - t0
    print(f"Generation took {gen_secs:.1f}s ({len(rows) / gen_secs:.2f} ex/s)")

    statuses = Counter()
    correct = 0
    preds = []
    for r, completion in zip(rows, raw):
        pred_sql = extract_sql(completion)
        ok, status = execution_match(
            r["db_path"], r["gold_sql"], pred_sql, timeout=args.timeout
        )
        statuses[status.split(":")[0]] += 1
        correct += int(ok)
        preds.append(
            {
                "id": r["id"],
                "db_id": r["db_id"],
                "question": r["question"],
                "gold_sql": r["gold_sql"],
                "pred_sql": pred_sql,
                "raw": completion,
                "correct": ok,
                "status": status,
            }
        )

    n = len(rows)
    acc = correct / n if n else 0.0
    metrics = {
        "model": args.model,
        "adapter": args.adapter,
        "backend": args.backend,
        "n": n,
        "execution_accuracy": round(acc, 4),
        "correct": correct,
        "status_counts": dict(statuses),
        "gen_seconds": round(gen_secs, 2),
        "examples_per_sec": round(n / gen_secs, 3) if gen_secs else None,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(args.out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nExecution accuracy: {acc:.1%}  ({correct}/{n})")
    print(f"Wrote {args.out_dir}/predictions.jsonl and metrics.json")


if __name__ == "__main__":
    main()
