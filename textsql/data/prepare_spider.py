"""Turn the raw Spider release into train/dev JSONL the rest of the repo uses.

Each output row is self-contained:
    {
      "id": int,
      "db_id": str,
      "db_path": str,           # absolute path to the .sqlite file
      "question": str,
      "gold_sql": str,          # reference query
      "schema": str,            # CREATE TABLE ... for the db
      "messages": [...],        # chat-format prompt (system + user)
    }

Storing `messages` here guarantees train and eval see the identical prompt.

Usage:
    python -m textsql.data.prepare_spider \
        --spider-dir data/spider --out-dir data/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from textsql.prompts import build_messages
from textsql.schema import get_schema


def _db_path(spider_dir: Path, db_id: str, test: bool = False) -> Path:
    sub = "test_database" if test else "database"
    return (spider_dir / sub / db_id / f"{db_id}.sqlite").resolve()


def _load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_split(spider_dir: Path, source_files: list[str], test: bool) -> list[dict]:
    rows: list[dict] = []
    idx = 0
    skipped = 0
    for fname in source_files:
        fpath = spider_dir / fname
        if not fpath.exists():
            print(f"  [warn] missing {fpath}, skipping")
            continue
        for ex in _load_json(fpath):
            db_id = ex["db_id"]
            db_path = _db_path(spider_dir, db_id, test=test)
            if not db_path.exists():
                skipped += 1
                continue
            try:
                schema = get_schema(str(db_path))
            except Exception as e:  # noqa: BLE001 - report and skip bad dbs
                print(f"  [warn] schema failed for {db_id}: {e}")
                skipped += 1
                continue
            question = ex["question"]
            gold = ex.get("query", ex.get("SQL", ""))
            rows.append(
                {
                    "id": idx,
                    "db_id": db_id,
                    "db_path": str(db_path),
                    "question": question,
                    "gold_sql": gold,
                    "schema": schema,
                    "messages": build_messages(schema, question),
                }
            )
            idx += 1
    if skipped:
        print(f"  [info] skipped {skipped} examples (missing db / bad schema)")
    return rows


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):>6} rows -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spider-dir", default="data/spider", type=Path)
    ap.add_argument("--out-dir", default="data/processed", type=Path)
    ap.add_argument(
        "--train-files",
        nargs="+",
        default=["train_spider.json", "train_others.json"],
    )
    ap.add_argument("--dev-file", default="dev.json")
    args = ap.parse_args()

    spider_dir = args.spider_dir.resolve()
    if not spider_dir.exists():
        raise SystemExit(
            f"{spider_dir} not found. Run scripts/download_spider.sh first."
        )

    print("Building train split...")
    train = _build_split(spider_dir, args.train_files, test=False)
    _write_jsonl(train, args.out_dir / "train.jsonl")

    print("Building dev split...")
    dev = _build_split(spider_dir, [args.dev_file], test=False)
    _write_jsonl(dev, args.out_dir / "dev.jsonl")


if __name__ == "__main__":
    main()
