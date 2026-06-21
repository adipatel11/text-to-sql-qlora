"""Latency + throughput rig for an OpenAI-compatible server.

Fires real text-to-SQL prompts at a target concurrency and reports:
  * latency p50 / p95 / p99 (and mean) in milliseconds
  * throughput: requests/sec and output tokens/sec
  * generated-token stats
  * optional peak GPU memory (sampled from nvidia-smi)

This is the speed half of the quality-vs-speed tradeoff. Point --base-url at
vLLM or llama.cpp's server and run the same command for every config.

Example:
    python -m textsql.eval.latency \
        --base-url http://localhost:8000/v1 --model qwen-sql \
        --data data/processed/dev.jsonl \
        --concurrency 16 --num-requests 256 --out results/vllm_awq_latency.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from textsql.prompts import build_messages


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class GpuSampler(threading.Thread):
    """Polls `nvidia-smi` for peak used memory (MiB). No-op if unavailable."""

    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_mib = 0
        self._stop = threading.Event()
        self.available = False

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL, text=True,
                )
                self.available = True
                used = max(int(x) for x in out.split())
                self.peak_mib = max(self.peak_mib, used)
            except Exception:  # noqa: BLE001 - no GPU / no nvidia-smi
                return
            self._stop.wait(self.interval)

    def stop(self) -> int:
        self._stop.set()
        self.join(1.0)
        return self.peak_mib if self.available else -1


def make_payloads(data: Path, n: int) -> list[dict]:
    rows = []
    with open(data, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    payloads = []
    for i in range(n):
        r = rows[i % len(rows)]
        msgs = r.get("messages") or build_messages(r["schema"], r["question"])
        payloads.append(msgs)
    return payloads


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--num-requests", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {args.api_key}"}
    payloads = make_payloads(args.data, args.num_requests + args.warmup)

    import requests

    def fire(messages: list[dict]) -> dict:
        body = {
            "model": args.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": args.max_new_tokens,
        }
        t0 = time.perf_counter()
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=300)
            resp.raise_for_status()
            j = resp.json()
            usage = j.get("usage", {})
            return {
                "ok": True,
                "latency": time.perf_counter() - t0,
                "out_tokens": usage.get("completion_tokens", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
            }
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "latency": time.perf_counter() - t0, "error": str(e)}

    # Warmup (not measured).
    print(f"Warmup ({args.warmup} requests)...")
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(fire, payloads[: args.warmup]))

    measured = payloads[args.warmup :]
    print(f"Measuring {len(measured)} requests at concurrency {args.concurrency}...")
    gpu = GpuSampler()
    gpu.start()

    results = []
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(fire, m) for m in measured]
        for fut in as_completed(futs):
            results.append(fut.result())
    wall = time.perf_counter() - wall_start
    peak_mib = gpu.stop()

    ok = [r for r in results if r["ok"]]
    failed = len(results) - len(ok)
    lat_ms = [r["latency"] * 1000 for r in ok]
    out_tokens = sum(r["out_tokens"] for r in ok)

    report = {
        "model": args.model,
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "num_requests": len(results),
        "failed": failed,
        "wall_seconds": round(wall, 3),
        "latency_ms": {
            "mean": round(statistics.fmean(lat_ms), 1) if lat_ms else None,
            "p50": round(percentile(lat_ms, 50), 1),
            "p95": round(percentile(lat_ms, 95), 1),
            "p99": round(percentile(lat_ms, 99), 1),
            "max": round(max(lat_ms), 1) if lat_ms else None,
        },
        "throughput": {
            "requests_per_sec": round(len(ok) / wall, 2) if wall else None,
            "output_tokens_per_sec": round(out_tokens / wall, 1) if wall else None,
        },
        "output_tokens_total": out_tokens,
        "peak_gpu_mem_mib": peak_mib,  # -1 if no GPU / nvidia-smi
        "max_new_tokens": args.max_new_tokens,
    }

    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
