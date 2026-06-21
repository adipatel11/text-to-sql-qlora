"""Aggregate every config's results into one table + a quality-vs-latency
Pareto plot.

Convention: each config lives in results/<name>/ with
    metrics.json   (from run_eval  -> execution_accuracy)
    latency.json   (from latency   -> p99 latency, throughput, memory)

Usage:
    python -m textsql.plot_pareto --results-dir results --out results/summary
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_configs(results_dir: Path) -> list[dict]:
    rows = []
    for sub in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        m_path, l_path = sub / "metrics.json", sub / "latency.json"
        row = {"config": sub.name}
        if m_path.exists():
            m = json.loads(m_path.read_text())
            row["execution_accuracy"] = m.get("execution_accuracy")
        if l_path.exists():
            l = json.loads(l_path.read_text())
            row["p50_ms"] = l.get("latency_ms", {}).get("p50")
            row["p95_ms"] = l.get("latency_ms", {}).get("p95")
            row["p99_ms"] = l.get("latency_ms", {}).get("p99")
            row["req_per_sec"] = l.get("throughput", {}).get("requests_per_sec")
            row["tok_per_sec"] = l.get("throughput", {}).get("output_tokens_per_sec")
            row["peak_gpu_mib"] = l.get("peak_gpu_mem_mib")
        if len(row) > 1:
            rows.append(row)
    return rows


def pareto_front(rows: list[dict]) -> set[str]:
    """Pareto-optimal = no other config has both higher accuracy AND lower p99."""
    pts = [r for r in rows if r.get("execution_accuracy") and r.get("p99_ms")]
    front = set()
    for a in pts:
        dominated = any(
            b is not a
            and b["execution_accuracy"] >= a["execution_accuracy"]
            and b["p99_ms"] <= a["p99_ms"]
            and (b["execution_accuracy"] > a["execution_accuracy"] or b["p99_ms"] < a["p99_ms"])
            for b in pts
        )
        if not dominated:
            front.add(a["config"])
    return front


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/summary"))
    args = ap.parse_args()

    rows = load_configs(args.results_dir)
    if not rows:
        raise SystemExit(f"No metrics found under {args.results_dir}")
    front = pareto_front(rows)
    for r in rows:
        r["pareto"] = r["config"] in front

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["config", "execution_accuracy", "p50_ms", "p95_ms", "p99_ms",
            "req_per_sec", "tok_per_sec", "peak_gpu_mib", "pareto"]
    csv_path = args.out.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    print(f"Wrote {csv_path}")

    # Print a quick markdown table to stdout for the README.
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in sorted(rows, key=lambda x: -(x.get("execution_accuracy") or 0)):
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[info] matplotlib not installed; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        if not (r.get("execution_accuracy") and r.get("p99_ms")):
            continue
        on_front = r["config"] in front
        ax.scatter(r["p99_ms"], r["execution_accuracy"] * 100,
                   s=90 if on_front else 50,
                   color="crimson" if on_front else "steelblue",
                   edgecolors="black", zorder=3)
        ax.annotate(r["config"], (r["p99_ms"], r["execution_accuracy"] * 100),
                    fontsize=8, xytext=(5, 4), textcoords="offset points")
    fpts = sorted((r for r in rows if r["config"] in front), key=lambda x: x["p99_ms"])
    if len(fpts) > 1:
        ax.plot([r["p99_ms"] for r in fpts],
                [r["execution_accuracy"] * 100 for r in fpts],
                "--", color="crimson", alpha=0.6, label="Pareto frontier")
        ax.legend()
    ax.set_xlabel("p99 latency (ms)  — lower is better")
    ax.set_ylabel("Execution accuracy (%)  — higher is better")
    ax.set_title("Text-to-SQL: quality vs. latency")
    ax.grid(True, alpha=0.3)
    plot_path = args.out.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
