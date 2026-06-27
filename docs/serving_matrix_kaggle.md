# Serving matrix on Kaggle (quantize → serve → Pareto)

End-to-end recipe for filling the quality-vs-latency matrix on a free Kaggle T4,
after QLoRA training has produced `out/qlora-spider`. Every config is scored with
the same `textsql` CLIs, so the comparison is apples-to-apples.

The flow is split across two notebooks so the long training run and the fragile
serving stack don't share a kernel:

1. **Training notebook** — data → baseline → QLoRA (1 epoch) → eval-ft → summary,
   plus the **merge** cell below so the run also emits the fp16 model the
   quantizers consume.
2. **Serving notebook** — consumes the training notebook's output, builds the
   quantized variants, serves each, and records accuracy + latency.

Reliability legend: 🟢 runs cleanly on Kaggle every time · 🟡 stretch, may need
version pins on a free T4.

---

## Part 1 — add to the TRAINING notebook (after the summary cell)

Merging is ~10 min and its output feeds every quantizer. Running it inside the
training commit means the merged fp16 model is saved next to the adapter, so the
serving notebook never has to re-train or re-merge.

```python
# Cell 9 — fold LoRA into a standalone fp16 model (the input to all quantizers)
!python -m textsql.export.merge_lora \
  --base {MODEL} --adapter out/qlora-spider --out out/merged-fp16 \
  --dtype float16 --device cpu
# 3rd quality point: merged fp16 via HF (your full-precision fine-tuned ceiling)
!python -m textsql.eval.run_eval --backend hf --model out/merged-fp16 --dtype float16 \
  --data data/processed/dev.jsonl --limit {LIMIT} --out-dir results/merged_fp16
```

After this, the committed output holds `out/qlora-spider`, `out/merged-fp16`, and
three quality points: `base_hf`, `ft_nf4`, `merged_fp16`.

---

## Part 2 — new notebook: "qwen-text2sql-quantize-serve"

**Wiring (do this first):** sidebar → **+ Add Input → Notebooks →** the training
notebook's latest version. Its output mounts under `/kaggle/input/…`, giving you
the merged model without re-training.

### Cell 1 — setup + locate the merged model

```python
import os, glob
os.chdir("/kaggle/working")
!rm -rf text-to-sql-qlora
!git clone -q https://github.com/adipatel11/text-to-sql-qlora.git
os.chdir("/kaggle/working/text-to-sql-qlora")

hits = glob.glob("/kaggle/input/**/out/merged-fp16", recursive=True)
assert hits, "Add your training notebook's latest output as Input (need out/merged-fp16)."
MERGED = hits[0]; SRC = MERGED.split("/out/merged-fp16")[0]
print("Merged model:", MERGED)
!mkdir -p results && cp -r "{SRC}/results/." results/ 2>/dev/null; ls results   # carry over quality points
import torch; print("CUDA:", torch.cuda.is_available())
LIMIT = 300
```

### Cell 2 — Spider data again (calibration + eval + latency prompts)

```python
import glob; from pathlib import Path; import kagglehub
base = kagglehub.dataset_download("jeromeblanchet/yale-universitys-spider-10-nlp-dataset")
sd = next(str(Path(p).parent) for p in glob.glob(f"{base}/**/dev.json", recursive=True)
          if list(Path(p).parent.glob("database/*/*.sqlite")))
!python -m textsql.data.prepare_spider --spider-dir "{sd}"
!wc -l data/processed/dev.jsonl
```

### Cell 3 — server harness (start → wait ready → score accuracy + latency → kill)

One helper drives every served config.

```python
import subprocess, time, requests, os, signal
def serve_and_measure(cmd, name, served_model="qwen-sql",
                      base_url="http://localhost:8000/v1", limit=300,
                      lat_concurrency=8, lat_requests=128, ready_timeout=600):
    os.makedirs(f"results/{name}", exist_ok=True)
    logf = open(f"results/{name}/server.log", "w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            logf.close(); raise RuntimeError(f"[{name}] server exited early — see results/{name}/server.log")
        try:
            if requests.get(base_url + "/models", timeout=2).status_code == 200: break
        except Exception: pass
        if time.time() - t0 > ready_timeout:
            proc.terminate(); logf.close(); raise TimeoutError(f"[{name}] not ready in {ready_timeout}s")
        time.sleep(3)
    print(f"[{name}] up in {time.time()-t0:.0f}s — scoring...")
    try:
        os.system(f"python -m textsql.eval.run_eval --backend openai --base-url {base_url} "
                  f"--model {served_model} --data data/processed/dev.jsonl --limit {limit} --out-dir results/{name}")
        os.system(f"python -m textsql.eval.latency --base-url {base_url} --model {served_model} "
                  f"--data data/processed/dev.jsonl --concurrency {lat_concurrency} "
                  f"--num-requests {lat_requests} --out results/{name}/latency.json")
    finally:
        proc.send_signal(signal.SIGINT)
        try: proc.wait(timeout=15)
        except Exception: proc.kill()
        logf.close(); print(f"[{name}] stopped.")
```

### Cell 4 — 🟢 GGUF / llama.cpp (CPU): the guaranteed Pareto

```python
!bash scripts/export_gguf.sh "{MERGED}" out/gguf Q4_K_M Q5_K_M Q8_0
!ls -lh out/gguf/*.gguf
```

```python
SRV = "llama.cpp/build/bin/llama-server"
for q, name in [("Q4_K_M","cpu_gguf_q4km"),("Q5_K_M","cpu_gguf_q5km"),("Q8_0","cpu_gguf_q8")]:
    serve_and_measure([SRV,"-m",f"out/gguf/model-{q}.gguf","--host","0.0.0.0","--port","8000","-c","4096"],
                      name=name, limit=LIMIT, lat_concurrency=4, lat_requests=64)
```

### Cell 5 — 🟡 AWQ + vLLM (GPU): the "production GPU" points

vLLM install can fight Kaggle's preinstalled torch. AWQ works on the T4's Turing
arch; **GPTQ's Marlin kernels need Ampere, so skip GPTQ on a T4.**

```python
!pip -q install autoawq
!python -m textsql.export.quantize_awq --model "{MERGED}" \
  --calib-data data/processed/train.jsonl --out out/awq-4bit
!pip -q install vllm
serve_and_measure(["vllm","serve","out/awq-4bit","--quantization","awq",
                   "--served-model-name","qwen-sql","--port","8000",
                   "--max-model-len","2048","--gpu-memory-utilization","0.9"],
                  name="gpu_vllm_awq", limit=LIMIT, lat_concurrency=16, lat_requests=256, ready_timeout=900)
```

Optional fp16-on-GPU reference point (same harness):

```python
serve_and_measure(["vllm","serve",MERGED,"--served-model-name","qwen-sql","--port","8000",
                   "--max-model-len","2048","--gpu-memory-utilization","0.9","--dtype","float16"],
                  name="gpu_vllm_fp16", limit=LIMIT, lat_concurrency=16, lat_requests=256, ready_timeout=900)
```

### Cell 6 — Pareto table + plot

```python
!python -m textsql.plot_pareto --results-dir results --out results/summary
from IPython.display import Image
Image("results/summary.png")
```

---

## Reliability notes

- **Cell 4 (GGUF/CPU) is the floor you can count on.** It compiles and runs
  cleanly, and on its own yields a legitimate 4-point quality-vs-latency Pareto
  across quantization levels (Q4_K_M / Q5_K_M / Q8_0 + the merged fp16 quality
  anchor). That is the deliverable.
- **Cell 5 (vLLM/AWQ) is the upside.** If it won't cooperate on the free T4, the
  project still stands on the CPU sweep plus the HF quality numbers, and the GPU
  column becomes a "future work" line rather than a blocker.
- CPU eval of ~300 examples per GGUF variant is slow (~15–20 min each); drop
  `LIMIT` to 150 if you want faster turnaround, keeping it consistent across
  configs for a fair table.
- Each config writes `results/<name>/metrics.json` (+ `latency.json` for served
  configs); `plot_pareto` aggregates whatever is present, so you can run the
  cells in any order and re-plot incrementally.
