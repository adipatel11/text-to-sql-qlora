# text-to-SQL: fine-tune → quantize → serve → benchmark

Fine-tune a small open coder model (Qwen2.5-Coder-3B-Instruct) with **QLoRA**
for text-to-SQL on **Spider**, then quantize it and benchmark the
quality-vs-speed tradeoff rigorously across two serving stacks.

The metric is **execution accuracy** — does the generated query return the same
rows as the reference? No LLM-as-judge fuzziness.

**Target headline:** *Fine-tuned Qwen2.5-Coder-3B with QLoRA for text-to-SQL
(X% execution accuracy on Spider), then benchmarked 4 quantization methods
across vLLM and llama.cpp — cut p99 latency N% and lifted throughput M% at
under 2 pp accuracy loss, identifying the Pareto-optimal config for GPU and CPU
serving.*

---

## The comparison matrix

The differentiator is one evaluation harness measured across every cell:

| Quantization      | GPU (vLLM)        | CPU (llama.cpp)   |
|-------------------|-------------------|-------------------|
| fp16 (baseline)   | ✅ vLLM           | —                 |
| bitsandbytes NF4  | ✅ HF backend     | —                 |
| GPTQ (4-bit)      | ✅ vLLM           | —                 |
| AWQ (4-bit)       | ✅ vLLM           | —                 |
| GGUF (Q4_K_M/Q5/Q8) | —               | ✅ llama.cpp      |

For every cell we report: execution accuracy, p50/p95/p99 latency, throughput
(req/s and tokens/s), peak memory, disk size — then plot the Pareto frontier.

**Why this is cheap to run:** eval and the latency rig depend only on a single
generation interface (`textsql/backends.py`). vLLM and llama.cpp both expose an
OpenAI-compatible API, so the *same* command scores all of them — only
`--base-url` changes.

---

## Repo layout

```
textsql/
  prompts.py            # prompt format + SQL extraction (single source of truth)
  schema.py             # CREATE TABLE schema straight from the .sqlite file
  backends.py           # HFBackend + OpenAIBackend (one generation interface)
  data/
    prepare_spider.py   # raw Spider -> train/dev JSONL (with pinned prompts)
  eval/
    execution.py        # execution-accuracy comparator (the honest metric)
    run_eval.py         # generate predictions + score, any backend
    latency.py          # p50/p95/p99, throughput, peak GPU mem
  train/
    qlora_train.py      # QLoRA SFT, completion-only loss
  plot_pareto.py        # results table + quality-vs-latency Pareto plot
configs/qlora_qwen3b.yaml
scripts/download_spider.sh
Makefile
```

Results convention: each config writes to `results/<name>/` with
`metrics.json` (accuracy) and `latency.json` (speed). `plot_pareto` joins them.

---

## Setup

> **Python note:** use **3.10–3.11**. The current environment has 3.14, which
> is too new for stable torch/vLLM/bitsandbytes wheels. Make a clean venv:
> `python3.11 -m venv .venv && source .venv/bin/activate`.

```bash
# Orchestration / eval box (CPU is fine):
pip install -r requirements-base.txt

# Training / local HF inference (CUDA GPU): install torch for your CUDA first.
pip install -r requirements-train.txt

# GPU serving:
pip install -r requirements-serve.txt
```

Compute: QLoRA on a 3B model fits a single 16–24 GB GPU (Kaggle free tier,
Colab, or a spot L4/A10). Drop to Qwen2.5-Coder-1.5B if memory is tight
(`--set model.name=Qwen/Qwen2.5-Coder-1.5B-Instruct`).

---

## Workflow

```bash
# 1. Data: download Spider and build prompts
make data

# 2. Baseline FIRST — measure the base model before any fine-tuning
make baseline-hf            # writes results/base_hf/metrics.json

# 3. Fine-tune with QLoRA
make train                  # writes adapter to out/qlora-spider

# 4. Evaluate the fine-tuned model (should clearly beat baseline)
make eval-ft                # writes results/ft_nf4/metrics.json

# 5. Build the summary table + Pareto plot once you have several configs
make pareto
```

### Serving matrix (the core)

Start a server, then run the **same** eval + latency commands against it.

**vLLM (GPU), e.g. fp16 / GPTQ / AWQ:**
```bash
# Merge the LoRA adapter into the base weights first (one-time), then quantize
# with auto-gptq / autoawq, or serve fp16 directly:
vllm serve <model-or-quantized-dir> --served-model-name qwen-sql --port 8000

python -m textsql.eval.run_eval --backend openai \
  --base-url http://localhost:8000/v1 --model qwen-sql \
  --data data/processed/dev.jsonl --out-dir results/vllm_awq
python -m textsql.eval.latency \
  --base-url http://localhost:8000/v1 --model qwen-sql \
  --data data/processed/dev.jsonl --concurrency 16 --num-requests 256 \
  --out results/vllm_awq/latency.json
```

**llama.cpp (CPU), GGUF:**
```bash
# Convert merged HF model -> GGUF, then quantize to Q4_K_M / Q5_K_M / Q8_0:
#   python llama.cpp/convert_hf_to_gguf.py <merged_dir> --outfile model-f16.gguf
#   ./llama.cpp/llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M
./llama.cpp/llama-server -m model-Q4_K_M.gguf --port 8000 -c 4096

python -m textsql.eval.run_eval --backend openai \
  --base-url http://localhost:8000/v1 --model gguf \
  --data data/processed/dev.jsonl --out-dir results/llamacpp_q4km
# (same latency command, point --base-url at this server)
```

**bitsandbytes NF4 (GPU, via HF backend, no server):**
```bash
python -m textsql.eval.run_eval --backend hf \
  --model Qwen/Qwen2.5-Coder-3B-Instruct --adapter out/qlora-spider \
  --load-in-4bit --data data/processed/dev.jsonl --out-dir results/hf_nf4
```

---

## Notes & honest caveats

- **Execution match vs. official test-suite eval.** `eval/execution.py` runs
  gold and prediction against the single dev database and compares result sets
  (order-sensitive only when gold has `ORDER BY`). This is the standard, light
  "execution match." The official Spider *test-suite* evaluator runs against
  many perturbed DBs to catch coincidental matches — swap it in for
  publication-grade numbers. Reported accuracy here is a close, slightly
  optimistic proxy.
- **Determinism.** Eval uses greedy decoding (temperature 0) so accuracy is
  reproducible across backends.
- **Prompt pinning.** `prompts.py` is the only place the prompt is defined;
  training and all eval backends import it, so the model is evaluated on
  exactly the format it was trained on.
- **BIRD** is a harder drop-in target if you want a tougher benchmark later.

## Stretch ideas (pick one)

Speculative decoding · multi-LoRA serving in vLLM · a length-bucketed batcher
vs. vLLM continuous batching head-to-head · a small Gradio demo (question +
schema → SQL + live speed numbers).
