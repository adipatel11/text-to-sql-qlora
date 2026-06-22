.PHONY: help test data baseline-hf train eval-ft merge quant-gptq quant-awq gguf pareto clean
SHELL := /bin/bash

MODEL    ?= Qwen/Qwen2.5-Coder-3B-Instruct
ADAPTER  ?= out/qlora-spider
MERGED   ?= out/merged-fp16
CALIB    ?= data/processed/train.jsonl
DEV      ?= data/processed/dev.jsonl
LIMIT    ?= 200

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

test: ## Run the unit tests (stdlib only, no deps required)
	python -m unittest discover -s tests -t . -v

data: ## Download Spider and build train/dev JSONL
	bash scripts/download_spider.sh
	python -m textsql.data.prepare_spider

baseline-hf: ## Zero-shot baseline of the base model (local HF)
	python -m textsql.eval.run_eval --backend hf --model $(MODEL) \
	  --data $(DEV) --limit $(LIMIT) --out-dir results/base_hf

train: ## QLoRA fine-tune on Spider train
	python -m textsql.train.qlora_train --config configs/qlora_qwen3b.yaml

eval-ft: ## Evaluate the fine-tuned adapter (4-bit) on dev
	python -m textsql.eval.run_eval --backend hf --model $(MODEL) \
	  --adapter $(ADAPTER) --load-in-4bit --data $(DEV) --out-dir results/ft_nf4

merge: ## Merge the LoRA adapter into base -> fp16 model (feeds quantizers)
	python -m textsql.export.merge_lora --base $(MODEL) --adapter $(ADAPTER) \
	  --out $(MERGED)

quant-gptq: ## GPTQ 4-bit quantize the merged model (vLLM GPU)
	python -m textsql.export.quantize_gptq --model $(MERGED) \
	  --calib-data $(CALIB) --out out/gptq-4bit

quant-awq: ## AWQ 4-bit quantize the merged model (vLLM GPU)
	python -m textsql.export.quantize_awq --model $(MERGED) \
	  --calib-data $(CALIB) --out out/awq-4bit

gguf: ## Convert merged model -> GGUF Q4_K_M/Q5_K_M/Q8_0 (llama.cpp CPU)
	bash scripts/export_gguf.sh $(MERGED) out/gguf

pareto: ## Build the results table + Pareto plot from results/
	python -m textsql.plot_pareto --results-dir results --out results/summary

clean: ## Remove generated results
	rm -rf results/*
