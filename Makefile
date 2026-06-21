.PHONY: help data baseline-hf train eval-ft pareto clean
SHELL := /bin/bash

MODEL    ?= Qwen/Qwen2.5-Coder-3B-Instruct
ADAPTER  ?= out/qlora-spider
DEV      ?= data/processed/dev.jsonl
LIMIT    ?= 200

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

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

pareto: ## Build the results table + Pareto plot from results/
	python -m textsql.plot_pareto --results-dir results --out results/summary

clean: ## Remove generated results
	rm -rf results/*
