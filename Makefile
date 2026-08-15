export PYTHONPATH := .

.PHONY: help install lint format check test-syntax clean \
	run-ch02-nb01 run-ch02-nb02 run-ch02-nb03 run-ch02 \
	run-ch03-nb01 run-ch03 \
	run-ch04-nb01 run-ch04-nb02 run-ch04 \
	run-ch05-nb01 run-ch05-nb02 run-ch05 \
	run-ch06-nb01 run-ch06-nb02 run-ch06-nb03 run-ch06-nb04 run-ch06 \
	run-ch07-nb01 run-ch07-nb02 run-ch07-nb03 run-ch07 \
	run-ch08-nb01 run-ch08-nb02 run-ch08-nb03 run-ch08 \
	run-ch09-nb01 run-ch09-nb02 run-ch09 \
	run-ch10-nb01 run-ch10 \
	run-ch11-nb01 run-ch11-nb02 run-ch11-nb03 run-ch11 \
	run-ch13-nb01 run-ch13-nb02 run-ch13 \
	run-ch14-nb01 run-ch14-nb02 run-ch14 \
	run-ch15-nb01 run-ch15-nb02 run-ch15

# Default target
.DEFAULT_GOAL := help

help: ## Display available Makefile commands and targets
	@echo "Domain-Specific Small Language Models — Project Automation"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Quality & Environment:"
	@grep -E '^(install|lint|format|check|test-syntax|clean):.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Chapter Runners (All 30 Scripts):"
	@grep -E '^run-ch[0-9]+(-nb[0-9]+)?:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[32m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all runtime and dev dependencies using uv
	uv sync

lint: ## Run linter checks with auto-fix (Ruff) and type checker (ty)
	uv run ruff check --fix .
	uv run ty check

format: ## Format Python source code using Ruff formatter
	uv run ruff format .

check: ## Run lint, format-check, and ty without modifying files
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check

test-syntax: ## Verify Python compilation syntax across all chapter scripts
	uv run python -m py_compile chapter_*/*.py

clean: ## Clean build artifacts, temporary models, and cache files
	rm -rf .pytest_cache .ruff_cache __pycache__ onnx_models outputs experiments local-pt-checkpoint local-8bit-checkpoint onnx
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# ---------------------------------------------------------------------------
# Chapter 2: Foundations & Fine-Tuning
# ---------------------------------------------------------------------------
run-ch02: run-ch02-nb01 ## Run Chapter 2 default demo
run-ch02-nb01: ## Ch02 NB01: FAISS Dense Vector Semantic Search
	uv run python chapter_2/ch02_nb01_faiss_search.py

run-ch02-nb02: ## Ch02 NB02: DistilBERT SQuAD Extractive QA Fine-Tuning
	uv run python chapter_2/ch02_nb02_squad_finetuning.py

run-ch02-nb03: ## Ch02 NB03: FLAN-T5 SAMSum Dialogue Summarization with LoRA
	uv run python chapter_2/ch02_nb03_lora_summarization.py

# ---------------------------------------------------------------------------
# Chapter 3: Domain Customization
# ---------------------------------------------------------------------------
run-ch03: run-ch03-nb01 ## Run Chapter 3 default demo
run-ch03-nb01: ## Ch03 NB01: Optuna Hyperparameter Tuning for Synthetic Code
	uv run python chapter_3/ch03_nb01_synthetic_tuning.py

# ---------------------------------------------------------------------------
# Chapter 4: Inference Optimization
# ---------------------------------------------------------------------------
run-ch04: run-ch04-nb01 ## Run Chapter 4 default demo
run-ch04-nb01: ## Ch04 NB01: GPT-2 KV-Cache & Left-Padded Batching Benchmark
	uv run python chapter_4/ch04_nb01_iozzia.py

run-ch04-nb02: ## Ch04 NB02: DeepSpeed Inference Latency & Speedup Benchmark
	uv run python chapter_4/ch04_nb02_iozzia.py

# ---------------------------------------------------------------------------
# Chapter 5: ONNX Runtime & Graph Optimization
# ---------------------------------------------------------------------------
run-ch05: run-ch05-nb01 ## Run Chapter 5 default demo
run-ch05-nb01: ## Ch05 NB01: BERT ONNX Conversion & CPU Graph Optimization
	uv run python chapter_5/ch05_nb01_iozzia.py

run-ch05-nb02: ## Ch05 NB02: GPT-2 ONNX GPU Conversion & Transformer Fusion
	uv run python chapter_5/ch05_nb02_iozzia.py

# ---------------------------------------------------------------------------
# Chapter 6: Quantization Spectrum
# ---------------------------------------------------------------------------
run-ch06: run-ch06-nb01 ## Run Chapter 6 default demo
run-ch06-nb01: ## Ch06 NB01: GPT-2 Absmax INT8 Quantization & Perplexity
	uv run python chapter_6/ch06_nb01_iozzia.py

run-ch06-nb02: ## Ch06 NB02: GPT-2 LLM.int8() Outlier Mixed-Precision Quantization
	uv run python chapter_6/ch06_nb02_iozzia.py

run-ch06-nb03: ## Ch06 NB03: DistilBERT Optimum Dynamic INT8 Quantization (AVX-512)
	uv run python chapter_6/ch06_nb03_iozzia.py

run-ch06-nb04: ## Ch06 NB04: GPT-2 AutoGPTQ 4-Bit Weight Quantization
	uv run python chapter_6/ch06_nb04_iozzia.py

# ---------------------------------------------------------------------------
# Chapter 7: Code Generation SLMs
# ---------------------------------------------------------------------------
run-ch07: run-ch07-nb01 ## Run Chapter 7 default demo
run-ch07-nb01: ## Ch07 NB01: CodeGen-350M Vanilla vs ONNX vs Quantized Latency
	uv run python chapter_7/ch07_nb01_iozzia.py

run-ch07-nb02: ## Ch07 NB02: CodeGen ReCode AST Syntax Validation on HumanEval
	uv run python chapter_7/ch07_nb02_iozzia.py

run-ch07-nb03: ## Ch07 NB03: StarCoder2-3B bfloat16 vs 8-Bit Quantization Benchmark
	uv run python chapter_7/ch07_nb03_iozzia.py

# ---------------------------------------------------------------------------
# Chapter 8: Domain SLMs in Biology & Material Science
# ---------------------------------------------------------------------------
run-ch08: run-ch08-nb01 ## Run Chapter 8 default demo
run-ch08-nb01: ## Ch08 NB01: ProtGPT2 Protein Sequence Generation & Perplexity
	uv run python chapter_8/ch08_nb01_iozzia.py

run-ch08-nb02: ## Ch08 NB02: AntibodyGPT Antigen-Conditioned Generation
	uv run python chapter_8/ch08_nb02_iozzia.py

run-ch08-nb03: ## Ch08 NB03: CrystaLLM Checkpoint Conversion for Crystal Generation
	uv run python chapter_8/ch08_nb03_iozzia.py

# ---------------------------------------------------------------------------
# Chapter 9: Memory Offloading & Outlier Smoothing
# ---------------------------------------------------------------------------
run-ch09: run-ch09-nb01 ## Run Chapter 9 default demo
run-ch09-nb01: ## Ch09 NB01: FlexGen OPT-1.3B RAM + Disk Offloading Inference
	uv run python chapter_9/ch09_nb01_iozzia.py

run-ch09-nb02: ## Ch09 NB02: SmoothQuant OPT-6.7B W8A8 Activation Outlier Smoothing
	uv run python chapter_9/ch09_nb02_iozzia.py

# ---------------------------------------------------------------------------
# Chapter 10: Performance Profiling
# ---------------------------------------------------------------------------
run-ch10: run-ch10-nb01 ## Run Chapter 10 default demo
run-ch10-nb01: ## Ch10 NB01: ONNX Runtime Operator Profiling & Graph Breakdown
	uv run python chapter_10/ch10_nb01_iozzia.py

# ---------------------------------------------------------------------------
# Chapter 11: Production Serving & Compilation
# ---------------------------------------------------------------------------
run-ch11: run-ch11-nb01 ## Run Chapter 11 default demo
run-ch11-nb01: ## Ch11 NB01: vLLM Offline Serving: GPT-2, Phi-3 & GGUF
	uv run python chapter_11/ch11_nb01_iozzia.py

run-ch11-nb02: ## Ch11 NB02: Endpoint Serving Candidate Latency & Scaling Benchmark
	uv run python chapter_11/ch11_nb02_iozzia.py

run-ch11-nb03: ## Ch11 NB03: MLC LLM Model Compilation & Streaming Inference
	uv run python chapter_11/ch11_nb03_iozzia.py

# ---------------------------------------------------------------------------
# Chapter 13: Local RAG & Autonomous Agents
# ---------------------------------------------------------------------------
run-ch13: run-ch13-nb01 ## Run Chapter 13 default demo
run-ch13-nb01: ## Ch13 NB01: Serverless Local RAG with LanceDB & Phi-3 GGUF
	uv run python chapter_13/ch13_nb01_iozzia.py

run-ch13-nb02: ## Ch13 NB02: SmolAgents Autonomous Flight Booking SLM Agent
	uv run python chapter_13/ch13_nb02_iozzia.py

# ---------------------------------------------------------------------------
# Chapter 14: Graph RAG & Hybrid Retrieval
# ---------------------------------------------------------------------------
run-ch14: run-ch14-nb01 ## Run Chapter 14 default demo
run-ch14-nb01: ## Ch14 NB01: Custom Graph RAG with Ollama & Leiden Community Detection
	uv run python chapter_14/ch14_nb01_iozzia.py

run-ch14-nb02: ## Ch14 NB02: Agentic Hybrid RAG (BM25 + Semantic + Web Search)
	uv run python chapter_14/ch14_nb02_iozzia.py

# ---------------------------------------------------------------------------
# Chapter 15: Test-Time Reasoning & GRPO
# ---------------------------------------------------------------------------
run-ch15: run-ch15-nb01 ## Run Chapter 15 default demo
run-ch15-nb01: ## Ch15 NB01: OptiLLM AutoThink & GSM8k Reasoning Benchmark
	uv run python chapter_15/ch15_nb01_iozzia.py

run-ch15-nb02: ## Ch15 NB02: GRPO + QLoRA Reinforcement Learning Reasoning Fine-Tuning
	uv run python chapter_15/ch15_nb02_iozzia.py
