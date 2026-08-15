# Domain-Specific Small Language Models

> **Note:** This repository is a Python script version of the original repository [virtualramblas/Domain-Specific-Small-Language-Models](https://github.com/virtualramblas/Domain-Specific-Small-Language-Models). All Jupyter Notebooks have been converted into production-ready `.py` scripts enhanced with [`rich`](https://github.com/Textualize/rich) terminal reporting, educational insights, and modern Python tooling (`uv`, `ruff`, `ty`).

Companion codebase for the **[Domain-Specific Small Language Models](https://shortener.manning.com/OwMa)** book (Guglielmo Iozzia, Manning Publications).

---

## ⚡ Quick Start & Automation

This project uses modern Python development tools (`uv`, `ruff`, `ty`, and `make`):

```bash
# Display all available targets
make help

# Install all dependencies (runtime + dev tools)
make install

# Check code quality (linting + formatting + type-checking)
make check

# Auto-fix linting issues and run type checker
make lint

# Auto-format codebase with Ruff formatter
make format

# Verify Python syntax across all chapter scripts
make test-syntax
```

---

## 🎨 Design System & Functional Architecture

Every script adheres to clean **Functional Programming** and a **Dark-Mode-First Eye-Friendly Terminal UI**:

- **Functional Programming Principles**:
  - Domain records are modeled as immutable structures (`@dataclass(frozen=True)`).
  - Business logic is partitioned into pure mathematical/statistical transformations (`common/functional.py` with `pipe`, `calculate_speedup`, `map_tuple`).
  - Side-effects (I/O, network queries, model state, and Rich UI rendering) are strictly isolated in dedicated view functions.
- **Eye-Friendly Dark-Mode Palette (`common/ui.py`)**:
  - Replaced glaring high-contrast colors with a pastel palette (Tokyo Night / Catppuccin Macchiato theme: Lavender `#b4befe`, Soft Blue `#89b4fa`, Mint Green `#a6e3a1`, Peach `#fab387`, Sky `#89dceb`).
  - Subtle borders, interactive progress spinners (`status_spinner`), and paced step-by-step executions (`pause()`).

---

## 🚀 Running Chapter Demos

You can run individual chapter demonstrations directly via `make` or `uv run`:

| Target | Description | File |
| :--- | :--- | :--- |
| `make run-ch02` | FAISS Semantic Vector Search | [`chapter_2/ch02_nb01_faiss_search.py`](chapter_2/ch02_nb01_faiss_search.py) |
| `make run-ch03` | Optuna Hyperparameter Optimization | [`chapter_3/ch03_nb01_synthetic_tuning.py`](chapter_3/ch03_nb01_synthetic_tuning.py) |
| `make run-ch04` | KV-Cache & Batching Benchmark | [`chapter_4/ch04_nb01_iozzia.py`](chapter_4/ch04_nb01_iozzia.py) |
| `make run-ch05` | ONNX Conversion & CPU Graph Optimization | [`chapter_5/ch05_nb01_iozzia.py`](chapter_5/ch05_nb01_iozzia.py) |
| `make run-ch06` | Absmax INT8 Quantization & Perplexity | [`chapter_6/ch06_nb01_iozzia.py`](chapter_6/ch06_nb01_iozzia.py) |
| `make run-ch07` | CodeGen 350M Latency Percentiles (P50–P99) | [`chapter_7/ch07_nb01_iozzia.py`](chapter_7/ch07_nb01_iozzia.py) |
| `make run-ch08` | ProtGPT2 Protein Sequence Generation | [`chapter_8/ch08_nb01_iozzia.py`](chapter_8/ch08_nb01_iozzia.py) |
| `make run-ch09` | FlexGen RAM + Disk Offloading for OPT-1.3B | [`chapter_9/ch09_nb01_iozzia.py`](chapter_9/ch09_nb01_iozzia.py) |
| `make run-ch10` | ONNX Runtime Operator Profiling | [`chapter_10/ch10_nb01_iozzia.py`](chapter_10/ch10_nb01_iozzia.py) |
| `make run-ch11` | Offline Serving with vLLM & GGUF | [`chapter_11/ch11_nb01_iozzia.py`](chapter_11/ch11_nb01_iozzia.py) |
| `make run-ch13` | Local RAG with LanceDB & Phi-3 Mini | [`chapter_13/ch13_nb01_iozzia.py`](chapter_13/ch13_nb01_iozzia.py) |
| `make run-ch14` | Custom Graph RAG with Ollama & NetworkX | [`chapter_14/ch14_nb01_iozzia.py`](chapter_14/ch14_nb01_iozzia.py) |
| `make run-ch15` | AutoThink & GSM8k Reasoning Benchmark | [`chapter_15/ch15_nb01_iozzia.py`](chapter_15/ch15_nb01_iozzia.py) |

---

## 📚 Repository Map & Topics

- **Chapter 2: Foundations & Fine-Tuning** (FAISS dense vector search, SQuAD span extraction, FLAN-T5 LoRA fine-tuning).
- **Chapter 3: Domain Customization** (Synthetic code generation with Manim dataset, Optuna hyperparameter optimization).
- **Chapter 4: Inference Optimization** (Autoregressive KV-caching, left-padded batching, DeepSpeed kernel acceleration).
- **Chapter 5: ONNX Runtime** (PyTorch to ONNX export, dynamic axes, transformer operator graph fusion, numerical parity verification).
- **Chapter 6: Quantization Spectrum** (Absmax INT8, LLM.int8() mixed precision, Optimum dynamic quantization, AutoGPTQ 4-bit).
- **Chapter 7: Code Generation SLMs** (CodeGen 350M, StarCoder2-3B, ReCode AST syntax pruning, HumanEval pass@k).
- **Chapter 8: Biology & Material Science** (ProtGPT2 protein generation, AntibodyGPT antigen conditioning, CrystaLLM crystal generation).
- **Chapter 9: Memory Scaling & Large SLMs** (FlexGen memory hierarchy offloading, SmoothQuant W8A8 activation outlier smoothing).
- **Chapter 10: Performance Profiling** (ONNX operator execution breakdown, kernel duration vs occurrences).
- **Chapter 11: Production Serving** (vLLM PagedAttention, endpoint benchmarking, MLC LLM TVM compilation).
- **Chapter 13: RAG & Autonomous Agents** (LanceDB serverless vector database, SmolAgents CodeAgent flight booking).
- **Chapter 14: Advanced Retrieval Systems** (Knowledge Graph extraction, Leiden community clustering, Hybrid search BM25 + Vector).
- **Chapter 15: Reasoning & Reinforcement Learning** (OptiLLM AutoThink test-time compute, GRPO + QLoRA Unsloth fine-tuning).
