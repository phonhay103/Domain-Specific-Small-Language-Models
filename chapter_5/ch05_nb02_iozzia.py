"""Optimizing GPT-2 with ONNX for GPU Inference.

Companion script for Chapter 5 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Demonstrates exporting GPT-2 (CausalLM) to ONNX, applying GPU-specific
graph optimizations (fused multi-head attention, fast GeLU), and benchmarking
generation latency between PyTorch CUDA and ONNX Runtime CUDA Execution Provider.
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Third-party
import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.transformers.optimizer import optimize_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Common functional & UI utilities
from common.functional import calculate_speedup
from common.ui import (
    STYLE_INDEX,
    STYLE_NUMBER,
    STYLE_PRIMARY,
    STYLE_SECONDARY,
    STYLE_SUCCESS,
    STYLE_TEXT,
    STYLE_WARNING,
    console,
    create_table,
    pause,
    render_banner,
    render_card,
    render_step,
    render_takeaways,
    status_spinner,
)


# ---------------------------------------------------------------------------
# Immutable Domain Records & Constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GPULatencyMeasurement:
    """Immutable GPU latency timing record."""

    engine_name: str
    avg_ms: float
    std_ms: float
    speedup: float


MODEL_ID = "openai-community/gpt2"
OUTPUT_DIR = "./output_dir"
ONNX_DIR = f"{OUTPUT_DIR}/onnx_gpt2"
BASE_ONNX_PATH = f"{ONNX_DIR}/gpt2.onnx"
OPT_ONNX_PATH = f"{ONNX_DIR}/gpt2_opt_gpu.onnx"
PROMPT = "The future of artificial intelligence in scientific discovery is"
WARMUP_RUNS = 3
BENCHMARK_RUNS = 20
MAX_NEW_TOKENS = 32


# ---------------------------------------------------------------------------
# Pure Functions & Benchmarking Helpers
# ---------------------------------------------------------------------------
def measure_gpu_latency_pure(
    run_fn: Callable[[], Any],
    warmup: int = WARMUP_RUNS,
    runs: int = BENCHMARK_RUNS,
) -> tuple[float, float]:
    """Pure benchmarking measurement with GPU synchronization."""
    for _ in range(warmup):
        run_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    latencies: list[float] = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        run_fn()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)

    return float(np.mean(latencies)), float(np.std(latencies))


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_latency_comparison(measurements: Sequence[GPULatencyMeasurement]) -> None:
    """Render GPU latency speedup table."""
    columns = [
        ("Engine / Runtime", STYLE_PRIMARY, "left"),
        ("Average Latency (ms)", STYLE_WARNING, "right"),
        ("Std Dev (ms)", STYLE_TEXT, "right"),
        ("Speedup vs PyTorch", STYLE_SUCCESS, "right"),
    ]
    rows = [(m.engine_name, f"{m.avg_ms:.2f} ms", f"± {m.std_ms:.2f}", f"{m.speedup:.2f}x") for m in measurements]
    console.print(create_table(f"GPU Forward Pass Latency ({BENCHMARK_RUNS} iterations)", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute GPT-2 ONNX GPU optimization pipeline and latency comparison."""
    os.makedirs(ONNX_DIR, exist_ok=True)

    render_banner(
        title="Optimizing GPT-2 with ONNX Runtime for GPU Inference",
        subtitle="Chapter 5: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_ID,
            "Execution Providers": "CUDAExecutionProvider vs PyTorch CUDA",
            "Benchmark Runs": str(BENCHMARK_RUNS),
        },
        icon="🚀",
    )

    # Step 1: Loading PyTorch GPT-2 Model
    render_step(1, "Loading PyTorch GPT-2 Model", icon="📋")
    with status_spinner(f"Loading '{MODEL_ID}' causal language model..."):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()

    render_card("Model Status", f"Model loaded on device [text.highlight]{model.device}[/text.highlight]", icon="✔")

    # Step 2: Exporting Base ONNX Graph
    render_step(2, "Tracing & Exporting Base ONNX Graph", icon="⚙️")
    dummy_inputs = tokenizer("Hello world", return_tensors="pt")
    if torch.cuda.is_available():
        dummy_inputs = {k: v.to("cuda") for k, v in dummy_inputs.items()}

    with status_spinner("Exporting dynamic ONNX computational graph..."):
        torch.onnx.export(
            model,
            (dummy_inputs["input_ids"],),
            BASE_ONNX_PATH,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=14,
        )
    render_card(
        "Base ONNX Export", f"ONNX graph exported to:\n[text.highlight]{BASE_ONNX_PATH}[/text.highlight]", icon="💾"
    )

    # Step 3: Applying GPU Graph Optimizations
    render_step(3, "Applying GPU Graph Optimizations (FastGELU & Fused Attention)", icon="⚡")
    with status_spinner("Fusing Multi-Head Attention and FastGELU kernels for CUDA..."):
        opt_model = optimize_model(
            BASE_ONNX_PATH,
            model_type="gpt2",
            num_heads=12,
            hidden_size=768,
            use_gpu=torch.cuda.is_available(),
            opt_level=99,
        )
        opt_model.save_model_to_file(OPT_ONNX_PATH)
    render_card(
        "GPU Optimization", f"Optimized model saved to:\n[text.highlight]{OPT_ONNX_PATH}[/text.highlight]", icon="✔"
    )

    # Step 4: Inference Latency Benchmark on GPU
    render_step(4, "Benchmarking Latency on CUDA Execution Provider", icon="📊")
    inputs = tokenizer(PROMPT, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    )
    opt_sess = ort.InferenceSession(OPT_ONNX_PATH, providers=providers)
    ort_inputs = {"input_ids": inputs["input_ids"].cpu().numpy().astype(np.int64)}

    with status_spinner("Benchmarking PyTorch forward pass..."):
        pt_avg, pt_std = measure_gpu_latency_pure(lambda: model(inputs["input_ids"]))
    with status_spinner("Benchmarking ONNX Runtime GPU forward pass..."):
        ort_avg, ort_std = measure_gpu_latency_pure(lambda: opt_sess.run(None, ort_inputs))

    measurements = (
        GPULatencyMeasurement("PyTorch Eager (CUDA FP32)", pt_avg, pt_std, 1.0),
        GPULatencyMeasurement("ONNX Runtime (CUDA Fused)", ort_avg, ort_std, calculate_speedup(pt_avg, ort_avg)),
    )
    render_latency_comparison(measurements)

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Why ONNX Runtime on GPU",
                "ONNX Runtime CUDA execution provider bypasses PyTorch operator overhead and leverages custom CUDA kernels tailored for specific GPU architectures.",
            ),
            (
                "FastGELU Kernel Fusion",
                "Replaces standard GELU approximations with fused polynomial CUDA instructions, eliminating intermediary memory loads.",
            ),
            (
                "Dynamic Sequence Axes",
                "Setting dynamic_axes on sequence_length allows arbitrary prompt lengths without requiring recompilation or static graph tracing.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
