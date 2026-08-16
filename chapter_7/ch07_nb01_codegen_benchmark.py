"""Benchmarking Python Code Generation with Vanilla, ONNX Converted and Quantized CodeGen Models.

Companion script for Chapter 7 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Benchmarks inference performance (latency percentiles, throughput) when generating
Python code using CodeGen 350M mono across PyTorch CPU, ONNX CPU, and INT8 Quantized ONNX.
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import gc
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Third-party
import numpy as np
import pandas as pd
import torch
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer, CodeGenForCausalLM, pipeline

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
    render_code_block,
    render_device_info,
    render_step,
    render_takeaways,
    status_spinner,
)


# ---------------------------------------------------------------------------
# Immutable Domain Records & Constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RuntimeLatencyDistribution:
    """Immutable latency percentiles and throughput metrics."""

    runtime_name: str
    avg_latency_ms: float
    p50_ms: float
    p75_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    throughput_seq_per_sec: float


MODEL_ID = "Salesforce/codegen-350M-mono"
LOCAL_CHECKPOINT_DIR = "local-pt-checkpoint"
ONNX_PATH = Path("onnx")
QUANTIZED_MODEL_FILE = "model_quantized.onnx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD_TOKEN_ID = 50256
MAX_GEN_LENGTH = 12
BENCHMARK_PROMPT = "def hello_world():"
WARMUP_RUNS = 10
BENCHMARK_RUNS = 100


# ---------------------------------------------------------------------------
# Pure Functions & Statistical Computation
# ---------------------------------------------------------------------------
def compute_latency_distribution(runtime_name: str, raw_seconds: Sequence[float]) -> RuntimeLatencyDistribution:
    """Pure calculation of latency percentiles and throughput."""
    times_ms = np.array(raw_seconds) * 1000.0
    avg_ms = float(np.mean(times_ms))
    throughput = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    return RuntimeLatencyDistribution(
        runtime_name=runtime_name,
        avg_latency_ms=avg_ms,
        p50_ms=float(np.percentile(times_ms, 50)),
        p75_ms=float(np.percentile(times_ms, 75)),
        p90_ms=float(np.percentile(times_ms, 90)),
        p95_ms=float(np.percentile(times_ms, 95)),
        p99_ms=float(np.percentile(times_ms, 99)),
        throughput_seq_per_sec=throughput,
    )


def benchmark_pipe_execution(pipe: Any, prompt: str) -> list[float]:
    """Pure timing loop collecting raw generation durations in seconds."""
    for _ in range(WARMUP_RUNS):
        _ = pipe(prompt)

    durations: list[float] = []
    for _ in range(BENCHMARK_RUNS):
        start = perf_counter()
        _ = pipe(prompt)
        durations.append(perf_counter() - start)
    return durations


def create_generation_pipe(model: Any, tokenizer: Any) -> Any:
    """Pure factory for text-generation pipeline."""
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=PAD_TOKEN_ID,
        truncation=True,
        max_length=MAX_GEN_LENGTH,
    )


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_metrics_table(profiles: Sequence[RuntimeLatencyDistribution]) -> None:
    """Render comprehensive latency percentiles and throughput table."""
    columns = [
        ("Runtime Engine", STYLE_PRIMARY, "left"),
        ("Avg Latency", STYLE_WARNING, "right"),
        ("P50 Median", STYLE_TEXT, "right"),
        ("P90 Latency", STYLE_TEXT, "right"),
        ("P95 Latency", STYLE_SECONDARY, "right"),
        ("P99 Tail", STYLE_WARNING, "right"),
        ("Throughput", STYLE_SUCCESS, "right"),
    ]
    rows = [
        (
            p.runtime_name,
            f"{p.avg_latency_ms:.2f} ms",
            f"{p.p50_ms:.2f} ms",
            f"{p.p90_ms:.2f} ms",
            f"{p.p95_ms:.2f} ms",
            f"{p.p99_ms:.2f} ms",
            f"{p.throughput_seq_per_sec:.2f} seq/s",
        )
        for p in profiles
    ]
    console.print(create_table(f"CodeGen-350M Benchmark Metrics ({BENCHMARK_RUNS} iterations)", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute CodeGen-350M optimization and latency benchmark."""
    render_banner(
        title="Benchmarking CodeGen-350M: Vanilla vs ONNX vs INT8 Quantized",
        subtitle="Chapter 7: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_ID,
            "Hardware": DEVICE.upper(),
            "Prompt": f'"{BENCHMARK_PROMPT}"',
            "Iterations": str(BENCHMARK_RUNS),
        },
        icon="🚀",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    profiles: list[RuntimeLatencyDistribution] = []

    # Phase 1: Vanilla PyTorch Benchmark
    render_step(1, "Vanilla PyTorch CPU Benchmark", icon="📋")
    with status_spinner(f"Loading '{MODEL_ID}' on PyTorch CPU..."):
        pt_model = CodeGenForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
        pt_model.eval()
        pt_pipe = create_generation_pipe(pt_model, tokenizer)
    render_device_info(DEVICE, model=pt_model)

    sample_out = pt_pipe(BENCHMARK_PROMPT)[0]["generated_text"]
    render_code_block(sample_out, language="python", title="PyTorch Code Generation Preview")

    with status_spinner(f"Benchmarking PyTorch CPU across {BENCHMARK_RUNS} runs..."):
        pt_durations = benchmark_pipe_execution(pt_pipe, BENCHMARK_PROMPT)
        profiles.append(compute_latency_distribution("PyTorch Eager (CPU)", pt_durations))

    # Phase 2: Base ONNX Model Benchmark
    render_step(2, "Exporting and Benchmarking Base ONNX Model", icon="⚙️")
    with status_spinner("Exporting CodeGen to ONNX graph..."):
        onnx_model = ORTModelForCausalLM.from_pretrained(MODEL_ID, export=True)
        onnx_model.save_pretrained(ONNX_PATH)
        tokenizer.save_pretrained(ONNX_PATH)
        onnx_pipe = create_generation_pipe(onnx_model, tokenizer)

    with status_spinner(f"Benchmarking Base ONNX CPU across {BENCHMARK_RUNS} runs..."):
        onnx_durations = benchmark_pipe_execution(onnx_pipe, BENCHMARK_PROMPT)
        profiles.append(compute_latency_distribution("ONNX Runtime (Base CPU)", onnx_durations))

    del onnx_pipe
    gc.collect()

    # Phase 3: 8-Bit Quantized ONNX Benchmark
    render_step(3, "Applying Dynamic AVX-512 VNNI INT8 Quantization", icon="⚡")
    with status_spinner("Quantizing CodeGen ONNX model to INT8..."):
        quantizer = ORTQuantizer.from_pretrained(onnx_model)
        dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=ONNX_PATH, quantization_config=dqconfig)

        quant_model = ORTModelForCausalLM.from_pretrained(ONNX_PATH, file_name=QUANTIZED_MODEL_FILE)
        quant_pipe = create_generation_pipe(quant_model, tokenizer)

    with status_spinner(f"Benchmarking Quantized ONNX CPU across {BENCHMARK_RUNS} runs..."):
        quant_durations = benchmark_pipe_execution(quant_pipe, BENCHMARK_PROMPT)
        profiles.append(compute_latency_distribution("ONNX Runtime (Quantized INT8)", quant_durations))

    # Phase 4: Comparative Performance Metrics
    render_step(4, "Evaluating Latency Percentiles & Throughput", icon="📊")
    render_metrics_table(profiles)

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Tail Latency (P99) Matters",
                "In interactive coding copilots (like GitHub Copilot or IDE autocompletion), P99 latency determines whether developers feel typing lag.",
            ),
            (
                "Optimum Dynamic Quantization",
                "Dynamically quantizes activations to INT8 at runtime while weights are stored in INT8, achieving near-optimal throughput on standard CPU servers without GPU dependencies.",
            ),
            (
                "Throughput vs Single-User Latency",
                "High throughput (sequences/second) enables serving multiple concurrent IDE code completion streams cost-effectively.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
