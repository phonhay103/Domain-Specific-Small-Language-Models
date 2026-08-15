"""Benchmarking Python Code Generation with Vanilla and 8-bit Quantized StarCoder2 Models.

Companion to Chapter 7 of "Domain Specific LLMs in Action" by Guglielmo Iozzia,
Manning Publications, 2024.

Benchmarks inference performance (latency percentiles, throughput) when generating Python code
using a vanilla StarCoder2-3B model and after 8-bit quantization with bitsandbytes.
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

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
    render_device_info,
    render_card,
    render_code_block,
    render_step,
    render_takeaways,
    status_spinner,
)


# ---------------------------------------------------------------------------
# Immutable Domain Records & Constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StarCoderLatencyProfile:
    """Latency percentiles and throughput distribution metrics."""

    variant_name: str
    avg_latency_ms: float
    p50_ms: float
    p75_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    throughput_seq_per_sec: float


MODEL_ID = "bigcode/starcoder2-3b"
CHECKPOINT_SAVE_DIR = "local-pt-checkpoint"
CHECKPOINT_8BIT_SAVE_DIR = "local-8bit-checkpoint"
BENCHMARK_PROMPT = "def print_hello_world():"
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
PIPELINE_TEMPERATURE = 0.2
PIPELINE_TOP_P = 0.95
PIPELINE_MAX_LENGTH = 14


# ---------------------------------------------------------------------------
# Pure Functions & Benchmarking Logic
# ---------------------------------------------------------------------------
def compute_latency_percentiles(variant_name: str, raw_durations_sec: Sequence[float]) -> StarCoderLatencyProfile:
    """Pure calculation of latency percentiles and throughput."""
    times_ms = np.array(raw_durations_sec) * 1000.0
    avg_ms = float(np.mean(times_ms))
    throughput = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    return StarCoderLatencyProfile(
        variant_name=variant_name,
        avg_latency_ms=avg_ms,
        p50_ms=float(np.percentile(times_ms, 50)),
        p75_ms=float(np.percentile(times_ms, 75)),
        p90_ms=float(np.percentile(times_ms, 90)),
        p95_ms=float(np.percentile(times_ms, 95)),
        p99_ms=float(np.percentile(times_ms, 99)),
        throughput_seq_per_sec=throughput,
    )


def benchmark_pipe_runs(pipe: Any, prompt: str) -> list[float]:
    """Pure timing loop collecting latency durations in seconds."""
    for _ in range(WARMUP_ITERATIONS):
        _ = pipe(prompt)

    durations: list[float] = []
    for _ in range(BENCHMARK_ITERATIONS):
        start = perf_counter()
        _ = pipe(prompt)
        durations.append(perf_counter() - start)
    return durations


def create_code_pipe(model: Any, tokenizer: Any) -> Any:
    """Pure factory for text-generation pipeline."""
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        use_cache=True,
        temperature=PIPELINE_TEMPERATURE,
        top_p=PIPELINE_TOP_P,
        max_length=PIPELINE_MAX_LENGTH,
    )


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_latency_profile_table(profiles: Sequence[StarCoderLatencyProfile]) -> None:
    """Render latency percentiles and throughput table."""
    columns = [
        ("Model Configuration", STYLE_PRIMARY, "left"),
        ("Avg Latency", STYLE_WARNING, "right"),
        ("P50 Median", STYLE_TEXT, "right"),
        ("P90 Latency", STYLE_TEXT, "right"),
        ("P95 Latency", STYLE_SECONDARY, "right"),
        ("P99 Tail", STYLE_WARNING, "right"),
        ("Throughput", STYLE_SUCCESS, "right"),
    ]
    rows = [
        (
            p.variant_name,
            f"{p.avg_latency_ms:.2f} ms",
            f"{p.p50_ms:.2f} ms",
            f"{p.p90_ms:.2f} ms",
            f"{p.p95_ms:.2f} ms",
            f"{p.p99_ms:.2f} ms",
            f"{p.throughput_seq_per_sec:.2f} seq/s",
        )
        for p in profiles
    ]
    console.print(create_table(f"StarCoder2-3B Benchmark Summary ({BENCHMARK_ITERATIONS} iterations)", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute StarCoder2-3B precision benchmark pipeline."""
    render_banner(
        title="Benchmarking StarCoder2-3B: bfloat16 vs 8-bit Quantized",
        subtitle="Chapter 7: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_ID,
            "Prompt": f'"{BENCHMARK_PROMPT}"',
            "Iterations": str(BENCHMARK_ITERATIONS),
        },
        icon="🚀",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    render_device_info(device)
    profiles: list[StarCoderLatencyProfile] = []

    # Phase 1: Vanilla StarCoder2-3B (bfloat16)
    render_step(1, "Vanilla StarCoder2-3B (bfloat16) Pipeline", icon="📋")
    with status_spinner(f"Loading '{MODEL_ID}' in bfloat16 onto {device}..."):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)
        model.eval()
        vanilla_pipe = create_code_pipe(model, tokenizer)

    sample_out = vanilla_pipe(BENCHMARK_PROMPT)[0]["generated_text"]
    render_code_block(sample_out, language="python", title="Vanilla StarCoder2 Output Preview")

    tokenizer.save_pretrained(CHECKPOINT_SAVE_DIR)
    model.save_pretrained(CHECKPOINT_SAVE_DIR)

    with status_spinner(f"Benchmarking vanilla model across {BENCHMARK_ITERATIONS} runs..."):
        durations_fp = benchmark_pipe_runs(vanilla_pipe, BENCHMARK_PROMPT)
        profiles.append(compute_latency_percentiles("Vanilla StarCoder2 (bfloat16)", durations_fp))

    # Phase 2: 8-Bit Quantized StarCoder2-3B
    render_step(2, "8-Bit Quantized StarCoder2-3B via BitsAndBytes", icon="⚡")
    model.cpu()
    del model
    del vanilla_pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with status_spinner("Loading 8-bit quantized model..."):
        q_config = BitsAndBytesConfig(load_in_8bit=True)
        tokenizer_q = AutoTokenizer.from_pretrained(CHECKPOINT_SAVE_DIR)
        model_q = AutoModelForCausalLM.from_pretrained(CHECKPOINT_SAVE_DIR, quantization_config=q_config)
        model_q.eval()
        q_pipe = create_code_pipe(model_q, tokenizer_q)

    q_sample_out = q_pipe(BENCHMARK_PROMPT)[0]["generated_text"]
    render_code_block(q_sample_out, language="python", title="Quantized StarCoder2 Output Preview")

    with status_spinner(f"Benchmarking 8-bit model across {BENCHMARK_ITERATIONS} runs..."):
        durations_q = benchmark_pipe_runs(q_pipe, BENCHMARK_PROMPT)
        profiles.append(compute_latency_percentiles("Quantized StarCoder2 (8-bit)", durations_q))

    # Phase 3: Performance Metrics
    render_step(3, "Evaluating Latency Distributions & Throughput", icon="📊")
    render_latency_profile_table(profiles)

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "StarCoder2 Architecture",
                "Optimized for Fill-in-the-Middle (FIM) and code completion tasks with multi-query attention (MQA).",
            ),
            (
                "bfloat16 vs 8-bit Quantization",
                "8-bit weights cut GPU memory from ~6GB down to ~3GB VRAM, allowing 3B code models to fit in consumer GPU environments while preserving syntactic generation quality.",
            ),
            (
                "Latency Distribution Profile",
                "Measuring P50 to P99 reveals whether quantization introduces memory access spikes or maintains steady generation speed.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
