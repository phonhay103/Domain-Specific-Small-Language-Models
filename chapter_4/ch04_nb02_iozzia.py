"""Benchmarking DeepSpeed-Inference Acceleration.

Companion script for Chapter 4 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Demonstrates initializing DeepSpeed-Inference on an autoregressive language
model (GPT-2) and comparing its generation latency against the base PyTorch model.
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Third-party
import deepspeed
import numpy as np
import torch
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
class LatencyProfile:
    """Immutable latency summary statistics."""

    avg_ms: float
    std_ms: float
    p95_ms: float


@dataclass(frozen=True)
class EngineComparison:
    """Comparative benchmarking results for PyTorch vs DeepSpeed."""

    base_profile: LatencyProfile
    ds_profile: LatencyProfile
    speedup_factor: float


MODEL_ID = "openai-community/gpt2"
PROMPT = "The greatest technological advancement of the 21st century is"
BENCHMARK_PROMPTS: tuple[str, ...] = (
    "Artificial intelligence in healthcare will",
    "Quantum computing enables researchers to",
    "Renewable energy systems are becoming",
    "Autonomous vehicles rely heavily on",
    "Large language models have transformed",
)

WARMUP_RUNS = 2
BENCHMARK_RUNS = 10
MAX_NEW_TOKENS = 64
REPLACE_METHOD = "auto"


# ---------------------------------------------------------------------------
# Pure Functions & Benchmarking Helpers
# ---------------------------------------------------------------------------
def compute_latency_profile(raw_latencies_ms: Sequence[float]) -> LatencyProfile:
    """Pure calculation of statistical latency percentiles and variance."""
    return LatencyProfile(
        avg_ms=float(np.mean(raw_latencies_ms)),
        std_ms=float(np.std(raw_latencies_ms)),
        p95_ms=float(np.percentile(raw_latencies_ms, 95)),
    )


def measure_engine_latency(
    model: Any,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> LatencyProfile:
    """Benchmark autoregressive token generation across warmups and timed runs."""
    for _ in range(WARMUP_RUNS):
        inputs = tokenizer(prompts[0], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for prompt in prompts[:BENCHMARK_RUNS]:
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return compute_latency_profile(latencies_ms)


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_comparison_table(comp: EngineComparison) -> None:
    """Render latency speedup comparison table."""
    columns = [
        ("Engine / Configuration", STYLE_PRIMARY, "left"),
        ("Average Latency", STYLE_WARNING, "right"),
        ("Std Dev", STYLE_TEXT, "right"),
        ("P95 Tail Latency", STYLE_SECONDARY, "right"),
        ("Speedup Factor", STYLE_SUCCESS, "right"),
    ]
    rows = [
        (
            "Base PyTorch (FP16)",
            f"{comp.base_profile.avg_ms:.2f} ms",
            f"± {comp.base_profile.std_ms:.2f}",
            f"{comp.base_profile.p95_ms:.2f} ms",
            "1.00x",
        ),
        (
            "DeepSpeed-Inference (Fused)",
            f"{comp.ds_profile.avg_ms:.2f} ms",
            f"± {comp.ds_profile.std_ms:.2f}",
            f"{comp.ds_profile.p95_ms:.2f} ms",
            f"{comp.speedup_factor:.2f}x",
        ),
    ]
    console.print(create_table("Inference Latency & Throughput Benchmark", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute DeepSpeed-Inference kernel acceleration comparison."""
    render_banner(
        title="DeepSpeed-Inference Kernel Acceleration Benchmark",
        subtitle="Chapter 4: Domain-Specific Small Language Models",
        metadata={
            "Base Model": MODEL_ID,
            "Acceleration": "DeepSpeed CUDA Kernel Injection",
            "Benchmark Runs": str(BENCHMARK_RUNS),
        },
        icon="🚀",
    )

    # Step 1: Loading Base Model
    render_step(1, "Loading Base Model onto GPU in FP16", icon="📋")
    with status_spinner(f"Loading '{MODEL_ID}' with FP16 precision..."):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()

    render_card(
        "Device Status",
        f"Model loaded on device [text.highlight]{model.device}[/text.highlight] in FP16 precision.",
        icon="✔",
    )

    # Step 2: Base PyTorch Generation Sample
    render_step(2, "Generating Baseline Output with Standard PyTorch", icon="💬")
    inputs = tokenizer(PROMPT, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with status_spinner("Generating sample output with base PyTorch model..."):
        with torch.inference_mode():
            base_out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        base_text = tokenizer.decode(base_out_ids[0], skip_special_tokens=True)

    render_card("Base PyTorch Output", base_text, icon="📄")

    # Step 3: Initializing DeepSpeed-Inference
    render_step(3, "Initializing DeepSpeed-Inference with Fused CUDA Kernels", icon="⚙️")
    with status_spinner("Injecting fused attention and LayerNorm kernels..."):
        ds_engine = deepspeed.init_inference(
            model,
            mp_size=1,
            dtype=torch.float16,
            replace_with_kernel_inject=True,
            replace_method=REPLACE_METHOD,
        )
        ds_model = ds_engine.module
    render_card("Engine Status", "DeepSpeed-Inference engine successfully initialized with kernel injection.", icon="✔")

    # Step 4: DeepSpeed Generation Sample
    render_step(4, "Generating Sample Output with DeepSpeed Fused Kernels", icon="✨")
    with status_spinner("Generating sample with DeepSpeed-Inference..."):
        with torch.inference_mode():
            ds_out_ids = ds_model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        ds_text = tokenizer.decode(ds_out_ids[0], skip_special_tokens=True)

    render_card("DeepSpeed Accelerated Output", ds_text, icon="⚡")

    # Step 5: Comparative Latency Benchmarking
    render_step(5, "Benchmarking Latency, Variance, and P95 Scaling", icon="📊")
    extended_prompts = list(BENCHMARK_PROMPTS) * (BENCHMARK_RUNS // len(BENCHMARK_PROMPTS) + 1)

    with status_spinner(f"Benchmarking Base PyTorch across {BENCHMARK_RUNS} iterations..."):
        base_profile = measure_engine_latency(model, tokenizer, extended_prompts)

    with status_spinner(f"Benchmarking DeepSpeed-Inference across {BENCHMARK_RUNS} iterations..."):
        ds_profile = measure_engine_latency(ds_model, tokenizer, extended_prompts)

    comp = EngineComparison(
        base_profile=base_profile,
        ds_profile=ds_profile,
        speedup_factor=calculate_speedup(base_profile.avg_ms, ds_profile.avg_ms),
    )
    render_comparison_table(comp)

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Kernel Injection",
                "DeepSpeed replaces PyTorch Transformer layers with hand-tuned CUDA kernels that fuse Multi-Head Attention and LayerNorm directly into GPU registers.",
            ),
            (
                "Memory Bandwidth Reduction",
                "Fusing kernels eliminates round-trip High-Bandwidth Memory (HBM) read/writes between individual operations.",
            ),
            (
                "Drop-in Speedup",
                "deepspeed.init_inference() requires zero model retraining or weight conversion while delivering immediate 1.5x-2.5x latency improvements.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
