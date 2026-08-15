"""Benchmarking Different Versions of a Small Language Model Before Deployment on an Endpoint.

Companion to chapter 11 of "Domain Specific LLMs in Action"
by Guglielmo Iozzia (Manning Publications, 2024).

Benchmarks different runtime representations of GPT-2 small (PyTorch CPU, Base ONNX,
Optimized ONNX, and Optimized FP16) across varying token lengths before FastAPI endpoint deployment.
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import gc
import sys
import timeit
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Third-party
import numpy as np
import torch
from onnxruntime import InferenceSession
from onnxruntime.transformers.optimizer import optimize_model
from transformers import BatchEncoding, GPT2Model, GPT2Tokenizer

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
class PredeploymentBenchmarkResult:
    """Benchmark timing for a specific model deployment variant."""

    variant_name: str
    avg_latency_ms: float
    speedup_ratio: float


@dataclass(frozen=True)
class SequenceLengthScalingRow:
    """Latency across variants for a specific input sequence length."""

    sequence_length: int
    base_onnx_ms: float
    opt_onnx_ms: float
    opt_fp16_ms: float


DEVICE = "cpu"
MODEL_ID = "openai-community/gpt2"
MODEL_SAVE_PATH = Path("gpt2")
ONNX_MODEL_PATH = "gpt2_onnx.onnx"
OPTIMIZED_ONNX_PATH = "gpt2_optimized.onnx"
OPTIMIZED_FP16_MODEL_PATH = "optimized_fp16.onnx"
BENCHMARK_PROMPT = "Today is Saturday and"
MAX_SEQUENCE_LENGTH = 1024
ORT_PROVIDERS = ["CPUExecutionProvider"]
BENCHMARK_SEQUENCE_LENGTHS = (1, 4, 64, 256, 512, 1024)
BENCHMARK_WARMUP_RUNS = 10
BENCHMARK_TIMED_RUNS = 100
ONNX_OPSET_VERSION = 18


# ---------------------------------------------------------------------------
# Pure Functions & Benchmarking Helpers
# ---------------------------------------------------------------------------
def measure_fn_latency_ms(fn: Callable[[], Any]) -> float:
    """Pure benchmarking measurement using timeit."""
    for _ in range(BENCHMARK_WARMUP_RUNS):
        fn()
    seconds_per_iter = timeit.timeit(fn, number=BENCHMARK_TIMED_RUNS) / BENCHMARK_TIMED_RUNS
    return float(seconds_per_iter * 1000.0)


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_summary_table(results: Sequence[PredeploymentBenchmarkResult]) -> None:
    """Render single prompt candidate latency table."""
    columns = [
        ("Deployment Candidate", STYLE_PRIMARY, "left"),
        ("Average Latency (ms)", STYLE_WARNING, "right"),
        ("Speedup vs PyTorch", STYLE_SUCCESS, "right"),
    ]
    rows = [(r.variant_name, f"{r.avg_latency_ms:.3f} ms", f"{r.speedup_ratio:.2f}x") for r in results]
    console.print(create_table("Pre-Deployment Candidate Latency Comparison", columns, rows))
    pause()


def render_scaling_table(rows_data: Sequence[SequenceLengthScalingRow]) -> None:
    """Render sequence length latency scaling table."""
    columns = [
        ("Sequence Length", STYLE_PRIMARY, "center"),
        ("Base ONNX (ms)", STYLE_TEXT, "right"),
        ("Optimized ONNX (ms)", STYLE_WARNING, "right"),
        ("Optimized FP16 (ms)", STYLE_SUCCESS, "right"),
    ]
    rows = [
        (
            f"{r.sequence_length} tokens",
            f"{r.base_onnx_ms:.2f} ms",
            f"{r.opt_onnx_ms:.2f} ms",
            f"{r.opt_fp16_ms:.2f} ms",
        )
        for r in rows_data
    ]
    console.print(create_table("Latency Scaling by Sequence Length (ms)", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute pre-deployment model benchmarking pipeline."""
    render_banner(
        title="Benchmarking SLM Deployments: PyTorch vs ONNX vs FP16",
        subtitle="Chapter 11: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_ID,
            "Target Endpoint": "FastAPI Microservice",
            "Timed Iterations": str(BENCHMARK_TIMED_RUNS),
        },
        icon="🚀",
    )

    # Step 1: PyTorch CPU Baseline Benchmark
    render_step(1, "Loading Model & PyTorch CPU Baseline Benchmark", icon="📋")
    with status_spinner(f"Loading '{MODEL_ID}'..."):
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
        model: GPT2Model = GPT2Model.from_pretrained(MODEL_ID)
        model.eval()
        model.save_pretrained(MODEL_SAVE_PATH)

    inputs_base = tokenizer(BENCHMARK_PROMPT, return_tensors="pt").to(DEVICE)
    pt_ms = measure_fn_latency_ms(lambda m=model, inp=inputs_base: m(**inp))
    render_card("PyTorch Baseline", f"PyTorch CPU Latency: [brand.secondary]{pt_ms:.3f} ms[/brand.secondary]", icon="✔")

    # Step 2: Exporting PyTorch to ONNX Graph
    render_step(2, "Exporting Static Computation Graph to ONNX", icon="⚙️")
    input_ids = tokenizer(BENCHMARK_PROMPT, add_special_tokens=True, return_attention_mask=False, return_tensors="pt")
    input_tensor = input_ids["input_ids"].type(torch.int32)

    with status_spinner("Tracing computation graph..."):
        torch.onnx.export(
            model,
            f=ONNX_MODEL_PATH,
            args=(input_tensor,),
            input_names=["input_ids"],
            output_names=["logits"],
            quantization=False,
            var_output_seq=True,
            do_constant_folding=True,
            opset_version=ONNX_OPSET_VERSION,
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
        )
    render_card("ONNX Export", f"Saved to [text.highlight]{ONNX_MODEL_PATH}[/text.highlight]", icon="💾")
    del model
    gc.collect()

    # Step 3: Base ONNX Model Benchmark
    render_step(3, "Benchmarking Base ONNX Session", icon="⏱️")
    sess = InferenceSession(ONNX_MODEL_PATH, providers=ORT_PROVIDERS)
    encodings_dict = tokenizer.batch_encode_plus([BENCHMARK_PROMPT])
    ort_input_ids = torch.tensor(encodings_dict["input_ids"], dtype=torch.int32).cpu().numpy()
    ort_inputs = {"input_ids": ort_input_ids}

    base_onnx_ms = measure_fn_latency_ms(lambda: sess.run(None, ort_inputs))
    render_card(
        "Base ONNX", f"Base ONNX CPU Latency: [brand.secondary]{base_onnx_ms:.3f} ms[/brand.secondary]", icon="✔"
    )
    del sess
    gc.collect()

    # Step 4: ONNX Graph Optimization
    render_step(4, "Fusing Transformer Attention Kernels", icon="⚡")
    with status_spinner("Applying graph-level operator fusion..."):
        optimized_model = optimize_model(input=ONNX_MODEL_PATH, model_type="gpt2", use_gpu=False)
        optimized_model.save_model_to_file(OPTIMIZED_ONNX_PATH)

    optimized_sess = InferenceSession(OPTIMIZED_ONNX_PATH, providers=ORT_PROVIDERS)
    opt_onnx_ms = measure_fn_latency_ms(lambda: optimized_sess.run(None, input_feed=ort_inputs))
    render_card(
        "Optimized ONNX",
        f"Optimized ONNX CPU Latency: [brand.secondary]{opt_onnx_ms:.3f} ms[/brand.secondary]",
        icon="✔",
    )
    del optimized_sess
    gc.collect()

    # Step 5: FP16 Quantized ONNX Model
    render_step(5, "Downcasting to FP16 Precision", icon="✨")
    with status_spinner("Converting optimized ONNX model to FP16..."):
        optimized_fp16_model = deepcopy(optimized_model)
        optimized_fp16_model.convert_float_to_float16()
        optimized_fp16_model.save_model_to_file(OPTIMIZED_FP16_MODEL_PATH)

    del optimized_model
    gc.collect()

    optimized_fp16_sess = InferenceSession(OPTIMIZED_FP16_MODEL_PATH, providers=ORT_PROVIDERS)
    opt_fp16_ms = measure_fn_latency_ms(lambda: optimized_fp16_sess.run(None, input_feed=ort_inputs))

    summary_results = (
        PredeploymentBenchmarkResult("PyTorch CPU Baseline", pt_ms, 1.0),
        PredeploymentBenchmarkResult("Base ONNX CPU", base_onnx_ms, calculate_speedup(pt_ms, base_onnx_ms)),
        PredeploymentBenchmarkResult("Optimized ONNX CPU", opt_onnx_ms, calculate_speedup(pt_ms, opt_onnx_ms)),
        PredeploymentBenchmarkResult("Optimized FP16 ONNX CPU", opt_fp16_ms, calculate_speedup(pt_ms, opt_fp16_ms)),
    )
    render_summary_table(summary_results)

    # Step 6: Sequence Length Scaling Comparison
    render_step(6, "Evaluating Latency Scaling by Input Token Length", icon="📊")
    tokenizer.pad_token = tokenizer.eos_token
    sess = InferenceSession(ONNX_MODEL_PATH, providers=ORT_PROVIDERS)
    optimized_sess = InferenceSession(OPTIMIZED_ONNX_PATH, providers=ORT_PROVIDERS)

    scaling_rows: list[SequenceLengthScalingRow] = []
    with status_spinner("Testing sequence lengths from 1 to 1024 tokens..."):
        for n in BENCHMARK_SEQUENCE_LENGTHS:
            txt = " ".join(["word"] * n)
            encoded = dict(
                tokenizer(
                    txt,
                    max_length=MAX_SEQUENCE_LENGTH,
                    return_tensors="np",
                    return_attention_mask=False,
                )
            )
            encoded["input_ids"] = encoded["input_ids"].astype(np.int32)

            t_base = measure_fn_latency_ms(lambda inp=encoded: sess.run(None, {"input_ids": inp["input_ids"]}))
            t_opt = measure_fn_latency_ms(lambda inp=encoded: optimized_sess.run(None, inp))
            t_fp16 = measure_fn_latency_ms(lambda inp=encoded: optimized_fp16_sess.run(None, inp))

            scaling_rows.append(
                SequenceLengthScalingRow(
                    sequence_length=n,
                    base_onnx_ms=t_base,
                    opt_onnx_ms=t_opt,
                    opt_fp16_ms=t_fp16,
                )
            )

    render_scaling_table(scaling_rows)

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Pre-Deployment Latency Profiling",
                "Benchmarking across varying sequence lengths (16 to 1024 tokens) reveals scaling characteristics before production rollout.",
            ),
            (
                "ONNX CPU Acceleration",
                "Graph fusion combined with dynamic FP16/INT8 conversion provides consistent 2x-3x speedups on standard server CPUs.",
            ),
            (
                "Memory-Bound Attention",
                "As sequence length grows, attention memory traffic scales quadratically without KV-caching or optimized fused attention kernels.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
