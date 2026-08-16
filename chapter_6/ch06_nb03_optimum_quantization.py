"""Quantization of a Finetuned BERT Model with HF's Optimum.

Companion script for Chapter 6 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Introduces quantization of an encoder-only classification model (Banking77)
using Hugging Face Optimum and ONNX Runtime dynamic AVX512-VNNI quantization.
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import os
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
from datasets import load_dataset
from evaluate import evaluator
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer, pipeline

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
    render_device_info,
    render_step,
    render_takeaways,
    status_spinner,
)


# ---------------------------------------------------------------------------
# Immutable Domain Records & Constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelArtifactSizes:
    """File size comparison between FP32 and INT8 ONNX binaries."""

    fp32_size_mb: float
    int8_size_mb: float
    reduction_ratio: float


@dataclass(frozen=True)
class LatencyBenchmarkSummary:
    """Latency benchmark metrics across iterations."""

    avg_ms: float
    std_ms: float
    p95_ms: float


MODEL_ID = "optimum/distilbert-base-uncased-finetuned-banking77"
DATASET_ID = "PolyAI/banking77"
ONNX_PATH = Path("onnx")
ORIGINAL_MODEL_NAME = "model.onnx"
QUANTIZED_MODEL_NAME = "model_quantized.onnx"
BENCHMARK_PROMPT = (
    "Dear Sir/Madam, my name is William. I am getting in touch because I didn't "
    "get a response from you yet. What actions do I need to do to get my new card "
    "which I have requested 3 weeks ago? Please help me and answer this email as "
    "soon as possible. Have a nice rest of the day. Best Regards."
) * 2
VANILLA_ACCURACY_BASELINE = 0.925
WARMUP_RUNS = 10
BENCHMARK_RUNS = 300


# ---------------------------------------------------------------------------
# Pure Functions & Helpers
# ---------------------------------------------------------------------------
def compute_size_reduction(fp32_mb: float, int8_mb: float) -> ModelArtifactSizes:
    """Pure calculation of artifact compression."""
    ratio = (1.0 - int8_mb / fp32_mb) * 100.0 if fp32_mb > 0 else 0.0
    return ModelArtifactSizes(fp32_size_mb=fp32_mb, int8_size_mb=int8_mb, reduction_ratio=ratio)


def measure_pipeline_latency(payload_prompt: str, pipe: Any) -> LatencyBenchmarkSummary:
    """Pure timing loop measuring pipeline latency and percentiles."""
    for _ in range(WARMUP_RUNS):
        _ = pipe(payload_prompt)

    latencies = []
    for _ in range(BENCHMARK_RUNS):
        start_time = perf_counter()
        _ = pipe(payload_prompt)
        latencies.append(perf_counter() - start_time)

    avg_ms = float(1000.0 * np.mean(latencies))
    std_ms = float(1000.0 * np.std(latencies))
    p95_ms = float(1000.0 * np.percentile(latencies, 95))

    return LatencyBenchmarkSummary(avg_ms=avg_ms, std_ms=std_ms, p95_ms=p95_ms)


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_sizes_table(sizes: ModelArtifactSizes) -> None:
    """Render file size compression table."""
    columns = [
        ("Model Artifact", STYLE_PRIMARY, "left"),
        ("Disk Size (MB)", STYLE_WARNING, "right"),
        ("Compression Ratio", STYLE_SUCCESS, "right"),
    ]
    rows = [
        ("Original FP32 ONNX", f"{sizes.fp32_size_mb:.2f} MB", "1.00x"),
        (
            "Quantized INT8 ONNX",
            f"{sizes.int8_size_mb:.2f} MB",
            f"-{sizes.reduction_ratio:.1f}% ({sizes.fp32_size_mb / sizes.int8_size_mb:.2f}x smaller)",
        ),
    ]
    console.print(create_table("ONNX Model File Size Comparison", columns, rows))
    pause()


def render_accuracy_table(q_acc: float) -> None:
    """Render classification accuracy retention table."""
    columns = [
        ("Model Configuration", STYLE_PRIMARY, "left"),
        ("Accuracy (%)", STYLE_SUCCESS, "right"),
        ("Retention vs FP32", STYLE_WARNING, "right"),
    ]
    rows = [
        ("Vanilla FP32 Baseline", f"{VANILLA_ACCURACY_BASELINE * 100:.2f}%", "100.00%"),
        ("Dynamic INT8 Quantized", f"{q_acc * 100:.2f}%", f"{q_acc / VANILLA_ACCURACY_BASELINE * 100:.2f}%"),
    ]
    console.print(create_table("Accuracy Retention on Banking77 Test Set", columns, rows))
    pause()


def render_latency_comparison(orig: LatencyBenchmarkSummary, q8: LatencyBenchmarkSummary) -> None:
    """Render latency speedup table."""
    columns = [
        ("Model Variant", STYLE_PRIMARY, "left"),
        ("Average Latency", STYLE_WARNING, "right"),
        ("Std Dev", STYLE_TEXT, "right"),
        ("P95 Latency", STYLE_SECONDARY, "right"),
        ("Speedup Factor", STYLE_SUCCESS, "right"),
    ]
    rows = [
        ("Vanilla FP32", f"{orig.avg_ms:.2f} ms", f"± {orig.std_ms:.2f}", f"{orig.p95_ms:.2f} ms", "1.00x"),
        (
            "Quantized INT8",
            f"{q8.avg_ms:.2f} ms",
            f"± {q8.std_ms:.2f}",
            f"{q8.p95_ms:.2f} ms",
            f"{calculate_speedup(orig.avg_ms, q8.avg_ms):.2f}x",
        ),
    ]
    console.print(create_table("Inference Latency & Speedup Benchmark (CPU)", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute Optimum BERT quantization and benchmarking pipeline."""
    render_banner(
        title="Quantization of Finetuned BERT with Hugging Face Optimum",
        subtitle="Chapter 6: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_ID,
            "Dataset": DATASET_ID,
            "Target Architecture": "Intel AVX-512 VNNI",
        },
        icon="🚀",
    )

    # Step 1: Loading & Exporting FP32 Model
    render_step(1, "Exporting FP32 DistilBERT to ONNX Format", icon="📋")
    with status_spinner(f"Exporting '{MODEL_ID}' to ONNX..."):
        model = ORTModelForSequenceClassification.from_pretrained(MODEL_ID, export=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model.save_pretrained(ONNX_PATH)
        tokenizer.save_pretrained(ONNX_PATH)
    render_device_info(model.device, model=model)

    vanilla_clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
    test_query = "Could you assist me in checking my card validity?"
    sample_pred = vanilla_clf(test_query)[0]

    render_card(
        title="Vanilla Model Prediction",
        content=(
            f"[text.muted]Query:[/text.muted] {test_query}\n"
            f"[status.success]Predicted Intent:[/status.success] [text.highlight]{sample_pred['label']}[/text.highlight] "
            f"([text.dim]Confidence: {sample_pred['score']:.4f}[/text.dim])"
        ),
        icon="✔",
    )

    # Step 2: Quantizing Model with Optimum
    render_step(2, "Applying Dynamic AVX-512 VNNI INT8 Quantization", icon="⚙️")
    with status_spinner("Applying dynamic quantization via Optimum..."):
        dynamic_quantizer = ORTQuantizer.from_pretrained(model)
        dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        dynamic_quantizer.quantize(save_dir=ONNX_PATH, quantization_config=dqconfig)

    orig_mb = os.path.getsize(ONNX_PATH / ORIGINAL_MODEL_NAME) / (1024 * 1024)
    q8_mb = os.path.getsize(ONNX_PATH / QUANTIZED_MODEL_NAME) / (1024 * 1024)
    render_sizes_table(compute_size_reduction(orig_mb, q8_mb))

    # Step 3: Verifying Quantized Pipeline
    render_step(3, "Verifying Quantized Pipeline Inference", icon="✨")
    with status_spinner("Loading quantized ONNX runtime session..."):
        q_model = ORTModelForSequenceClassification.from_pretrained(ONNX_PATH, file_name=QUANTIZED_MODEL_NAME)
        tokenizer_q = AutoTokenizer.from_pretrained(ONNX_PATH)
        q8_clf = pipeline("text-classification", model=q_model, tokenizer=tokenizer_q)

    q_sample_pred = q8_clf(test_query)[0]
    render_card(
        title="Quantized Model Prediction",
        content=(
            f"[text.muted]Query:[/text.muted] {test_query}\n"
            f"[status.success]Predicted Intent:[/status.success] [text.highlight]{q_sample_pred['label']}[/text.highlight] "
            f"([text.dim]Confidence: {q_sample_pred['score']:.4f}[/text.dim])"
        ),
        icon="⚡",
    )

    # Step 4: Evaluating Accuracy
    render_step(4, "Evaluating Intent Accuracy on Banking77 Test Split", icon="📊")
    with status_spinner("Running batch evaluation on test split..."):
        eval_engine = evaluator("text-classification")
        eval_dataset = load_dataset(DATASET_ID, split="test")
        results = eval_engine.compute(
            model_or_pipeline=q8_clf,
            data=eval_dataset,
            metric="accuracy",
            input_column="text",
            label_column="label",
            label_mapping=q_model.config.label2id,
            strategy="simple",
        )
    render_accuracy_table(results["accuracy"])

    # Step 5: Latency Benchmarking
    render_step(5, "Benchmarking Latency & CPU Throughput", icon="🚀")
    orig_summary = measure_pipeline_latency(BENCHMARK_PROMPT, vanilla_clf)
    q8_summary = measure_pipeline_latency(BENCHMARK_PROMPT, q8_clf)
    render_latency_comparison(orig_summary, q8_summary)

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "HuggingFace Optimum with AVX-512 VNNI",
                "Uses Intel Vector Neural Network Instructions (VNNI) on modern CPUs to execute 8-bit integer dot products in hardware registers.",
            ),
            (
                "Accuracy Retention",
                "On Banking77 classification, Dynamic INT8 preserves >99% of original FP32 accuracy while halving disk/memory size.",
            ),
            (
                "CPU Edge Deployment",
                "ONNX Runtime dynamic quantization is ideal for edge CPU microservices where GPUs are unavailable.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
