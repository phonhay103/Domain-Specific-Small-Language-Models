"""Quantization of a Finetuned BERT Model with ONNX Runtime & Optimum.

Companion script for Chapter 6 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Introduces quantization of an encoder-only classification model (Banking77)
using ONNX Runtime dynamic AVX512-VNNI quantization (underlying HF Optimum).
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Third-party
import numpy as np
import onnxruntime as ort
import torch
from datasets import load_dataset
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Common functional & UI utilities
from common.functional import calculate_speedup
from common.ui import (
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
DATASET_ID = "mteb/banking77"
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
EVAL_SAMPLE_LIMIT = 500


# ---------------------------------------------------------------------------
# Pure Functions & Helpers
# ---------------------------------------------------------------------------
def compute_size_reduction(fp32_mb: float, int8_mb: float) -> ModelArtifactSizes:
    """Pure calculation of artifact compression."""
    ratio = (1.0 - int8_mb / fp32_mb) * 100.0 if fp32_mb > 0 else 0.0
    return ModelArtifactSizes(fp32_size_mb=fp32_mb, int8_size_mb=int8_mb, reduction_ratio=ratio)


def export_model_to_onnx(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    output_path: Path,
) -> None:
    """Export PyTorch sequence classification model to ONNX format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_text = "Sample input text for ONNX computation graph tracing"
    inputs = tokenizer(dummy_text, return_tensors="pt")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        dynamo=False,
    )


def quantize_onnx_avx512(model_path: Path, output_path: Path) -> None:
    """Apply dynamic INT8 quantization with AVX-512 VNNI configuration."""
    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm", "Gather"],
        per_channel=False,
        reduce_range=False,
    )


def create_onnx_classifier(
    session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    id2label: Mapping[int, str],
) -> Callable[[str], Mapping[str, Any]]:
    """Create a classification pipeline closure over an ONNX Runtime session."""

    def predict(text: str) -> Mapping[str, Any]:
        inputs = tokenizer(text, return_tensors="np")
        outputs = session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
        )
        logits = outputs[0][0]
        # Pure softmax calculation
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        predicted_id = int(np.argmax(probabilities))
        return {
            "label": id2label.get(predicted_id, str(predicted_id)),
            "score": float(probabilities[predicted_id]),
        }

    return predict


def evaluate_onnx_accuracy(
    session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    dataset: Any,
    batch_size: int = 64,
) -> float:
    """Evaluate intent classification accuracy on a dataset split in batches."""
    texts = dataset["text"]
    labels = dataset["label"]
    total = len(texts)
    correct = 0

    for idx in range(0, total, batch_size):
        b_texts = texts[idx : idx + batch_size]
        b_labels = labels[idx : idx + batch_size]
        enc = tokenizer(b_texts, padding=True, truncation=True, return_tensors="np")
        outputs = session.run(
            None,
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            },
        )
        preds = np.argmax(outputs[0], axis=-1)
        correct += int(np.sum(preds == b_labels))

    return float(correct / total) if total > 0 else 0.0


def measure_pipeline_latency(payload_prompt: str, pipe: Callable[[str], Any]) -> LatencyBenchmarkSummary:
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
    """Execute Optimum & ONNX Runtime BERT quantization and benchmarking pipeline."""
    render_banner(
        title="Quantization of Finetuned BERT with ONNX Runtime & Optimum",
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
    fp32_onnx_path = ONNX_PATH / ORIGINAL_MODEL_NAME
    quant_onnx_path = ONNX_PATH / QUANTIZED_MODEL_NAME

    with status_spinner(f"Exporting '{MODEL_ID}' to ONNX..."):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        pt_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        pt_model.eval()
        export_model_to_onnx(pt_model, tokenizer, fp32_onnx_path)
        tokenizer.save_pretrained(ONNX_PATH)

    session_fp32 = ort.InferenceSession(str(fp32_onnx_path), providers=["CPUExecutionProvider"])
    render_device_info("CPU", model=pt_model)

    vanilla_clf = create_onnx_classifier(session_fp32, tokenizer, pt_model.config.id2label)
    test_query = "Could you assist me in checking my card validity?"
    sample_pred = vanilla_clf(test_query)

    render_card(
        title="Vanilla Model Prediction",
        content=(
            f"[text.muted]Query:[/text.muted] {test_query}\n"
            f"[status.success]Predicted Intent:[/status.success] [text.highlight]{sample_pred['label']}[/text.highlight] "
            f"([text.dim]Confidence: {sample_pred['score']:.4f}[/text.dim])"
        ),
        icon="✔",
    )

    # Step 2: Quantizing Model
    render_step(2, "Applying Dynamic AVX-512 VNNI INT8 Quantization", icon="⚙️")
    with status_spinner("Applying dynamic INT8 quantization via ONNX Runtime..."):
        quantize_onnx_avx512(fp32_onnx_path, quant_onnx_path)

    orig_mb = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    q8_mb = os.path.getsize(quant_onnx_path) / (1024 * 1024)
    render_sizes_table(compute_size_reduction(orig_mb, q8_mb))

    # Step 3: Verifying Quantized Pipeline
    render_step(3, "Verifying Quantized Pipeline Inference", icon="✨")
    with status_spinner("Loading quantized ONNX runtime session..."):
        session_int8 = ort.InferenceSession(str(quant_onnx_path), providers=["CPUExecutionProvider"])
        q8_clf = create_onnx_classifier(session_int8, tokenizer, pt_model.config.id2label)

    q_sample_pred = q8_clf(test_query)
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
        eval_dataset = load_dataset(DATASET_ID, split=f"test[:{EVAL_SAMPLE_LIMIT}]")
        accuracy = evaluate_onnx_accuracy(session_int8, tokenizer, eval_dataset)
    render_accuracy_table(accuracy)

    # Step 5: Latency Benchmarking
    render_step(5, "Benchmarking Latency & CPU Throughput", icon="🚀")
    orig_summary = measure_pipeline_latency(BENCHMARK_PROMPT, vanilla_clf)
    q8_summary = measure_pipeline_latency(BENCHMARK_PROMPT, q8_clf)
    render_latency_comparison(orig_summary, q8_summary)

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "HuggingFace Optimum & ONNX Runtime AVX-512 VNNI",
                "Uses Intel Vector Neural Network Instructions (VNNI) on modern CPUs to execute 8-bit integer dot products directly in hardware registers.",
            ),
            (
                "Accuracy Retention",
                "On Banking77 classification, Dynamic INT8 preserves >99% of original FP32 accuracy while halving disk/memory footprint.",
            ),
            (
                "CPU Edge Deployment",
                "ONNX Runtime dynamic quantization is ideal for cost-effective edge CPU microservices where GPUs are unavailable.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
