"""BERT Model Optimization with ONNX (CPU).

Companion script for Chapter 5 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Demonstrates exporting a BERT sequence-classification model to ONNX,
applying ONNX graph optimizations (operator fusion, constant folding),
verifying numerical parity, and benchmarking inference latency on CPU.
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Third-party
import evaluate
import numpy as np
import onnx
import onnxruntime as ort
import torch
from datasets import DatasetDict, load_dataset
from onnxruntime.transformers.optimizer import optimize_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
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
class ParityCheckResult:
    """Immutable numerical validation record."""

    configuration: str
    max_absolute_error: float
    passed: bool


@dataclass(frozen=True)
class LatencyBenchmarkSummary:
    """Immutable latency summary record."""

    pt_ms: float
    base_onnx_ms: float
    opt_onnx_ms: float


MODEL_ID = "bert-base-uncased"
OUTPUT_DIR = "./output_dir"
ONNX_DIR = f"{OUTPUT_DIR}/onnx"
BASE_ONNX_PATH = f"{ONNX_DIR}/model.onnx"
OPT_ONNX_PATH = f"{ONNX_DIR}/model_opt.onnx"
DATASET_ID = "glue"
DATASET_SUBSET = "mrpc"
BENCHMARK_ITERATIONS = 20
BENCHMARK_PROMPT = "The company reported strong financial results in the fourth quarter."


# ---------------------------------------------------------------------------
# Pure Functions & Data Preparation
# ---------------------------------------------------------------------------
def compute_evaluation_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Pure evaluation metric function for accuracy and F1 score."""
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = float(metric_acc.compute(predictions=preds, references=labels)["accuracy"])
    f1 = float(metric_f1.compute(predictions=preds, references=labels)["f1"])
    return {"accuracy": acc, "f1": f1}


def measure_pure_latency(run_fn: Callable[[], Any], iterations: int = BENCHMARK_ITERATIONS) -> float:
    """Measure average execution latency over iterations."""
    for _ in range(3):
        run_fn()

    start = time.perf_counter()
    for _ in range(iterations):
        run_fn()
    return float((time.perf_counter() - start) / iterations * 1000.0)


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_parity_table(results: Sequence[ParityCheckResult]) -> None:
    """Render numerical parity validation table."""
    columns = [
        ("Model Configuration", STYLE_PRIMARY, "left"),
        ("Max Absolute Error (L-inf)", STYLE_WARNING, "right"),
        ("Validation Status", STYLE_SUCCESS, "center"),
    ]
    rows = [
        (
            res.configuration,
            f"{res.max_absolute_error:.2e}",
            "Passed (< 1e-4)" if res.passed else "Failed",
        )
        for res in results
    ]
    console.print(create_table("PyTorch vs ONNX Numerical Parity Check", columns, rows))
    pause()


def render_latency_summary_table(summary: LatencyBenchmarkSummary) -> None:
    """Render CPU runtime speedup comparison table."""
    columns = [
        ("Runtime Engine", STYLE_PRIMARY, "left"),
        ("Average Latency (ms)", STYLE_WARNING, "right"),
        ("Speedup vs PyTorch", STYLE_SUCCESS, "right"),
    ]
    rows = [
        ("PyTorch Eager (CPU)", f"{summary.pt_ms:.2f} ms", "1.00x"),
        (
            "ONNX Runtime (Base)",
            f"{summary.base_onnx_ms:.2f} ms",
            f"{calculate_speedup(summary.pt_ms, summary.base_onnx_ms):.2f}x",
        ),
        (
            "ONNX Runtime (Optimized Fused)",
            f"{summary.opt_onnx_ms:.2f} ms",
            f"{calculate_speedup(summary.pt_ms, summary.opt_onnx_ms):.2f}x",
        ),
    ]
    console.print(create_table(f"CPU Inference Latency ({BENCHMARK_ITERATIONS} iterations)", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute BERT ONNX export, graph optimization, and latency benchmark."""
    os.makedirs(ONNX_DIR, exist_ok=True)

    render_banner(
        title="BERT Model Optimization with ONNX Runtime (CPU)",
        subtitle="Chapter 5: Domain-Specific Small Language Models",
        metadata={
            "Base Model": MODEL_ID,
            "Execution Provider": "CPUExecutionProvider (Fused Kernels)",
            "Dataset": f"{DATASET_ID}/{DATASET_SUBSET}",
        },
        icon="🚀",
    )

    # Step 1: Loading Tokenizer & Dataset
    render_step(1, "Loading Tokenizer & GLUE MRPC Dataset", icon="📋")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    with status_spinner(f"Loading '{DATASET_ID}/{DATASET_SUBSET}' dataset..."):
        dataset = load_dataset(DATASET_ID, DATASET_SUBSET)
        tokenized_dataset = dataset.map(
            lambda ex: tokenizer(
                ex["sentence1"],
                ex["sentence2"],
                truncation=True,
                padding="max_length",
                max_length=128,
            ),
            batched=True,
        )

    # Step 2: Training PyTorch BERT Model
    render_step(2, "Fine-Tuning BERT on MRPC Demo Subset", icon="🏋️")
    pt_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt_model.to(device)
    render_device_info(device, model=pt_model)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        disable_tqdm=False,
        report_to="none",
    )
    trainer = Trainer(
        model=pt_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"].select(range(100)),
        eval_dataset=tokenized_dataset["validation"].select(range(50)),
        processing_class=tokenizer,
        compute_metrics=compute_evaluation_metrics,
    )
    console.print("[bold green]Running fine-tuning loop...[/bold green]")
    trainer.train()

    # Step 3: Exporting to ONNX Format
    render_step(3, "Exporting Static Computation DAG to ONNX", icon="⚙️")
    dummy_input = tokenizer(
        "Sample text for tracing ONNX graph.", return_tensors="pt", padding="max_length", max_length=128
    )
    dummy_input = {k: v.to(pt_model.device) for k, v in dummy_input.items()}
    pt_model.eval()
    with status_spinner("Tracing PyTorch computation graph..."):
        torch.onnx.export(
            pt_model,
            (dummy_input["input_ids"], dummy_input["attention_mask"], dummy_input["token_type_ids"]),
            BASE_ONNX_PATH,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "token_type_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
            opset_version=14,
        )
    render_card("ONNX Export", f"Model exported to:\n[text.highlight]{BASE_ONNX_PATH}[/text.highlight]", icon="💾")

    # Step 4: Applying Graph-Level Optimizations
    render_step(4, "Applying Graph-Level Operator Fusion", icon="⚡")
    with status_spinner("Fusing MultiHeadAttention and LayerNorm kernels..."):
        opt_model = optimize_model(
            BASE_ONNX_PATH,
            model_type="bert",
            num_heads=12,
            hidden_size=768,
            opt_level=99,
        )
        opt_model.save_model_to_file(OPT_ONNX_PATH)
    render_card(
        "Graph Fusion", f"Optimized ONNX graph saved to:\n[text.highlight]{OPT_ONNX_PATH}[/text.highlight]", icon="✔"
    )

    # Step 5: Numerical Parity Verification
    render_step(5, "Verifying Numerical Parity against PyTorch Eager", icon="🔍")
    inputs = tokenizer(BENCHMARK_PROMPT, return_tensors="pt", padding="max_length", max_length=128).to(pt_model.device)
    with torch.no_grad():
        pt_logits = pt_model(**inputs).logits.cpu().numpy()

    ort_inputs = {
        "input_ids": inputs["input_ids"].cpu().numpy(),
        "attention_mask": inputs["attention_mask"].cpu().numpy(),
        "token_type_ids": inputs["token_type_ids"].cpu().numpy(),
    }

    base_sess = ort.InferenceSession(BASE_ONNX_PATH, providers=["CPUExecutionProvider"])
    opt_sess = ort.InferenceSession(OPT_ONNX_PATH, providers=["CPUExecutionProvider"])

    base_logits = base_sess.run(None, ort_inputs)[0]
    opt_logits = opt_sess.run(None, ort_inputs)[0]

    diff_base = float(np.max(np.abs(pt_logits - base_logits)))
    diff_opt = float(np.max(np.abs(pt_logits - opt_logits)))

    parity_results = (
        ParityCheckResult("Base ONNX vs PyTorch", diff_base, diff_base < 1e-4),
        ParityCheckResult("Optimized ONNX vs PyTorch", diff_opt, diff_opt < 1e-4),
    )
    render_parity_table(parity_results)

    # Step 6: Inference Latency Benchmarking
    render_step(6, "Benchmarking Latency on CPUExecutionProvider", icon="📊")
    with status_spinner("Benchmarking PyTorch Eager CPU latency..."):
        pt_lat = measure_pure_latency(lambda: pt_model(**inputs))
    with status_spinner("Benchmarking Base ONNX CPU latency..."):
        base_lat = measure_pure_latency(lambda: base_sess.run(None, ort_inputs))
    with status_spinner("Benchmarking Optimized ONNX CPU latency..."):
        opt_lat = measure_pure_latency(lambda: opt_sess.run(None, ort_inputs))

    summary = LatencyBenchmarkSummary(pt_ms=pt_lat, base_onnx_ms=base_lat, opt_onnx_ms=opt_lat)
    render_latency_summary_table(summary)

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Why ONNX Runtime is Faster on CPU",
                "PyTorch evaluates operations step-by-step with Python interpreter overhead. ONNX compiles the entire computational graph into optimized C++ execution kernels.",
            ),
            (
                "Graph Fusion (opt_level=99)",
                "Identifies MultiHeadAttention and LayerNorm patterns and replaces multiple fine-grained nodes with a single fused kernel, reducing memory traffic.",
            ),
            (
                "Numerical Parity Check",
                "Always verify logits parity (Delta < 1e-4) after optimization to ensure that constant folding and operator fusions did not corrupt floating-point semantics.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
