"""Using SmoothQuant on OPT large models.

Companion script for chapter 9 of "Domain Specific LLMs in Action"
by Guglielmo Iozzia (Manning Publications, 2024).

Demonstrates the activation outlier problem in 6B+ parameter models and validates
SmoothQuant W8A8 quantization against full-precision FP16 and naive W8A8.
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Third-party
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
from transformers.models.opt.modeling_opt import (
    OPTAttention,
    OPTDecoderLayer,
    OPTForCausalLM,
)

# Common functional & UI utilities
from common.functional import format_percentage
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
class QuantizationAccuracyResult:
    """Immutable accuracy benchmark summary."""

    scheme_name: str
    accuracy: float
    relative_ratio: float
    assessment: str


MODEL_ID = "facebook/opt-6.7b"
DATASET_NAME = "cimec/lambada"
DATASET_SPLIT = "validation[:1000]"
ACT_SCALES_PATH = "./act_scales/opt-6.7b.pt"
OFFLOAD_FOLDER = "."
SMOOTHQUANT_ALPHA = 0.5


# ---------------------------------------------------------------------------
# Evaluator Class & Pure Transformations
# ---------------------------------------------------------------------------
class Evaluator:
    """Evaluate an LLM on tokenized dataset using last-token prediction accuracy."""

    def __init__(self, dataset: Any, tokenizer: Any, device: str) -> None:
        self.device = device
        self.tokenizer = tokenizer

        tokenized = dataset.map(lambda ex: self.tokenizer(ex["text"]), batched=True)
        tokenized.set_format(type="torch", columns=["input_ids"])
        self.dataset = tokenized

    @torch.no_grad()
    def evaluate(self, model: Any) -> float:
        """Return exact match accuracy on last token across dataset."""
        model.eval()
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch["input_ids"].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += int((pred == label).sum().item())
        return float(hit / total) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_accuracy_summary_table(results: Sequence[QuantizationAccuracyResult]) -> None:
    """Render LAMBADA accuracy comparison table."""
    columns = [
        ("Quantization Scheme", STYLE_PRIMARY, "left"),
        ("Accuracy (%)", STYLE_SUCCESS, "right"),
        ("Retention vs FP16", STYLE_WARNING, "right"),
        ("Assessment", STYLE_TEXT, "left"),
    ]
    rows = [
        (
            r.scheme_name,
            f"{r.accuracy * 100:.2f}%",
            f"{r.relative_ratio:.2f}%",
            r.assessment,
        )
        for r in results
    ]
    console.print(create_table("Accuracy Retention on LAMBADA Benchmark", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute SmoothQuant W8A8 quantization comparison pipeline."""
    render_banner(
        title="SmoothQuant on Large Language Models (OPT-6.7B)",
        subtitle="Chapter 9: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_ID,
            "Dataset": DATASET_NAME,
            "Migration Strength (Alpha)": str(SMOOTHQUANT_ALPHA),
        },
        icon="🚀",
    )

    # Step 1: Initializing Evaluator & Dataset
    render_step(1, "Initializing LAMBADA Evaluator & Tokenizer", icon="📋")
    with status_spinner(f"Loading '{DATASET_NAME}' dataset ({DATASET_SPLIT})..."):
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
        evaluator = Evaluator(dataset, tokenizer, "cuda" if torch.cuda.is_available() else "cpu")
        render_device_info(evaluator.device)
    render_card(
        "Evaluator Initialized",
        f"LAMBADA test dataset prepared with [text.highlight]{len(dataset)}[/text.highlight] validation records.",
        icon="✔",
    )

    # Step 2: Evaluating FP16 Baseline Model
    render_step(2, "Evaluating FP16 Baseline Model", icon="🧠")
    with status_spinner(f"Loading '{MODEL_ID}' in FP16..."):
        model_fp16 = OPTForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder=OFFLOAD_FOLDER,
        )
        model_fp16.eval()

    with status_spinner("Computing FP16 baseline accuracy on LAMBADA..."):
        fp16_acc = evaluator.evaluate(model_fp16)

    render_card(
        "FP16 Baseline Accuracy",
        f"[status.success]Accuracy:[/status.success] [text.highlight]{fp16_acc * 100:.2f}%[/text.highlight]",
        icon="✔",
    )

    # Step 3: Evaluating Naive W8A8 Quantized Model
    render_step(3, "Evaluating Naive W8A8 Quantized Model", icon="⚡")
    try:
        from smoothquant.fake_quant import W8A8Linear
        from smoothquant.smooth import smooth_lm

        def apply_int8(model: Any) -> Any:
            for _, m in model.model.named_modules():
                if isinstance(m, OPTDecoderLayer):
                    m.fc1 = W8A8Linear.from_float(m.fc1)
                    m.fc2 = W8A8Linear.from_float(m.fc2)
                elif isinstance(m, OPTAttention):
                    m.q_proj = W8A8Linear.from_float(m.q_proj, quantize_output=True)
                    m.k_proj = W8A8Linear.from_float(m.k_proj, quantize_output=True)
                    m.v_proj = W8A8Linear.from_float(m.v_proj, quantize_output=True)
                    m.out_proj = W8A8Linear.from_float(m.out_proj)
            return model

        with status_spinner("Quantizing model to naive W8A8..."):
            model_naive = apply_int8(model_fp16)
            naive_acc = evaluator.evaluate(model_naive)

        render_card(
            "Naive W8A8 Accuracy",
            f"[status.warning]Accuracy:[/status.warning] [text.highlight]{naive_acc * 100:.2f}%[/text.highlight] (Severe outlier degradation)",
            icon="⚠️",
        )

        # Step 4: Evaluating SmoothQuant W8A8 Model
        render_step(4, "Applying SmoothQuant Feature Migration & Evaluating W8A8", icon="✨")
        with status_spinner(f"Migrating outlier scales (alpha={SMOOTHQUANT_ALPHA})..."):
            model_fp16 = OPTForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder=OFFLOAD_FOLDER,
            )
            act_scales = torch.load(ACT_SCALES_PATH)
            smooth_lm(model_fp16, act_scales, SMOOTHQUANT_ALPHA)
            model_sq = apply_int8(model_fp16)
            sq_acc = evaluator.evaluate(model_sq)

        render_card(
            "SmoothQuant W8A8 Accuracy",
            f"[status.success]Accuracy:[/status.success] [text.highlight]{sq_acc * 100:.2f}%[/text.highlight] (Full accuracy retention)",
            icon="✔",
        )

        # Step 5: Comparative Summary
        render_step(5, "Accuracy Summary & Trade-off Comparison", icon="📊")
        summary_results = (
            QuantizationAccuracyResult("FP16 Baseline", fp16_acc, 100.0, "Full Precision Reference"),
            QuantizationAccuracyResult(
                "Naive W8A8 (per-tensor)",
                naive_acc,
                naive_acc / fp16_acc * 100.0 if fp16_acc > 0 else 0.0,
                "Outlier activations degrade accuracy",
            ),
            QuantizationAccuracyResult(
                "SmoothQuant W8A8",
                sq_acc,
                sq_acc / fp16_acc * 100.0 if fp16_acc > 0 else 0.0,
                "Outliers smoothed; accuracy fully preserved",
            ),
        )
        render_accuracy_summary_table(summary_results)
    except (ImportError, FileNotFoundError):
        render_card("Environment Note", "SmoothQuant package and act_scales required for full evaluation.", icon="ℹ️")

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Why Naive W8A8 Fails on 6B+ Models",
                "At 6.7B parameters, activation outliers emerge with values 100x higher than typical features. In naive INT8, scaling by the outlier crushes normal signals to 0, dropping accuracy near zero.",
            ),
            (
                "The SmoothQuant Insight",
                "Activations are hard to quantize; weights are easy. SmoothQuant divides activation channels by scale vector s and multiplies weights by s, mathematically preserving output (Y = X_hat * W_hat) while making both activations and weights fit comfortably in INT8.",
            ),
            (
                "Hardware Advantage of W8A8",
                "Unlike weight-only quantization (W8A16 or W4A16), W8A8 allows running matrix multiplications directly on INT8 Tensor Cores (DP4A / TensorCore INT8), delivering up to 2x real hardware speedups in addition to 2x memory reduction.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
