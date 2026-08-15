"""Quantization of the GPT-2 Small Model (absmax).

Companion script for Chapter 6 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Introduces absmax quantization of a decoder-only language model (GPT-2 Small).
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import sys
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Third-party
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Common functional & UI utilities
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
class TensorStats:
    """Statistical summary of a flattened tensor array."""

    min_val: float
    max_val: float
    mean_val: float
    std_val: float


@dataclass(frozen=True)
class QuantizationEvaluation:
    """Evaluation summary for original vs quantized model."""

    fp32_perplexity: float
    int8_perplexity: float
    fp32_text: str
    int8_text: str


MODEL_ID = "openai-community/gpt2"
DEVICE = "cpu"
GENERATION_PROMPT = "My favourite school subject is"
MAX_GEN_LENGTH = 100
TOP_K = 30
HIST_BINS = 150
HIST_RANGE = (-2, 2)
PLOT_DPI = 300


# ---------------------------------------------------------------------------
# Pure Functions for Quantization & Statistics
# ---------------------------------------------------------------------------
def compute_absmax_scale(tensor: torch.Tensor) -> float:
    """Pure calculation of symmetric absmax scale factor."""
    max_val = float(torch.max(torch.abs(tensor)))
    return 127.0 / max_val if max_val > 0.0 else 1.0


def quantize_absmax_pure(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure functional transformation: maps float tensor to int8 and dequantized float."""
    scale = compute_absmax_scale(tensor)
    quant = (scale * tensor).round().clamp(-127, 127).to(torch.int8)
    dequant = quant.to(torch.float32) / scale
    return quant, dequant


def compute_tensor_stats(array: np.ndarray) -> TensorStats:
    """Pure calculation of statistical distribution metrics."""
    return TensorStats(
        min_val=float(array.min()),
        max_val=float(array.max()),
        mean_val=float(array.mean()),
        std_val=float(array.std()),
    )


def calculate_pure_perplexity(model: Any, tokenizer: Any, text: str, device: str) -> float:
    """Pure perplexity score calculation."""
    encodings = tokenizer(text, return_tensors="pt").to(device)
    input_ids = encodings.input_ids
    target_ids = input_ids.clone()
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
    return float(torch.exp(outputs.loss).item())


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_stats_table(fp_stats: TensorStats, int8_stats: TensorStats) -> None:
    """Render weight distribution statistics table."""
    columns = [
        ("Variant", STYLE_PRIMARY, "left"),
        ("Min", STYLE_TEXT, "right"),
        ("Max", STYLE_TEXT, "right"),
        ("Mean", STYLE_SUCCESS, "right"),
        ("Std Dev", STYLE_WARNING, "right"),
    ]
    rows = [
        (
            "Original FP32",
            f"{fp_stats.min_val:.4f}",
            f"{fp_stats.max_val:.4f}",
            f"{fp_stats.mean_val:.4f}",
            f"{fp_stats.std_val:.4f}",
        ),
        (
            "Absmax Dequantized",
            f"{int8_stats.min_val:.4f}",
            f"{int8_stats.max_val:.4f}",
            f"{int8_stats.mean_val:.4f}",
            f"{int8_stats.std_val:.4f}",
        ),
    ]
    console.print(create_table("Weight Distribution Statistics", columns, rows))
    pause()


def render_evaluation_table(eval_res: QuantizationEvaluation) -> None:
    """Render perplexity and quality comparison table."""
    columns = [
        ("Model Variant", STYLE_PRIMARY, "left"),
        ("Perplexity Score", STYLE_SUCCESS, "right"),
        ("Assessment", STYLE_TEXT, "left"),
    ]
    rows = [
        ("Original FP32", f"{eval_res.fp32_perplexity:.2f}", "Baseline precision reference"),
        ("Absmax INT8", f"{eval_res.int8_perplexity:.2f}", "Slight quantization loss; meaning retained"),
    ]
    console.print(create_table("Perplexity & Output Quality Comparison", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute absmax quantization pipeline on GPT-2."""
    render_banner(
        title="Quantization of GPT-2 Small (Absmax INT8)",
        subtitle="Chapter 6: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_ID,
            "Hardware": DEVICE.upper(),
            "Prompt": f'"{GENERATION_PROMPT}"',
        },
        icon="🚀",
    )

    # Step 1: Loading Base Model & Tokenizer
    render_step(1, "Loading Base Model & Memory Footprint Analysis", icon="📋")
    with status_spinner(f"Loading '{MODEL_ID}' on {DEVICE}..."):
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    fp_memory = model.get_memory_footprint()
    render_card(
        title="Model Footprint",
        content=(
            f"[text.muted]Memory Size:[/text.muted] [text.highlight]{fp_memory:,} bytes[/text.highlight] "
            f"([brand.secondary]{fp_memory / 1024 / 1024:.2f} MB[/brand.secondary])\n"
            f"[text.muted]Data Type:[/text.muted] [text.main]Float32 (32 bits per parameter)[/text.main]"
        ),
        icon="✔",
    )

    # Step 2: Applying Absmax Quantization
    render_step(2, "Applying Symmetric Absmax INT8 Quantization", icon="⚙️")
    original_weights = [param.data.clone() for param in model.parameters()]

    model_abs = deepcopy(model)
    weights_abs_list = []
    with status_spinner("Scaling and quantizing all model weight matrices..."):
        for param in model_abs.parameters():
            _, dequantized = quantize_absmax_pure(param.data)
            param.data = dequantized
            weights_abs_list.append(dequantized)

    render_card(
        "Quantization Complete", "Absmax INT8 scaling applied to all transformer projection matrices.", icon="✔"
    )

    # Step 3: Weight Distribution Analysis
    render_step(3, "Statistical Weight Distribution Analysis", icon="📊")
    weights_flat = np.concatenate([t.cpu().numpy().flatten() for t in original_weights])
    weights_abs_flat = np.concatenate([t.cpu().numpy().flatten() for t in weights_abs_list])

    fp_stats = compute_tensor_stats(weights_flat)
    int8_stats = compute_tensor_stats(weights_abs_flat)
    render_stats_table(fp_stats, int8_stats)

    # Step 4: Text Generation Comparison
    render_step(4, "Generating Continuations: FP32 vs Absmax INT8", icon="✨")
    input_ids = tokenizer.encode(GENERATION_PROMPT, return_tensors="pt").to(DEVICE)

    with status_spinner("Generating text with original FP32 model..."):
        out_fp = model.generate(
            inputs=input_ids,
            max_length=MAX_GEN_LENGTH,
            do_sample=True,
            top_k=TOP_K,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=input_ids.new_ones(input_ids.shape),
        )
        original_text = tokenizer.decode(out_fp[0], skip_special_tokens=True)

    with status_spinner("Generating text with Absmax INT8 model..."):
        out_int8 = model_abs.generate(
            inputs=input_ids,
            max_length=MAX_GEN_LENGTH,
            do_sample=True,
            top_k=TOP_K,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=input_ids.new_ones(input_ids.shape),
        )
        absmax_text = tokenizer.decode(out_int8[0], skip_special_tokens=True)

    render_card("Original Model Output (FP32)", original_text, icon="📄")
    render_card("Absmax Quantized Output (INT8)", absmax_text, icon="⚡")

    # Step 5: Perplexity Evaluation
    render_step(5, "Perplexity & Language Naturalness Evaluation", icon="📈")
    with status_spinner("Evaluating cross-entropy loss and perplexity..."):
        ppl_fp = calculate_pure_perplexity(model, tokenizer, original_text, DEVICE)
        ppl_int8 = calculate_pure_perplexity(model_abs, tokenizer, absmax_text, DEVICE)

    eval_result = QuantizationEvaluation(
        fp32_perplexity=ppl_fp,
        int8_perplexity=ppl_int8,
        fp32_text=original_text,
        int8_text=absmax_text,
    )
    render_evaluation_table(eval_result)

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Symmetric vs Asymmetric Quantization",
                "Absmax is symmetric (centered at 0, no zero-point offset needed), making matrix arithmetic faster on hardware.",
            ),
            (
                "Why Perplexity Increases Slightly",
                "Discarding 24 bits of float precision introduces rounding quantization noise in quantized space, which is magnified if outliers stretch the scale factor.",
            ),
            (
                "Limitations of Naive Absmax",
                "Because one extreme outlier forces all other values to be mapped to a few integer bins, modern approaches use block-wise quantization (LLM.int8(), GPTQ) or activation smoothing (SmoothQuant).",
            ),
        ),
    )


if __name__ == "__main__":
    main()
