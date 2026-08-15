"""Quantization of the GPT-2 Small Model with LLM.int8().

Companion script for Chapter 6 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Demonstrates LLM.int8() mixed-precision decomposition of GPT-2 using bitsandbytes.
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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    render_device_info,
    render_card,
    render_step,
    render_takeaways,
    status_spinner,
)


# ---------------------------------------------------------------------------
# Immutable Domain Records & Constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MemoryComparison:
    """Memory consumption footprint metrics."""

    fp32_bytes: int
    int8_bytes: int
    saved_ratio: float


@dataclass(frozen=True)
class PrecisionPerplexityResult:
    """Perplexity metrics across FP32 and LLM.int8()."""

    fp32_ppl: float
    int8_ppl: float


MODEL_ID = "openai-community/gpt2"
GENERATION_PROMPT = "My favourite school subject is"
MAX_GEN_LENGTH = 100
TOP_K = 30
HIST_BINS = 150
HIST_RANGE = (-2, 2)
PLOT_DPI = 300


# ---------------------------------------------------------------------------
# Pure Functions & Helpers
# ---------------------------------------------------------------------------
def extract_model_weight_array(model: Any) -> np.ndarray:
    """Pure array extraction: flatten all parameters into a 1-D numpy array."""
    return np.concatenate([param.data.clone().cpu().numpy().flatten() for param in model.parameters()])


def compute_memory_reduction(fp32_bytes: int, int8_bytes: int) -> MemoryComparison:
    """Pure calculation of memory savings."""
    ratio = (1.0 - int8_bytes / fp32_bytes) * 100.0 if fp32_bytes > 0 else 0.0
    return MemoryComparison(fp32_bytes=fp32_bytes, int8_bytes=int8_bytes, saved_ratio=ratio)


def calculate_model_perplexity(model: Any, tokenizer: Any, text: str, device: torch.device) -> float:
    """Pure calculation of perplexity score."""
    encodings = tokenizer(text, return_tensors="pt").to(device)
    input_ids = encodings.input_ids
    target_ids = input_ids.clone()
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
    return float(torch.exp(outputs.loss).item())


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_memory_table(mem: MemoryComparison) -> None:
    """Render memory footprint comparison table."""
    columns = [
        ("Model Variant", STYLE_PRIMARY, "left"),
        ("Memory (Bytes)", STYLE_TEXT, "right"),
        ("Memory (MB)", STYLE_WARNING, "right"),
        ("VRAM Reduction", STYLE_SUCCESS, "right"),
    ]
    rows = [
        ("GPT-2 (FP32)", f"{mem.fp32_bytes:,}", f"{mem.fp32_bytes / 1024 / 1024:.2f} MB", "Baseline (0%)"),
        (
            "GPT-2 (LLM.int8)",
            f"{mem.int8_bytes:,}",
            f"{mem.int8_bytes / 1024 / 1024:.2f} MB",
            f"-{mem.saved_ratio:.1f}%",
        ),
    ]
    console.print(create_table("Model Memory Footprint Comparison", columns, rows))
    pause()


def render_perplexity_table(res: PrecisionPerplexityResult) -> None:
    """Render perplexity quality comparison table."""
    columns = [
        ("Model Precision", STYLE_PRIMARY, "left"),
        ("Perplexity Score", STYLE_SUCCESS, "right"),
        ("Status / Notes", STYLE_TEXT, "left"),
    ]
    rows = [
        ("FP32 Full Precision", f"{res.fp32_ppl:.2f}", "Baseline model perplexity"),
        ("LLM.int8() Mixed Precision", f"{res.int8_ppl:.2f}", "Preserves perplexity with ~50% VRAM savings"),
    ]
    console.print(create_table("Perplexity Quality Metrics", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute LLM.int8() mixed-precision quantization demo."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    render_banner(
        title="Quantization of GPT-2 Small with LLM.int8()",
        subtitle="Chapter 6: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_ID,
            "Library": "bitsandbytes",
            "Hardware": str(device).upper(),
        },
        icon="🚀",
    )

    render_device_info(device)

    # Step 1: Loading FP32 and LLM.int8() Models
    render_step(1, "Loading Full Precision and 8-Bit Decomposed Models", icon="📋")
    with status_spinner("Loading baseline FP32 model..."):
        model_fp32 = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    fp32_bytes = model_fp32.get_memory_footprint()

    with status_spinner("Loading LLM.int8() model via bitsandbytes..."):
        model_int8 = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", load_in_8bit=True)
    int8_bytes = model_int8.get_memory_footprint()

    mem_stats = compute_memory_reduction(fp32_bytes, int8_bytes)
    render_memory_table(mem_stats)

    # Step 2: Extracting & Visualizing Weight Distributions
    render_step(2, "Extracting Weight Matrices & Distribution Analysis", icon="📊")
    with status_spinner("Aggregating tensor parameters..."):
        weights_fp32 = extract_model_weight_array(model_fp32)
        weights_int8 = extract_model_weight_array(model_int8)

    render_card(
        title="Weight Matrix Summary",
        content=(
            f"[text.muted]FP32 Weights Mean:[/text.muted] [brand.secondary]{weights_fp32.mean():.4f}[/brand.secondary] "
            f"([text.dim]Std: {weights_fp32.std():.4f}[/text.dim])\n"
            f"[text.muted]INT8 Weights Mean:[/text.muted] [brand.secondary]{weights_int8.mean():.4f}[/brand.secondary] "
            f"([text.dim]Std: {weights_int8.std():.4f}[/text.dim])"
        ),
        icon="✔",
    )

    # Step 3: Text Generation Comparison
    render_step(3, "Generating Continuations: FP32 vs LLM.int8()", icon="✨")
    input_ids = tokenizer.encode(GENERATION_PROMPT, return_tensors="pt").to(device)

    with status_spinner("Generating text with FP32 baseline..."):
        out_fp = model_fp32.generate(
            inputs=input_ids,
            max_length=MAX_GEN_LENGTH,
            do_sample=True,
            top_k=TOP_K,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=input_ids.new_ones(input_ids.shape),
        )
        text_fp32 = tokenizer.decode(out_fp[0], skip_special_tokens=True)

    with status_spinner("Generating text with LLM.int8() quantized model..."):
        out_int8 = model_int8.generate(
            inputs=input_ids,
            max_length=MAX_GEN_LENGTH,
            do_sample=True,
            top_k=TOP_K,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=input_ids.new_ones(input_ids.shape),
        )
        text_int8 = tokenizer.decode(out_int8[0], skip_special_tokens=True)

    render_card("Original FP32 Generation", text_fp32, icon="📄")
    render_card("LLM.int8() Generation", text_int8, icon="⚡")

    # Step 4: Perplexity Evaluation
    render_step(4, "Perplexity Validation", icon="📈")
    with status_spinner("Evaluating sequence perplexity..."):
        ppl_fp = calculate_model_perplexity(model_fp32, tokenizer, text_fp32, device)
        ppl_int8 = calculate_model_perplexity(model_int8, tokenizer, text_int8, device)

    ppl_res = PrecisionPerplexityResult(fp32_ppl=ppl_fp, int8_ppl=ppl_int8)
    render_perplexity_table(ppl_res)

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Why LLM.int8() prevents quality degradation",
                "Naive INT8 fails on large models due to activation outliers. LLM.int8() dynamically isolates outlier feature dimensions into high-precision FP16 matrix multiplication while quantizing everything else.",
            ),
            (
                "50% VRAM Reduction",
                "Enables serving 7B-13B models on single consumer GPUs (e.g., RTX 3060/4060 with 8-12GB VRAM) without quality loss.",
            ),
            (
                "Inference Latency Trade-off",
                "While LLM.int8() dramatically cuts GPU memory, matrix decomposition requires mixed-precision kernel coordination, so inference throughput may be slightly lower than pure FP16 on high-end GPUs.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
