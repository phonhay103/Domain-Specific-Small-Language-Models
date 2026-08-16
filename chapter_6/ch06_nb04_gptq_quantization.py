"""4-bit Quantization of GPT-2 with Auto-GPTQ.

Companion script for Chapter 6 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Introduces 4-bit quantization of a decoder-only language model (GPT-2)
using the AutoGPTQ library with calibration datasets.
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import random
import sys
import time
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
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
from transformers import AutoTokenizer, TextGenerationPipeline

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
    render_device_info,
    render_step,
    render_takeaways,
    status_spinner,
)


# ---------------------------------------------------------------------------
# Immutable Domain Records & Constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GPTQConfigSummary:
    """Immutable parameter container for GPTQ quantization settings."""

    bits: int
    group_size: int
    desc_act: bool
    calibration_samples: int


MODEL_ID = "openai-community/gpt2"
QUANTIZED_MODEL_DIR = "gpt-2-4bit"
WIKITEXT_DATASET = "wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
QUANTIZE_BITS = 4
QUANTIZE_GROUP_SIZE = 128
DEFAULT_SEQ_LEN = 2048
NUM_CALIBRATION_SAMPLES = 128
CALIBRATION_SEED = 0
INFERENCE_PROMPT = "Auto-GPTQ is"
CUDA_DEVICE = "cuda:0"


# ---------------------------------------------------------------------------
# Pure Functions & Data Preparation
# ---------------------------------------------------------------------------
def prepare_calibration_slices(
    encoded_ids: torch.Tensor,
    nsamples: int,
    seqlen: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], ...]:
    """Pure slice selection: sample deterministic sequence windows from tokenized corpus."""
    random.seed(seed)
    total_tokens = encoded_ids.shape[1]
    slices = []
    for _ in range(nsamples):
        i = random.randint(0, total_tokens - seqlen - 1)
        j = i + seqlen
        inp = encoded_ids[:, i:j]
        slices.append({"input_ids": inp, "attention_mask": torch.ones_like(inp)})
    return tuple(slices)


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_gptq_config_table(config: GPTQConfigSummary) -> None:
    """Render GPTQ hyperparameter table."""
    columns = [
        ("Quantization Setting", STYLE_PRIMARY, "left"),
        ("Value", STYLE_SUCCESS, "right"),
    ]
    rows = [
        ("Target Weight Bitwidth", f"{config.bits}-bit INT"),
        ("Quantization Group Size", str(config.group_size)),
        ("Activation Order (desc_act)", str(config.desc_act)),
        ("Calibration Dataset Samples", str(config.calibration_samples)),
    ]
    console.print(create_table("AutoGPTQ Configuration Parameters", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute 4-bit AutoGPTQ quantization pipeline."""
    render_banner(
        title="4-Bit Quantization of GPT-2 with AutoGPTQ",
        subtitle="Chapter 6: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_ID,
            "Target Precision": f"{QUANTIZE_BITS}-bit",
            "Group Size": str(QUANTIZE_GROUP_SIZE),
        },
        icon="🚀",
    )

    # Step 1: Initializing Configuration
    render_step(1, "Initializing AutoGPTQ Quantization Specification", icon="📋")
    gptq_config = GPTQConfigSummary(
        bits=QUANTIZE_BITS,
        group_size=QUANTIZE_GROUP_SIZE,
        desc_act=False,
        calibration_samples=NUM_CALIBRATION_SAMPLES,
    )
    render_gptq_config_table(gptq_config)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    quantize_config = BaseQuantizeConfig(
        bits=QUANTIZE_BITS,
        group_size=QUANTIZE_GROUP_SIZE,
        desc_act=False,
    )

    with status_spinner(f"Loading '{MODEL_ID}' base weights..."):
        model = AutoGPTQForCausalLM.from_pretrained(MODEL_ID, quantize_config)
    model.seqlen = getattr(model.config, "n_positions", DEFAULT_SEQ_LEN)

    render_device_info("cuda" if torch.cuda.is_available() else "cpu", model=model)
    # Step 2: Preparing Calibration Dataset
    render_step(2, "Extracting WikiText-2 Calibration Slices", icon="⚙️")
    with status_spinner("Loading WikiText-2 calibration splits..."):
        traindata = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split="train")
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        traindataset = list(
            prepare_calibration_slices(trainenc.input_ids, NUM_CALIBRATION_SAMPLES, model.seqlen, CALIBRATION_SEED)
        )

    render_card(
        title="Calibration Data Prepared",
        content=f"[text.muted]Calibration Windows:[/text.muted] [text.highlight]{len(traindataset)}[/text.highlight] ([brand.secondary]{model.seqlen} tokens/window[/brand.secondary])",
        icon="✔",
    )

    # Step 3: Quantizing & Saving
    render_step(3, "Executing 4-Bit Hessian-Based AutoGPTQ Quantization", icon="🧠")
    with status_spinner("Computing optimal weight projections via second-order Hessian updates..."):
        model.quantize(traindataset, use_triton=False)
        model.save_quantized(QUANTIZED_MODEL_DIR, use_safetensors=True)

    render_card(
        "Quantized Model Saved",
        f"4-bit safetensors weights saved to:\n[text.highlight]{QUANTIZED_MODEL_DIR}[/text.highlight]",
        icon="💾",
    )

    # Step 4: Inference with 4-Bit Model
    render_step(4, "Evaluating 4-Bit Quantized Model Inference", icon="⚡")
    with status_spinner("Loading 4-bit model onto GPU and running generation..."):
        quantized_model = AutoGPTQForCausalLM.from_quantized(QUANTIZED_MODEL_DIR, device=CUDA_DEVICE, use_triton=False)
        output_ids = quantized_model.generate(**tokenizer(INFERENCE_PROMPT, return_tensors="pt").to(CUDA_DEVICE))
        direct_out = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    render_card("4-Bit Quantized Text Generation", direct_out, icon="✨")

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "GPTQ Second-Order Error Compensation",
                "GPTQ uses inverse Hessian information (H^-1) to update remaining unquantized weights whenever a weight column is rounded to 4-bit.",
            ),
            (
                "Group Size (e.g., 128)",
                "Group-wise scaling divides weight matrices into small blocks of 128 elements, each with its own scale factor to preserve dynamic range.",
            ),
            (
                "75% VRAM Reduction",
                "Shrinks 16-bit float models down to 4-bit weights, allowing 8B parameter models to run comfortably in <6GB VRAM.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
