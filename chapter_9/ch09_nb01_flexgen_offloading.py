"""Using FlexGen to Offload OPT Model Weights to RAM and Disk.

Companion script for Chapter 9 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Performs batch inference with Meta AI's OPT 1.3B model using the FlexGen
engine to offload model weights from VRAM to RAM and/or disk.
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer

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
class FlexGenPolicySpec:
    """Immutable configuration for multi-tier memory offloading."""

    gpu_weight_percent: int
    cpu_weight_percent: int
    gpu_cache_percent: int
    cpu_cache_percent: int
    gpu_compute_percent: int
    cpu_compute_percent: int
    weight_num_bits: int


MODEL_ID: str = "facebook/opt-1.3b"
OFFLOAD_DIR: str = "./flexgen_offload"
OPT_WEIGHTS_PATH: str = "~/opt_weights"

TOKENIZER_PADDING_SIDE: str = "left"
INPUT_MAX_LENGTH: int = 128
MAX_NEW_TOKENS: int = 32
TEMPERATURE: float = 0.7

DEFAULT_POLICY_SPEC = FlexGenPolicySpec(
    gpu_weight_percent=70,
    cpu_weight_percent=30,
    gpu_cache_percent=70,
    cpu_cache_percent=30,
    gpu_compute_percent=100,
    cpu_compute_percent=0,
    weight_num_bits=4,
)

PROMPTS: tuple[str, ...] = (
    (
        "Question: Where were the 2004 Olympics held?\n"
        "Answer: Athens, Greece\n"
        "Question: What is the longest river on the earth?\n"
        "Answer:"
    ),
    (
        "Extract the airport codes from this text.\n"
        'Text: "I want a flight from New York to San Francisco."\n'
        "Airport codes: JFK, SFO.\n"
        'Text: "I want you to book a flight from Phoenix to Las Vegas."\n'
        "Airport codes:"
    ),
)


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_policy_table(spec: FlexGenPolicySpec) -> None:
    """Render FlexGen offload resource allocation table."""
    columns = [
        ("Resource / Tier", STYLE_PRIMARY, "left"),
        ("GPU VRAM Allocation", STYLE_SUCCESS, "right"),
        ("Host RAM Allocation", STYLE_WARNING, "right"),
        ("Compression Mode", STYLE_SECONDARY, "right"),
    ]
    rows = [
        (
            "Model Parameter Weights",
            f"{spec.gpu_weight_percent}%",
            f"{spec.cpu_weight_percent}%",
            f"{spec.weight_num_bits}-bit INT",
        ),
        ("Attention KV Cache", f"{spec.gpu_cache_percent}%", f"{spec.cpu_cache_percent}%", "FP16 Uncompressed"),
        ("Computation Execution", f"{spec.gpu_compute_percent}%", f"{spec.cpu_compute_percent}%", "Async I/O Overlap"),
    ]
    console.print(create_table("FlexGen Memory Offloading & Compression Policy", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute FlexGen memory offloading pipeline."""
    render_banner(
        title="High-Throughput Offloading Inference with FlexGen",
        subtitle="Chapter 9: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_ID,
            "Weight Compression": f"{DEFAULT_POLICY_SPEC.weight_num_bits}-bit INT",
            "Batch Size": str(len(PROMPTS)),
        },
        icon="🚀",
    )

    # Step 1: Initializing Tokenizer & Offloading Policy
    render_step(1, "Configuring Tokenizer & Offload Distribution Policy", icon="📋")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side=TOKENIZER_PADDING_SIDE)
    tokenizer.add_bos_token = False
    stop_token_id: int = tokenizer("\n").input_ids[0]
    render_device_info("cuda" if torch.cuda.is_available() else "cpu")
    render_policy_table(DEFAULT_POLICY_SPEC)

    # Step 2: FlexGen Engine Loading
    render_step(2, "Initializing Multi-Tier FlexGen Execution Engine", icon="🧠")
    try:
        from flexllmgen.flex_opt import CompressionConfig, ExecutionEnv, OptLM, Policy

        env = ExecutionEnv.create(OFFLOAD_DIR)
        policy = Policy(
            len(PROMPTS),
            1,
            DEFAULT_POLICY_SPEC.gpu_weight_percent,
            DEFAULT_POLICY_SPEC.cpu_weight_percent,
            DEFAULT_POLICY_SPEC.gpu_cache_percent,
            DEFAULT_POLICY_SPEC.cpu_cache_percent,
            DEFAULT_POLICY_SPEC.gpu_compute_percent,
            DEFAULT_POLICY_SPEC.cpu_compute_percent,
            overlap=True,
            sep_layer=True,
            pin_weight=True,
            cpu_cache_compute=True,
            attn_sparsity=1.0,
            compress_weight=True,
            comp_weight_config=CompressionConfig(
                num_bits=DEFAULT_POLICY_SPEC.weight_num_bits,
                group_size=64,
                group_dim=0,
                symmetric=False,
            ),
            compress_cache=False,
            comp_cache_config=CompressionConfig(
                num_bits=4,
                group_size=64,
                group_dim=2,
                symmetric=False,
            ),
        )
        with status_spinner(f"Loading '{MODEL_ID}' with FlexGen ExecutionEnv..."):
            model = OptLM(MODEL_ID, env, OPT_WEIGHTS_PATH, policy)
        render_card("Engine Ready", "Model loaded across GPU VRAM and Host RAM offloading tiers.", icon="✔")

        # Step 3: Running Batch Generation
        render_step(3, "Running Batched Inference with Overlapped Memory Transfer", icon="✨")
        inputs = tokenizer(list(PROMPTS), padding="max_length", max_length=INPUT_MAX_LENGTH)
        with status_spinner("Generating batch outputs..."):
            output_ids = model.generate(
                inputs.input_ids,
                do_sample=True,
                temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
                stop=stop_token_id,
            )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for i, (prompt, out) in enumerate(zip(PROMPTS, outputs), start=1):
            render_card(
                title=f"Batch Output #{i}",
                content=f"[text.muted]Prompt:[/text.muted]\n{prompt}\n\n[status.success]Completion:[/status.success]\n{out.strip()}",
                icon="📄",
            )

        env.close_copy_threads()
    except ImportError:
        render_card("Environment Status", "FlexGen package required in environment for native execution.", icon="ℹ️")

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Hierarchical Memory Offloading",
                "FlexGen orchestrates tensors across GPU VRAM <-> CPU RAM <-> NVMe Disk, enabling 30B+ models to run on single GPUs.",
            ),
            (
                "Compute & I/O Overlapping",
                "Asynchronously prefetches next-layer weights from RAM while the GPU computes the current layer's attention.",
            ),
            (
                "Batch Throughput Optimization",
                "Grouping tokens into large batches amortizes the memory transfer latency across multiple queries.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
