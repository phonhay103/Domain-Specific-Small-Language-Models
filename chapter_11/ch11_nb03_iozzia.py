"""Small Language Model Conversion and Inference with MLC LLM.

Companion script for chapter 11 of "Domain Specific LLMs in Action"
by Guglielmo Iozzia (Manning Publications, 2024).

Demonstrates using MLC LLM (Machine Learning Compilation) with Apache TVM to compile
and serve RedPajama-INCITE-Instruct-3B-v1 natively across GPU backends.
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
class MLCDistributionSpec:
    """Immutable metadata for compiled MLC model artifact."""

    model_dir: str
    compiled_lib: str
    target_device: str
    quantization: str


MODEL_DIR = "./dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC"
MODEL_LIB = "./dist/libs/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-cuda.so"
CHAT_QUESTION = "What's the meaning of life?"

DEFAULT_SPEC = MLCDistributionSpec(
    model_dir=MODEL_DIR,
    compiled_lib=MODEL_LIB,
    target_device="CUDA / TVM JIT",
    quantization="q4f16_1 (4-bit Weights, FP16 Activations)",
)


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_mlc_spec_table(spec: MLCDistributionSpec) -> None:
    """Render compiled TVM binary deployment specifications."""
    columns = [
        ("Deployment Property", STYLE_PRIMARY, "left"),
        ("Value", STYLE_SUCCESS, "right"),
    ]
    rows = [
        ("Model Weights Path", spec.model_dir),
        ("Compiled Dynamic Library", spec.compiled_lib),
        ("Target Compilation Backend", spec.target_device),
        ("Quantization Mode", spec.quantization),
    ]
    console.print(create_table("MLC LLM Compiled Native Deployment", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute MLC LLM compilation and inference demonstration."""
    render_banner(
        title="Small Language Model Inference with MLC LLM",
        subtitle="Chapter 11: Domain-Specific Small Language Models",
        metadata={
            "Model Directory": MODEL_DIR,
            "Compiled Library": MODEL_LIB,
            "Compilation Engine": "Apache TVM / MLC",
        },
        icon="🚀",
    )

    # Step 1: Inspecting Architecture
    render_step(1, "Inspecting Compiled Native Runtime Specification", icon="📋")
    render_mlc_spec_table(DEFAULT_SPEC)

    # Step 2: Initializing MLCEngine
    render_step(2, "Initializing Native C++ MLCEngine", icon="🧠")
    try:
        from mlc_llm import MLCEngine

        with status_spinner("Loading compiled dynamic library into GPU registers..."):
            engine = MLCEngine(model=MODEL_DIR, model_lib=MODEL_LIB)
        render_card("Engine Ready", "MLCEngine successfully loaded with zero Python overhead.", icon="✔")

        # Step 3: Synchronous Chat Completion
        render_step(3, "Executing Synchronous Chat Completion", icon="💬")
        with status_spinner(f"Submitting query: '{CHAT_QUESTION}'..."):
            full_content = ""
            for response in engine.chat.completions.create(
                messages=[{"role": "user", "content": CHAT_QUESTION}],
                model=MODEL_DIR,
                stream=False,
            ):
                if response.choices:
                    full_content += response.choices[0].message.content or ""

        render_card(
            title="Synchronous Response",
            content=f"[text.muted]Question:[/text.muted] {CHAT_QUESTION}\n\n[status.success]Response:[/status.success]\n{full_content.strip()}",
            icon="📄",
        )

        # Step 4: Streaming Chat Completion
        render_step(4, "Executing Low-Latency Streaming Token Delivery", icon="⚡")
        render_card(
            "Streaming Query",
            f'[text.muted]Streaming Tokens for:[/text.muted] [brand.secondary]"{CHAT_QUESTION}"[/brand.secondary]',
            icon="✨",
        )
        streamed_text = ""
        for response in engine.chat.completions.create(
            messages=[{"role": "user", "content": CHAT_QUESTION}],
            model=MODEL_DIR,
            stream=True,
        ):
            for choice in response.choices:
                chunk = choice.delta.content or ""
                streamed_text += chunk
                console.print(f"[brand.primary]{chunk}[/brand.primary]", end="")
        console.print("\n")

        engine.terminate()
        render_card("Engine Cleaned", "MLCEngine terminated cleanly.", icon="✔")
    except (ImportError, Exception):
        render_card(
            "Environment Note",
            "MLC LLM wheels (mlc-llm-nightly) and compiled .so library required for execution.",
            icon="ℹ️",
        )

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Machine Learning Compilation (MLC)",
                "Uses Apache TVM to compile model computational graphs directly into native C++ dynamic libraries (.so / .dylib / .dll).",
            ),
            (
                "Cross-Platform Native Execution",
                "Runs with bare-metal speed across iOS (Metal), Android (Vulkan/OpenCL), Web (WebGPU), and Linux GPUs (CUDA/ROCm) without Python runtime dependencies.",
            ),
            (
                "Streaming Token Latency",
                "Streaming responses provide low Time-To-First-Token (TTFT) and seamless user experience on mobile and edge devices.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
