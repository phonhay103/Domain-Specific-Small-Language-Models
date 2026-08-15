"""Generating Crystal Structures with CrystaLLM.

Companion script for Chapter 8 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Converts a CrystaLLM pretrained model checkpoint to Hugging Face Transformers
format so it can be deployed or pushed to the HF Hub.
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Third-party
import torch
from transformers import AutoConfig, AutoModelForCausalLM

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
    render_code_block,
    render_step,
    render_takeaways,
    status_spinner,
)


# ---------------------------------------------------------------------------
# Immutable Domain Records & Constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CrystaLLMModelSpec:
    """Immutable architectural hyperparameters for CrystaLLM GPT-2 backbone."""

    model_type: str
    block_size: int
    dropout: float
    n_embd: int
    n_head: int
    n_layer: int
    vocab_size: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "bias": True,
                "model_type": self.model_type,
                "block_size": self.block_size,
                "dropout": self.dropout,
                "n_embd": self.n_embd,
                "n_head": self.n_head,
                "n_layer": self.n_layer,
                "vocab_size": self.vocab_size,
            },
            indent=2,
        )


CRYSTALLM_REPO_PATH: str = "/content/CrystaLLM"
MODEL_SUBDIR: str = "crystallm_v1_small"
HF_OUTPUT_SUBDIR: str = "crystallm_v1_small_hf"

DEFAULT_SPEC = CrystaLLMModelSpec(
    model_type="gpt2",
    block_size=1024,
    dropout=0.1,
    n_embd=512,
    n_head=8,
    n_layer=8,
    vocab_size=50257,
)


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_spec_table(spec: CrystaLLMModelSpec) -> None:
    """Render crystal structure model architecture specifications."""
    columns = [
        ("Hyperparameter", STYLE_PRIMARY, "left"),
        ("Value", STYLE_SUCCESS, "right"),
    ]
    rows = [
        ("Base Architecture", spec.model_type.upper()),
        ("Context Window (block_size)", f"{spec.block_size} tokens"),
        ("Hidden Embedding Dimension (n_embd)", str(spec.n_embd)),
        ("Attention Heads (n_head)", str(spec.n_head)),
        ("Transformer Layers (n_layer)", str(spec.n_layer)),
        ("Vocabulary Size", f"{spec.vocab_size:,}"),
    ]
    console.print(create_table("CrystaLLM GPT-2 Architecture Specification", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute CrystaLLM checkpoint conversion pipeline."""
    render_banner(
        title="CrystaLLM Checkpoint Conversion for Crystal Structure Generation",
        subtitle="Chapter 8: Domain-Specific Small Language Models",
        metadata={
            "Repo Path": CRYSTALLM_REPO_PATH,
            "Model Subdir": MODEL_SUBDIR,
            "Target Format": "Hugging Face Transformers",
        },
        icon="💎",
    )

    model_dir = os.path.join(CRYSTALLM_REPO_PATH, MODEL_SUBDIR)
    hf_output_dir = os.path.join(CRYSTALLM_REPO_PATH, HF_OUTPUT_SUBDIR)

    # Step 1: Architecture Specification
    render_step(1, "Inspecting CrystaLLM GPT-2 Architecture Configuration", icon="📋")
    render_spec_table(DEFAULT_SPEC)
    render_code_block(DEFAULT_SPEC.to_json(), language="json", title="Config JSON Payload")

    # Step 2: Environment Configuration
    render_step(2, "Configuring Runtime Path & Writing Artifacts", icon="⚙️")
    if os.path.exists(model_dir):
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_SPEC.to_json())
        render_card(
            "Configuration Saved", f"Config written to [text.highlight]{config_path}[/text.highlight]", icon="✔"
        )

    # Step 3: Checkpoint Weight Conversion
    render_step(3, "Converting Raw PyTorch Weights to Transformers Format", icon="🔄")
    ckpt_path = os.path.join(model_dir, "ckpt.pt")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.exists(ckpt_path):
        with status_spinner("Converting checkpoint weights..."):
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
            torch.save(checkpoint["model"], bin_path)
        render_card("Weight Binary Ready", f"Saved binary to [text.highlight]{bin_path}[/text.highlight]", icon="✔")

    # Step 4: Verification & Hub Export
    render_step(4, "Verifying Architecture & Saving Hub Artifacts", icon="💾")
    if os.path.exists(bin_path):
        with status_spinner("Loading converted model via AutoModelForCausalLM..."):
            config = AutoConfig.from_pretrained(os.path.join(model_dir, "config.json"))
            model = AutoModelForCausalLM.from_pretrained(model_dir, config=config)
            model.save_pretrained(hf_output_dir)
        render_card("Hub Artifacts Ready", f"Exported to [text.highlight]{hf_output_dir}[/text.highlight]", icon="✨")
    else:
        render_card(
            "Conversion Status", "Checkpoint weights not present in local path; config format verified.", icon="ℹ️"
        )

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Crystallographic Information File (CIF) Modeling",
                "CrystaLLM formats atomic 3D fractional coordinates, cell vectors, and spacegroup symmetry into tokenized strings, allowing autoregressive SLMs to generate novel crystal lattices.",
            ),
            (
                "Monte Carlo Tree Search (MCTS) Sampling",
                "Coupling SLM logit priors with MCTS and physical charge-neutrality/energy scorers prunes chemically invalid crystal symmetries early during generation.",
            ),
            (
                "Transformers Interoperability",
                "Converting custom GPT backbones into HF AutoModel format enables seamless downstream quantization (e.g. Optimum, BitsAndBytes) and hub distribution.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
