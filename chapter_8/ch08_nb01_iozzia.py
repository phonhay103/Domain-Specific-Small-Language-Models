"""Generating Protein Sequences with ProtGPT2 Locally.

Companion to Chapter 8 of "Domain Specific LLMs in Action" by Guglielmo Iozzia,
Manning Publications, 2024.

Generates protein sequences using the ProtGPT2 model and evaluates them with
both per-sequence and batch perplexity metrics.
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
from transformers import pipeline

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
class ProteinSequenceRecord:
    """Immutable representation of a generated protein sequence."""

    index: int
    raw_sequence: str
    clean_sequence: str
    length_aa: int
    perplexity: float


MODEL_ID = "nferruz/ProtGPT2"
GENERATION_PROMPT = "<|endoftext|>"
GENERATION_MAX_LENGTH = 100
GENERATION_TOP_K = 950
GENERATION_REPETITION_PENALTY = 1.2
GENERATION_NUM_SEQUENCES = 10
GENERATION_EOS_TOKEN_ID = 0
EVAL_DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Pure Functions & Perplexity Computation
# ---------------------------------------------------------------------------
def clean_protein_text(raw_text: str) -> str:
    """Pure string cleaner: remove special EOS tokens and whitespace."""
    return raw_text.replace(GENERATION_PROMPT, "").strip()


def compute_sequence_perplexity(model: Any, tokenizer: Any, text: str, device: str = EVAL_DEVICE) -> float:
    """Pure per-sequence perplexity calculator using cross-entropy loss."""
    encodings = tokenizer(text, return_tensors="pt").to(device)
    input_ids = encodings.input_ids
    target_ids = input_ids.clone()
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
    return float(torch.exp(outputs.loss).item())


def compute_batch_perplexity_pure(input_texts: Sequence[str], model: Any, tokenizer: Any) -> float:
    """Pure batch perplexity calculator with attention masking."""
    inputs = tokenizer(list(input_texts), return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    target_log_probs = target_log_probs * attention_mask[:, 1:].to(log_probs.dtype)

    nll = -target_log_probs.sum(dim=-1) / attention_mask[:, 1:].sum(dim=-1)
    return float(torch.mean(torch.exp(nll)).item())


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_protein_sequences_table(records: Sequence[ProteinSequenceRecord]) -> None:
    """Render generated protein sequences and perplexities table."""
    columns = [
        ("Seq #", STYLE_NUMBER, "center"),
        ("Length", STYLE_SECONDARY, "right"),
        ("Perplexity", STYLE_SUCCESS, "right"),
        ("Amino Acid Sequence Preview", STYLE_TEXT, "left"),
    ]
    rows = [
        (
            str(r.index),
            f"{r.length_aa} aa",
            f"{r.perplexity:.3f}",
            r.clean_sequence[:65] + "..." if len(r.clean_sequence) > 65 else r.clean_sequence,
        )
        for r in records
    ]
    console.print(create_table("Generated De Novo Protein Sequences", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute ProtGPT2 sequence generation and perplexity evaluation."""
    render_banner(
        title="Generating Protein Sequences with ProtGPT2 Locally",
        subtitle="Chapter 8: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_ID,
            "Max Length": f"{GENERATION_MAX_LENGTH} aa",
            "Sample Count": str(GENERATION_NUM_SEQUENCES),
            "Top-K Sampling": str(GENERATION_TOP_K),
        },
        icon="🧬",
    )

    # Step 1: Loading Pipeline & Generating Sequences
    render_step(1, "Loading ProtGPT2 Pipeline & Generating Sequences", icon="📋")
    with status_spinner(f"Loading '{MODEL_ID}' text-generation pipeline..."):
        protgpt2 = pipeline("text-generation", model=MODEL_ID)

    with status_spinner(f"Sampling {GENERATION_NUM_SEQUENCES} de novo protein sequences..."):
        raw_outputs = protgpt2(
            GENERATION_PROMPT,
            max_length=GENERATION_MAX_LENGTH,
            do_sample=True,
            top_k=GENERATION_TOP_K,
            repetition_penalty=GENERATION_REPETITION_PENALTY,
            num_return_sequences=GENERATION_NUM_SEQUENCES,
            eos_token_id=GENERATION_EOS_TOKEN_ID,
        )

    # Step 2: Evaluating Sequence Perplexities
    render_step(2, "Evaluating Per-Sequence Naturalness & Perplexity", icon="🔬")
    records: list[ProteinSequenceRecord] = []
    with status_spinner("Computing negative log-likelihood across sequences..."):
        for i, seq_item in enumerate(raw_outputs, start=1):
            raw_text = seq_item["generated_text"]
            clean_seq = clean_protein_text(raw_text)
            ppl = compute_sequence_perplexity(protgpt2.model, protgpt2.tokenizer, raw_text, EVAL_DEVICE)
            records.append(
                ProteinSequenceRecord(
                    index=i,
                    raw_sequence=raw_text,
                    clean_sequence=clean_seq,
                    length_aa=len(clean_seq),
                    perplexity=ppl,
                )
            )

    render_protein_sequences_table(records)

    # Step 3: Batch Perplexity Calculation
    render_step(3, "Evaluating Mean Batch Perplexity", icon="📊")
    protgpt2.tokenizer.pad_token = protgpt2.tokenizer.eos_token
    mean_batch_ppl = compute_batch_perplexity_pure(
        [r.raw_sequence for r in records],
        protgpt2.model,
        protgpt2.tokenizer,
    )

    render_card(
        title="Batch Naturalness Evaluation",
        content=(
            f"[status.success]Mean Batch Perplexity:[/status.success] [text.highlight]{mean_batch_ppl:.3f}[/text.highlight]\n"
            f"[text.dim]Lower perplexity indicates higher alignment with natural UniProt evolutionary distributions.[/text.dim]"
        ),
        icon="✨",
    )

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Protein Language Models (pLMs)",
                "Treat amino acid residues (A, C, D, E, etc.) as vocabulary tokens. The model learns biological evolutionary constraints and secondary folding physics directly from sequence pretraining.",
            ),
            (
                "Perplexity as a Naturalness Filter",
                "Generated de novo proteins with low perplexity (PPL < 15) match known folding distributions in UniProt/SwissProt, making them prime candidates for wet-lab synthesis.",
            ),
            (
                "Conditional Generation",
                "Can be primed with catalytic triad motifs or N-terminal leader sequences to design targeted biocatalysts.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
