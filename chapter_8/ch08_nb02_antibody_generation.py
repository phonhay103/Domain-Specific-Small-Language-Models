"""Antibody Generation with AntibodyGPT.

Companion script for Chapter 8 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Generates antibody sequences for a target antigen using the AntibodyGPT model
(AntibodyGeneration/fine-tuned-progen2-small).
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
from tokenizers import Tokenizer

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
class AntibodyCandidate:
    """Immutable representation of a generated antibody candidate."""

    candidate_id: int
    sequence: str
    length_aa: int


MODEL_ID: str = "AntibodyGeneration/fine-tuned-progen2-small"
LOCAL_MODEL_DIR: Path = Path("antibodygen")

TARGET_SEQUENCE: str = (
    "MQIPQAPWPVVWAVLQLGWRPGWFLDSPDRPWNPPTFSPALLVVTEGDNATFTCSFSNTSESFVLNWYRMSPSNQTDKLAAFPEDR"
    "SQPGQDCRFRVTQLPNGRDFHMSVVRARRNDSGTYLCGAISLAPKAQIKESLRAELRVTERRAEVPTAHPSPSPRPAGQFQTLVVGV"
    "VGGLLGSLVLLVWVLAVICSRAARGTIGARRTGQPLKEDPSAVPVFSVDYGELDFQWREKTPEPPVPCVPEQTEYATIVFPSGMGTS"
    "SPARRGSADGPRSAQPLRPEDGHCSWPL"
)

NUMBER_OF_SEQUENCES: int = 2
MAX_GENERATION_LENGTH: int = 1024
TOP_P: float = 0.9
TEMPERATURE: float = 0.8


# ---------------------------------------------------------------------------
# Pure Functions & Helpers
# ---------------------------------------------------------------------------
def clean_antibody_sequence(raw_seq: str) -> str:
    """Pure string cleaner: remove artifact tokens."""
    return raw_seq.replace("2", "").strip()


def create_antibody_candidates(raw_sequences: Sequence[str]) -> tuple[AntibodyCandidate, ...]:
    """Pure transformation into immutable candidates."""
    return tuple(
        AntibodyCandidate(
            candidate_id=i,
            sequence=clean_antibody_sequence(seq),
            length_aa=len(clean_antibody_sequence(seq)),
        )
        for i, seq in enumerate(raw_sequences, start=1)
    )


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_candidates_table(candidates: Sequence[AntibodyCandidate]) -> None:
    """Render generated antibody candidates table."""
    columns = [
        ("Candidate #", STYLE_NUMBER, "center"),
        ("Length (aa)", STYLE_SECONDARY, "right"),
        ("Sequence Preview", STYLE_TEXT, "left"),
    ]
    rows = [
        (
            f"Antibody {c.candidate_id}",
            f"{c.length_aa} aa",
            c.sequence[:70] + "..." if len(c.sequence) > 70 else c.sequence,
        )
        for c in candidates
    ]
    console.print(create_table("Generated Antigen-Conditioned Antibodies", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute AntibodyGPT generation pipeline."""
    render_banner(
        title="Antibody Generation with AntibodyGPT (ProGen2 Small)",
        subtitle="Chapter 8: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_ID,
            "Sequences": str(NUMBER_OF_SEQUENCES),
            "Temperature": str(TEMPERATURE),
            "Top-P": str(TOP_P),
        },
        icon="💉",
    )

    # Step 1: Target Antigen Sequence
    render_step(1, "Target Antigen Sequence Specification", icon="🧬")
    render_card(
        title="Target Antigen",
        content=(
            f"[text.muted]Sequence Length:[/text.muted] [brand.secondary]{len(TARGET_SEQUENCE)} amino acids[/brand.secondary]\n\n"
            f"[text.dim]{TARGET_SEQUENCE}[/text.dim]"
        ),
        icon="🎯",
    )

    # Step 2: Loading AntibodyGPT Model
    render_step(2, "Loading AntibodyGPT Weights & Tokenizer", icon="🧠")
    try:
        from models.progen.modeling_progen import ProGenForCausalLM

        with status_spinner(f"Loading '{MODEL_ID}'..."):
            model = ProGenForCausalLM.from_pretrained(MODEL_ID)
            tokenizer = Tokenizer.from_pretrained(MODEL_ID)
            model.save_pretrained(LOCAL_MODEL_DIR)
        render_card(
            "Model Ready",
            f"AntibodyGPT model loaded and cached to [text.highlight]{LOCAL_MODEL_DIR}[/text.highlight]",
            icon="✔",
        )
    except ImportError:
        render_card("Environment Note", "Official ProGen repository classes required in PYTHONPATH.", icon="⚠️")
        return

    # Step 3: Generating Conditioned Sequences
    render_step(3, "Sampling Conditioned Antibody Complementarity Regions", icon="✨")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenized = tokenizer.encode(TARGET_SEQUENCE)
    input_tensor = torch.tensor([tokenized.ids]).to(device)
    model = model.to(device)
    render_device_info(device, model=model)
    pad_id = tokenizer.encode("<|pad|>").ids[0]

    with status_spinner(f"Sampling {NUMBER_OF_SEQUENCES} antibody sequences conditioned on antigen..."):
        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_length=MAX_GENERATION_LENGTH,
                pad_token_id=pad_id,
                do_sample=True,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                num_return_sequences=NUMBER_OF_SEQUENCES,
            )

    as_lists = [output[i, ...].detach().cpu().numpy().tolist() for i in range(output.shape[0])]
    raw_seqs = tokenizer.decode_batch(as_lists)
    candidates = create_antibody_candidates(raw_seqs)

    # Step 4: Displaying Generated Sequences
    render_step(4, "Displaying Antibody Candidate Structures", icon="📊")
    render_candidates_table(candidates)

    for c in candidates:
        render_card(f"Antibody Candidate #{c.candidate_id} ({c.length_aa} aa)", c.sequence, icon="🔬")

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Antigen Conditioning",
                "By prepending target antigen epitopes to the context prompt, the generative model samples complementary CDR (Complementarity-Determining Region) loops.",
            ),
            (
                "ProGen2 Architecture",
                "Tailored for protein and antibody sequence generation, incorporating evolutionary token probabilities across billions of natural immunoglobulins.",
            ),
            (
                "Sampling Hyperparameters",
                "Top-p (0.9) and Temperature (0.8) balance structural novelty with physicochemical viability (avoiding hydrophobic aggregation).",
            ),
        ),
    )


if __name__ == "__main__":
    main()
