"""Antibody Generation with AntibodyGPT.

Companion script for Chapter 8 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Generates antibody sequences for a target antigen using the AntibodyGPT model
(AntibodyGeneration/fine-tuned-progen2-small). Requires hardware acceleration.

Setup notes:
  # Downgrade HF Transformers for compatibility with ProGenForCausalLM.
  # ProGenForCausalLM inherits from PreTrainedModel; starting from Transformers
  # 4.50 it no longer inherits from GenerationMixin, losing the generate() method.
  # pip install transformers==4.49.0
  #
  # Clone the official repo first:
  # git clone https://github.com/joethequant/docker_protein_generator.git
  # Then run this script from inside docker_protein_generator/.
"""

# stdlib
from pathlib import Path

# third-party
import torch
from tokenizers import Tokenizer

from models.progen.modeling_progen import ProGenForCausalLM  # from cloned repo

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID: str = "AntibodyGeneration/fine-tuned-progen2-small"
LOCAL_MODEL_DIR: Path = Path("antibodygen")

# Target antigen sequence to generate antibodies for
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
# Functions
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_id: str,
    local_dir: Path,
) -> tuple[ProGenForCausalLM, Tokenizer]:
    """Download (or load) the AntibodyGPT model and tokenizer.

    Saves the model locally to *local_dir* on first run (~588 MB on disk).
    """
    model = ProGenForCausalLM.from_pretrained(model_id)
    tokenizer = Tokenizer.from_pretrained(model_id)
    model.save_pretrained(local_dir)
    return model, tokenizer


def generate_antibody_sequences(
    model: ProGenForCausalLM,
    tokenizer: Tokenizer,
    target_sequence: str,
    num_sequences: int,
    max_length: int,
    top_p: float,
    temperature: float,
) -> list[str]:
    """Generate antibody sequences conditioned on *target_sequence*.

    Tokenises the antigen sequence, moves tensors to the available device,
    runs greedy-with-sampling decoding, then decodes and cleans the outputs.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenized = tokenizer.encode(target_sequence)
    input_tensor = torch.tensor([tokenized.ids]).to(device)

    model = model.to(device)

    pad_id = tokenizer.encode("<|pad|>").ids[0]

    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_length=max_length,
            pad_token_id=pad_id,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=num_sequences,
        )

    # Convert output tensor to nested Python lists for batch decoding
    as_lists = lambda batch: [
        batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])
    ]
    sequences = tokenizer.decode_batch(as_lists(output))

    # Remove padding/special token artifacts
    if sequences:
        sequences = [seq.replace("2", "") for seq in sequences]

    return sequences


def main() -> None:
    """Orchestrate model loading, generation, and result display."""
    model, tokenizer = load_model_and_tokenizer(MODEL_ID, LOCAL_MODEL_DIR)

    sequences = generate_antibody_sequences(
        model=model,
        tokenizer=tokenizer,
        target_sequence=TARGET_SEQUENCE,
        num_sequences=NUMBER_OF_SEQUENCES,
        max_length=MAX_GENERATION_LENGTH,
        top_p=TOP_P,
        temperature=TEMPERATURE,
    )

    print(f"Generated {len(sequences)} antibody sequence(s):")
    for i, seq in enumerate(sequences, 1):
        print(f"\n[{i}] {seq}")


if __name__ == "__main__":
    main()
