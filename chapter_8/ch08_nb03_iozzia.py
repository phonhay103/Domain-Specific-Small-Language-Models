"""Generating Crystal Structures with CrystaLLM.

Companion script for Chapter 8 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Converts a CrystaLLM pretrained model checkpoint to Hugging Face Transformers
format so it can be pushed to the HF Hub.  The script does NOT require GPU
acceleration.

Setup notes (run once before executing this script):
  # Clone the CrystaLLM repo:
  # git clone https://github.com/lantunes/CrystaLLM.git
  #
  # Install missing dependencies:
  # pip install pymatgen==2023.3.23 omegaconf
  #
  # Add CrystaLLM to the Python path and change into the repo dir, then:
  # python bin/download.py crystallm_v1_small.tar.gz
  # tar xvf crystallm_v1_small.tar.gz
  #
  # Generate a sample prompt and run inference / MCTS (see comments below).
  # Evaluate generated CIF files:
  #   python bin/evaluate_cifs.py colab_processed_cifs.tar.gz -o colab_processed_cifs.csv
  #
  # NOTE: Please read the CrystaLLM OS license before sharing any checkpoints
  # through the HF Hub without consent from the original authors.
  #
  # Upgrade NumPy if needed:
  # pip install -U numpy

# CIF generation commands (illustrative — run from within CrystaLLM/):
#   python bin/make_prompt_file.py Na2Cl2 sample_prompt.txt --spacegroup P4/nmm
#   python bin/sample.py out_dir=crystallm_v1_small start=FILE:sample_prompt.txt \
#       num_samples=2 top_k=10 max_new_tokens=3000 device=cpu target=file
#   python bin/postprocess.py . colab_processed_cifs
#
# Monte Carlo Tree Search alternative:
#   python bin/mcts.py out_dir=crystallm_v1_small device=cpu dtype=bfloat16 \
#       start=FILE:sample_prompt.txt tree_width=5 max_depth=2000 selector=puct \
#       c=1.0 num_simulations=1000 reward_k=2.0 scorer=random \
#       top_child_weight_cutoff=0.9999 bypass_only_child=True \
#       mcts_out_dir=colab_mcts_cifs
"""

# stdlib
import os
import sys

# third-party
import torch
from transformers import AutoConfig, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CRYSTALLM_REPO_PATH: str = "/content/CrystaLLM"
MODEL_SUBDIR: str = "crystallm_v1_small"
HF_OUTPUT_SUBDIR: str = "crystallm_v1_small_hf"

# JSON configuration for the CrystaLLM small model (GPT-2 architecture)
CRYSTALL_SMALL_CONFIG: str = """{
  "bias": true,
  "model_type": "gpt2",
  "block_size": 1024,
  "dropout": 0.1,
  "n_embd": 512,
  "n_head": 8,
  "n_layer": 8,
  "vocab_size": 50257
}"""


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def configure_python_path(repo_path: str) -> None:
    """Append *repo_path* to sys.path and PYTHONPATH so CrystaLLM is importable."""
    sys.path.append(repo_path)
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + f":{repo_path}"


def write_model_config(model_dir: str, config_json: str) -> None:
    """Append the JSON *config_json* to config.json inside *model_dir*."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "a") as f:
        f.write(config_json)
    print(f"Config written to {config_path}")


def convert_checkpoint_to_bin(model_dir: str) -> None:
    """Convert a PyTorch .pt checkpoint to a .bin file for Transformers compatibility.

    Loads the checkpoint, extracts model parameters, and saves them in the
    format expected by AutoModelForCausalLM.from_pretrained().
    """
    ckpt_path = os.path.join(model_dir, "ckpt.pt")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")

    # Load the checkpoint file
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))

    # Extract the model parameters
    params = checkpoint["model"]

    # Save the parameters to a .bin file
    torch.save(params, bin_path)
    print(f"Checkpoint converted and saved to {bin_path}")


def load_as_transformer_model(model_dir: str) -> AutoModelForCausalLM:
    """Load the converted checkpoint through the Transformers AutoModel API."""
    config_path = os.path.join(model_dir, "config.json")
    config = AutoConfig.from_pretrained(config_path)
    model = AutoModelForCausalLM.from_pretrained(model_dir, config=config)
    return model


def save_for_hub(model: AutoModelForCausalLM, output_dir: str) -> None:
    """Save *model* in HF format to *output_dir*, ready for Hub upload.

    save_pretrained() also generates any required accessory files
    (tokenizer config, special tokens map, etc.).
    """
    model.save_pretrained(output_dir)
    print(f"HF-format model saved to {output_dir}")
    print(
        "The model is now ready to be uploaded to the HF Hub.\n"
        "Please verify the CrystaLLM OS license before sharing checkpoints publicly."
    )


def main() -> None:
    """Orchestrate CrystaLLM → Transformers conversion pipeline."""
    model_dir = os.path.join(CRYSTALLM_REPO_PATH, MODEL_SUBDIR)
    hf_output_dir = os.path.join(CRYSTALLM_REPO_PATH, HF_OUTPUT_SUBDIR)

    configure_python_path(CRYSTALLM_REPO_PATH)
    write_model_config(model_dir, CRYSTALL_SMALL_CONFIG)
    convert_checkpoint_to_bin(model_dir)

    transformer_model = load_as_transformer_model(model_dir)
    save_for_hub(transformer_model, hf_output_dir)


if __name__ == "__main__":
    main()
