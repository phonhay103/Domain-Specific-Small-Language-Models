"""4-bit Quantization of GPT-2 with Auto-GPTQ.

This script is a companion of chapter 6 of the "Domain Specific LLMs in Action"
book, author Guglielmo Iozzia, Manning Publications, 2024.
It introduces readers to 4-bit quantization of a decoder-only language model,
GPT-2, using the AutoGPTQ library. It requires hardware acceleration (CUDA).
More details about the code can be found in the related book's chapter.

# Install notes (run once in your environment):
#   export BUILD_CUDA_EXT=0
#   pip install -q auto-gptq
#   pip install --force-reinstall datasets
"""

# --- stdlib ---
import random

# --- third-party ---
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
from transformers import AutoTokenizer, TextGenerationPipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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
# Functions
# ---------------------------------------------------------------------------

def get_wikitext2(nsamples: int, seed: int, seqlen: int, tokenizer) -> tuple:
    """Load and prepare WikiText-2 calibration data for quantization."""
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Load dataset and preprocess
    traindata = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split="train")
    testdata = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split="test")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset, testenc


def load_tokenizer(model_id: str) -> AutoTokenizer:
    """Load the tokenizer, falling back to fast tokenizer on failure."""
    try:
        return AutoTokenizer.from_pretrained(model_id, use_fast=False)
    except Exception:
        return AutoTokenizer.from_pretrained(model_id, use_fast=True)


def load_and_configure_model(model_id: str, quantize_config: BaseQuantizeConfig):
    """Download the unquantized model and resolve its sequence length from config."""
    model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)
    model_config = model.config.to_dict()
    seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
    if any(k in model_config for k in seq_len_keys):
        for key in seq_len_keys:
            if key in model_config:
                model.seqlen = model_config[key]
                break
    else:
        print(
            "The model's sequence length cannot be retrieved from its configuration. "
            f"It will then be set to {DEFAULT_SEQ_LEN}."
        )
        model.seqlen = DEFAULT_SEQ_LEN
    return model


def quantize_and_save(model, traindataset: list, save_dir: str) -> None:
    """Quantize the model with the calibration dataset and persist to disk.

    The saved safetensors file is approximately 1.02 GB.
    """
    # The examples must be a list of dicts with "input_ids" and "attention_mask" keys
    model.quantize(traindataset, use_triton=False)
    model.save_quantized(save_dir, use_safetensors=True)


def run_inference(quantized_model, tokenizer, prompt: str) -> None:
    """Run direct generation and pipeline inference with the quantized model."""
    # Direct generation
    output = tokenizer.decode(
        quantized_model.generate(
            **tokenizer(prompt, return_tensors="pt").to(CUDA_DEVICE)
        )[0]
    )
    print(output)

    # HF Transformers pipelines are supported too for 4-bit quantized models
    gen_pipeline = TextGenerationPipeline(
        model=quantized_model, tokenizer=tokenizer, device=CUDA_DEVICE
    )
    print(gen_pipeline(prompt)[0]["generated_text"])


def plot_weight_comparison(model_id: str, quantize_config, quantized_model) -> None:
    """Display histograms comparing original and 4-bit quantized model weights."""
    original_model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)

    weights = np.concatenate(
        [p.data.clone().cpu().numpy().flatten() for p in original_model.parameters()]
    )
    weights_int8 = np.concatenate(
        [p.data.clone().cpu().numpy().flatten() for p in quantized_model.parameters()]
    )

    plt.style.use("ggplot")
    fig, axs = plt.subplots(1, figsize=(10, 10), dpi=300, sharex=True)

    # Plot histograms for original and quantized weights
    axs.hist(weights, bins=150, alpha=0.5, label="Original weights",
             color="yellow", range=(-0.5, 0.5))
    axs.hist(weights_int8, bins=150, alpha=0.5, label="LLM.int8() weights",
             color="blue", range=(-0.5, 0.5))

    axs.grid(True, linestyle="--", alpha=0.6)
    axs.legend()
    axs.set_title("Comparison of Original and LLM.int8() Weights", fontsize=16)
    axs.set_xlabel("Weights", fontsize=14)
    axs.set_ylabel("Count", fontsize=14)
    axs.yaxis.set_major_formatter(ticker.EngFormatter())  # Human-readable y-ticks

    plt.rc("font", size=12)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Orchestrate tokenizer loading, model quantization, inference, and weight comparison."""
    tokenizer = load_tokenizer(MODEL_ID)

    quantize_config = BaseQuantizeConfig(
        bits=QUANTIZE_BITS,
        group_size=QUANTIZE_GROUP_SIZE,
        desc_act=False,
    )

    model = load_and_configure_model(MODEL_ID, quantize_config)

    # Prepare calibration dataset and quantize
    traindataset, _ = get_wikitext2(
        NUM_CALIBRATION_SAMPLES, CALIBRATION_SEED, model.seqlen, tokenizer
    )
    quantize_and_save(model, traindataset, QUANTIZED_MODEL_DIR)

    # Load quantized model for inference
    quantized_model = AutoGPTQForCausalLM.from_quantized(
        QUANTIZED_MODEL_DIR, device=CUDA_DEVICE, use_triton=False
    )

    run_inference(quantized_model, tokenizer, INFERENCE_PROMPT)
    plot_weight_comparison(MODEL_ID, quantize_config, quantized_model)


if __name__ == "__main__":
    main()
