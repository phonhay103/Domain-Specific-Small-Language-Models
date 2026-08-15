"""Quantization of the GPT-2 Small Model with LLM.int8().

Companion script for Chapter 6 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Demonstrates LLM.int8() quantization (https://arxiv.org/abs/2208.07339) of
GPT-2 Small using the bitsandbytes library. Requires hardware acceleration (GPU).

Install prerequisites:
    pip install accelerate bitsandbytes
"""

# Third-party
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "openai-community/gpt2"
GENERATION_PROMPT = "My favourite school subject is"
MAX_GEN_LENGTH = 100
TOP_K = 30
HIST_BINS = 150
HIST_RANGE = (-2, 2)
PLOT_DPI = 300


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_fp32_model(model_id: str):
    """Load the full-precision (FP32) GPT-2 model with automatic device mapping."""
    return AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")


def load_int8_model(model_id: str):
    """Load the LLM.int8() quantized GPT-2 model with automatic device mapping."""
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_8bit=True,
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def collect_weights(model) -> np.ndarray:
    """Flatten all parameter tensors of *model* into a single 1-D numpy array."""
    return np.concatenate(
        [param.data.clone().cpu().numpy().flatten() for param in model.parameters()]
    )


def plot_weight_distributions(
    weights: np.ndarray,
    weights_int8: np.ndarray,
) -> None:
    """Plot overlapping histograms comparing FP32 and LLM.int8() weights."""
    plt.style.use("ggplot")
    fig, axs = plt.subplots(1, figsize=(10, 10), dpi=PLOT_DPI, sharex=True)

    axs.hist(weights, bins=HIST_BINS, alpha=0.5, label="Original weights",
             color="blue", range=HIST_RANGE)
    axs.hist(weights_int8, bins=HIST_BINS, alpha=0.5, label="LLM.int8() weights",
             color="yellow", range=HIST_RANGE)

    axs.grid(True, linestyle="--", alpha=0.6)
    axs.legend()
    axs.set_title("Comparison of Original and LLM.int8() Weights", fontsize=16)
    axs.set_xlabel("Weights", fontsize=14)
    axs.set_ylabel("Count", fontsize=14)
    axs.yaxis.set_major_formatter(ticker.EngFormatter())  # Make y-ticks more human readable

    plt.rc("font", size=12)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Text generation & evaluation
# ---------------------------------------------------------------------------
def generate_text(
    model,
    tokenizer,
    input_text: str,
    device: torch.device,
    max_length: int = MAX_GEN_LENGTH,
) -> str:
    """Generate text from *model* given *input_text*."""
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output = model.generate(
        inputs=input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=TOP_K,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=input_ids.new_ones(input_ids.shape),
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def calculate_perplexity(
    model,
    tokenizer,
    text: str,
    device: torch.device,
) -> torch.Tensor:
    """Calculate perplexity of *model* on *text*."""
    encodings = tokenizer(text, return_tensors="pt").to(device)
    input_ids = encodings.input_ids
    target_ids = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

    neg_log_likelihood = outputs.loss
    return torch.exp(neg_log_likelihood)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    """Run LLM.int8() quantization demo: load, visualize, generate, evaluate."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load both model variants
    model = load_fp32_model(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"Model size: {model.get_memory_footprint():,} bytes")

    model_int8 = load_int8_model(MODEL_ID)
    print(f"Model size (int8): {model_int8.get_memory_footprint():,} bytes")

    # Compare weight distributions
    weights = collect_weights(model)
    weights_int8 = collect_weights(model_int8)
    plot_weight_distributions(weights, weights_int8)

    # Generate text with both model versions
    original_text = generate_text(model, tokenizer, GENERATION_PROMPT, device)
    text_int8 = generate_text(model_int8, tokenizer, GENERATION_PROMPT, device)
    print(f"Original model:\n{original_text}")
    print(f"LLM.int8() model:\n{text_int8}")

    # Evaluate perplexity for both versions
    perplexity = calculate_perplexity(model, tokenizer, original_text, device)
    perplexity_int8 = calculate_perplexity(model_int8, tokenizer, text_int8, device)
    print(f"Original Perplexity:   {perplexity.item():.2f}")
    print(f"LLM.int8() perplexity: {perplexity_int8.item():.2f}")


if __name__ == "__main__":
    main()
