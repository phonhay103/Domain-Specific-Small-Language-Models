"""Quantization of the GPT-2 Small Model (absmax).

Companion script for Chapter 6 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Introduces absmax quantization of a decoder-only language model (GPT-2 Small).
Does not require hardware acceleration.
"""

# Standard library
from copy import deepcopy

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
DEVICE = "cpu"
GENERATION_PROMPT = "My favourite school subject is"
MAX_GEN_LENGTH = 100
TOP_K = 30
HIST_BINS = 150
HIST_RANGE = (-2, 2)
PLOT_DPI = 300


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------
def absmax_quantize(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply absmax quantization/dequantization to tensor *X*.

    Returns:
        (X_quant, X_dequant): int8 quantized tensor and float dequantized tensor.
    """
    # Scale so that the maximum absolute value maps to 127
    scale = 127 / torch.max(torch.abs(X))
    X_quant = (scale * X).round()
    X_dequant = X_quant / scale
    return X_quant.to(torch.int8), X_dequant


def apply_absmax_to_model(model) -> tuple:
    """Clone *model* and replace all weights with their absmax-dequantized versions.

    Returns:
        (model_abs, weights_abs): cloned quantized model and list of dequantized weight tensors.
    """
    model_abs = deepcopy(model)
    weights_abs = []
    for param in model_abs.parameters():
        _, dequantized = absmax_quantize(param.data)
        param.data = dequantized
        weights_abs.append(dequantized)
    return model_abs, weights_abs


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def plot_weight_distributions(
    weights: np.ndarray,
    weights_abs: np.ndarray,
) -> None:
    """Plot overlapping histograms comparing original and absmax-quantized weights."""
    plt.style.use("ggplot")
    fig, axs = plt.subplots(1, figsize=(10, 10), dpi=PLOT_DPI, sharex=True)

    axs.hist(weights, bins=HIST_BINS, alpha=0.5, label="Original weights",
             color="blue", range=HIST_RANGE)
    axs.hist(weights_abs, bins=HIST_BINS, alpha=0.5, label="Absmax weights",
             color="yellow", range=HIST_RANGE)

    axs.grid(True, linestyle="--", alpha=0.6)
    axs.legend()
    axs.set_title("Comparison of Original and Absmax Quantized Weights", fontsize=16)
    axs.set_xlabel("Weights", fontsize=14)
    axs.set_ylabel("Count", fontsize=14)
    axs.yaxis.set_major_formatter(ticker.EngFormatter())  # Make y-ticks more human readable

    plt.rc("font", size=12)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Text generation & evaluation
# ---------------------------------------------------------------------------
def generate_text(model, tokenizer, input_text: str, max_length: int = MAX_GEN_LENGTH) -> str:
    """Generate text from *model* given *input_text*."""
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
    output = model.generate(
        inputs=input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=TOP_K,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=input_ids.new_ones(input_ids.shape),
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def calculate_perplexity(model, tokenizer, text: str, device: str) -> torch.Tensor:
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
    """Run absmax quantization demo: load, quantize, visualize, generate, evaluate."""
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"Model size: {model.get_memory_footprint():,} bytes")

    # Capture original weights before quantization modifies the clone
    original_weights = [param.data.clone() for param in model.parameters()]

    # Apply absmax quantization to a deep copy of the model
    model_abs, weights_abs_list = apply_absmax_to_model(model)

    # Flatten all weights for histogram comparison
    weights_flat = np.concatenate([t.cpu().numpy().flatten() for t in original_weights])
    weights_abs_flat = np.concatenate([t.cpu().numpy().flatten() for t in weights_abs_list])

    # Visualise weight distributions
    plot_weight_distributions(weights_flat, weights_abs_flat)

    # Generate text with both model versions
    original_text = generate_text(model, tokenizer, GENERATION_PROMPT)
    absmax_text = generate_text(model_abs, tokenizer, GENERATION_PROMPT)
    print(f"Original model:\n{original_text}")
    print(f"Absmax model:\n{absmax_text}")

    # Evaluate perplexity
    perplexity = calculate_perplexity(model, tokenizer, original_text, DEVICE)
    perplexity_absmax = calculate_perplexity(model_abs, tokenizer, absmax_text, DEVICE)
    print(f"Original perplexity:  {perplexity.item():.2f}")
    print(f"Absmax perplexity:    {perplexity_absmax.item():.2f}")


if __name__ == "__main__":
    main()
