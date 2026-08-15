"""
Generating Protein Sequences with ProtGPT2 Locally.

Companion to Chapter 8 of "Domain Specific LLMs in Action" by Guglielmo Iozzia,
Manning Publications, 2024.

Generates protein sequences using the ProtGPT2 model and evaluates them with
both per-sequence and batch perplexity metrics. GPU acceleration is not required.
"""

from typing import Dict, List

import torch
from transformers import pipeline

# Model constant
MODEL_ID = "nferruz/ProtGPT2"

# Generation hyperparameters
GENERATION_PROMPT = "<|endoftext|>"
GENERATION_MAX_LENGTH = 100
GENERATION_TOP_K = 950
GENERATION_REPETITION_PENALTY = 1.2
GENERATION_NUM_SEQUENCES = 10
GENERATION_EOS_TOKEN_ID = 0

# Perplexity evaluation settings
EVAL_DEVICE = "cpu"


# ---- Model loading ----

def load_pipeline(model_id: str):
    """Load the ProtGPT2 text-generation pipeline from the HF Hub."""
    return pipeline("text-generation", model=model_id)


# ---- Sequence generation ----

def generate_sequences(protgpt2) -> List[Dict]:
    """Generate protein sequences using ProtGPT2 and return raw output dicts."""
    return protgpt2(
        GENERATION_PROMPT,
        max_length=GENERATION_MAX_LENGTH,
        do_sample=True,
        top_k=GENERATION_TOP_K,
        repetition_penalty=GENERATION_REPETITION_PENALTY,
        num_return_sequences=GENERATION_NUM_SEQUENCES,
        eos_token_id=GENERATION_EOS_TOKEN_ID,
    )


# ---- Perplexity metrics ----

def calculate_perplexity(model, tokenizer, text: str, device: str) -> torch.Tensor:
    """Compute per-sequence perplexity using the model's NLL loss."""
    encodings = tokenizer(text, return_tensors="pt").to(device)

    input_ids = encodings.input_ids
    target_ids = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

    neg_log_likelihood = outputs.loss
    perplexity = torch.exp(neg_log_likelihood)
    return perplexity


def calculate_batch_perplexity(
    input_texts: List[str],
    model,
    tokenizer,
) -> Dict[str, torch.Tensor]:
    """
    Calculate perplexity for a batch of input texts using a pretrained language model.

    Args:
        input_texts: A list of input texts to evaluate.

    Returns:
        A dict with 'perplexities' (one score per sequence) and 'mean_perplexity'.
    """
    # Tokenize the batch of texts with padding for uniform length
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Pass the input batch through the model to get logits
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Shift the logits and input_ids to align targets correctly
    # Logits dimensions are: (batch_size, seq_length, vocab_size)
    shift_logits = logits[:, :-1, :]  # Ignore the last token's logits
    shift_labels = input_ids[:, 1:]   # Skip the first token in the labels

    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Gather the log probabilities for the correct tokens
    target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask out positions corresponding to padding tokens
    target_log_probs = target_log_probs * attention_mask[:, 1:].to(log_probs.dtype)

    # Compute the mean negative log-likelihood for each sequence
    negative_log_likelihood = (
        -target_log_probs.sum(dim=-1) / attention_mask[:, 1:].sum(dim=-1)
    )

    # Compute perplexity for each sequence
    perplexities = torch.exp(negative_log_likelihood)

    # Take mean of perplexities of each batch
    mean_perplexity_score = torch.mean(perplexities)

    return {"perplexities": perplexities, "mean_perplexity": mean_perplexity_score}


# ---- Main orchestration ----

def main() -> None:
    """Load ProtGPT2, generate protein sequences, and evaluate perplexity."""
    protgpt2 = load_pipeline(MODEL_ID)

    # Generate protein sequences
    sequences = generate_sequences(protgpt2)
    for seq in sequences:
        print(seq)

    # Per-sequence perplexity
    for seq in sequences:
        print(
            calculate_perplexity(
                protgpt2.model, protgpt2.tokenizer, seq["generated_text"], EVAL_DEVICE
            )
        )

    # Batch perplexity — set pad token to EOS before batching
    protgpt2.tokenizer.pad_token = protgpt2.tokenizer.eos_token
    sequence_texts = [seq["generated_text"] for seq in sequences]
    print(
        f"Perplexity scores: "
        f"{calculate_batch_perplexity(sequence_texts, protgpt2.model, protgpt2.tokenizer)}"
    )


if __name__ == "__main__":
    main()
