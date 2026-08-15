"""GPT-Neo inference with the HF's Transformers Library.

This script is a companion of chapter 4 of the "Domain Specific LLMs in Action"
book, author Guglielmo Iozzia, Manning Publications, 2024.
The code introduces readers to inference (text generation) with the GPT-Neo model
using the Hugging Face Transformers library. It can be executed with hardware
acceleration (GPU).
More details about the code can be found in the book's chapter.

# Install the missing requirements before running (HF's Accelerate only):
#   pip install accelerate
"""

import time

import numpy as np
import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "EleutherAI/gpt-neo-2.7B"
PAD_TOKEN_ID = 50256
TEMPERATURE = 0.9
MAX_LENGTH = 200
MAX_LENGTH_LONG = 300
BENCHMARK_RUNS = 20
BENCHMARK_RUNS_TOTAL = 21

COMPLETION_PROMPT = "The story so far: in the beginning, the universe was created."

FEW_SHOT_PROMPT = """
Sentence: This movie is very nice.
Sentiment: positive

#####

Sentence: I hated this movie, it sucks.
Sentiment: negative

#####

Sentence: This movie was actually pretty funny.
Sentiment: positive

#####

Sentence: This movie could have been better.
Sentiment: neutral
"""

CODE_GEN_PROMPT = (
    "Instruction: Generate a Python function that lets you reverse a list of integers.\n\n"
    "Answer: "
)

BATCH_TEXTS = [
    "Once there was a man ",
    "The weather today will be ",
    "A great soccer player must ",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_id: str, device: torch.device) -> tuple[GPTNeoForCausalLM, GPT2Tokenizer]:
    """Download and load GPT-Neo model and tokenizer from the HF Hub."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    model = GPTNeoForCausalLM.from_pretrained(model_id, device_map="auto")
    model.to(device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def generate_text(
    model: GPTNeoForCausalLM,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    device: torch.device,
    max_length: int = MAX_LENGTH,
) -> str:
    """Run single-prompt text generation and return decoded output."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        temperature=TEMPERATURE,
        max_length=max_length,
        pad_token_id=PAD_TOKEN_ID,
    )
    return tokenizer.decode(generated_ids[0])


def run_batch_completion(
    model: GPTNeoForCausalLM,
    tokenizer: GPT2Tokenizer,
    texts: list[str],
    device: torch.device,
    max_length: int = 50,
) -> list[str]:
    """Run batch text completion and return decoded outputs."""
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    encoding = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            **encoding,
            do_sample=True,
            temperature=TEMPERATURE,
            max_length=max_length,
            pad_token_id=PAD_TOKEN_ID,
        )
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------
def benchmark_kv_cache(
    model: GPTNeoForCausalLM,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    device: torch.device,
    runs: int = BENCHMARK_RUNS,
) -> None:
    """Compare generation time with and without the KV cache."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    for use_cache in (True, False):
        times = []
        for _ in range(runs):
            start = time.time()
            model.generate(
                input_ids,
                do_sample=True,
                temperature=TEMPERATURE,
                max_length=MAX_LENGTH,
                pad_token_id=PAD_TOKEN_ID,
                use_cache=use_cache,
            )
            times.append(time.time() - start)
        label = "Using" if use_cache else "No"
        print(f"{label} KV cache: {round(np.mean(times), 3)} +- {round(np.std(times), 3)} seconds")


def benchmark_total_generation(
    model: GPTNeoForCausalLM,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    device: torch.device,
    max_length: int = MAX_LENGTH_LONG,
    runs: int = BENCHMARK_RUNS_TOTAL,
) -> None:
    """Benchmark total generation time, discarding the first (warm-up) run."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    times = []
    for _ in range(runs):
        start = time.time()
        model.generate(
            input_ids,
            do_sample=True,
            temperature=TEMPERATURE,
            max_length=max_length,
            pad_token_id=PAD_TOKEN_ID,
        )
        times.append(time.time() - start)
    print(
        f"Average Total Generation time: "
        f"{round(np.mean(times[1:]), 3)} +- {round(np.std(times[1:]), 3)} seconds"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Orchestrate all GPT-Neo inference and benchmarking demonstrations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model(MODEL_ID, device)

    # Verify where each model layer was loaded (GPU, RAM, or disk)
    print("HF device map:", model.hf_device_map)

    # Standard text completion
    print(generate_text(model, tokenizer, COMPLETION_PROMPT, device))

    # Few-shot sentiment classification
    print(generate_text(model, tokenizer, FEW_SHOT_PROMPT, device))

    # Python code generation
    print(generate_text(model, tokenizer, CODE_GEN_PROMPT, device))

    # Batch text completion
    batch_outputs = run_batch_completion(model, tokenizer, BATCH_TEXTS, device)
    for text in batch_outputs:
        print("---------")
        print(text)

    # Benchmarking: KV cache vs. no KV cache
    benchmark_kv_cache(model, tokenizer, COMPLETION_PROMPT, device)

    # Benchmarking: total generation time
    benchmark_total_generation(model, tokenizer, COMPLETION_PROMPT, device)


if __name__ == "__main__":
    main()
