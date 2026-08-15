"""Accelerating inference for GPT-Neo with DeepSpeed.

This script is a companion of chapter 4 of the "Domain Specific LLMs in Action"
book, author Guglielmo Iozzia, Manning Publications, 2024.
The code introduces readers to the DeepSpeed library to accelerate inference for
the GPT-Neo model for text generation tasks. It can be executed with hardware
acceleration (GPU).
More details about the code can be found in the book's chapter.

# Install the missing dependencies before running (DeepSpeed and HF's Accelerate):
#   pip install deepspeed accelerate
"""

import os
from time import perf_counter

import deepspeed
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "EleutherAI/gpt-neo-2.7B"
PAD_TOKEN_ID = 50256
EXAMPLE_PROMPT = "The story so far: in the beginning, the universe was created."

# DeepSpeed distributed environment (single-GPU setup)
DS_MASTER_ADDR = "localhost"
DS_MASTER_PORT = "9999"
DS_RANK = "0"
DS_LOCAL_RANK = "0"
DS_WORLD_SIZE = "1"

GENERATION_MAX_LENGTH = 300
LATENCY_WARMUP_RUNS = 2
LATENCY_MEASURE_RUNS = 20


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------
def measure_latency(
    model,
    tokenizer: AutoTokenizer,
    payload: str,
    device,
    generation_args: dict | None = None,
) -> tuple[str, float]:
    """Measure generation latency with GPU warm-up.

    Returns a summary string and the P95 latency in milliseconds.
    """
    if generation_args is None:
        generation_args = {}
    input_ids = tokenizer(payload, return_tensors="pt").input_ids.to(device)
    latencies: list[float] = []

    # GPU warm-up before benchmarking
    for _ in range(LATENCY_WARMUP_RUNS):
        _ = model.generate(input_ids, **generation_args)

    # Runs used for measuring the latency
    for _ in range(LATENCY_MEASURE_RUNS):
        start_time = perf_counter()
        _ = model.generate(input_ids, **generation_args)
        latencies.append(perf_counter() - start_time)

    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies, 95)

    summary = (
        f"P95 latency (ms) - {time_p95_ms}; "
        f"Average latency (ms) - {time_avg_ms:.2f} +\\- {time_std_ms:.2f};"
    )
    return summary, time_p95_ms


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_base_model(model_id: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Download base GPT-Neo 2.7B in half-precision from the HF Hub."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"model is loaded on device {model.device.type}")
    return model, tokenizer


def init_deepspeed_model(model) -> deepspeed.InferenceEngine:
    """Optimize the model for GPU inference using DeepSpeed."""
    # Configure distributed environment for single-GPU DeepSpeed inference
    os.environ["MASTER_ADDR"] = DS_MASTER_ADDR
    os.environ["MASTER_PORT"] = DS_MASTER_PORT
    os.environ["RANK"] = DS_RANK
    os.environ["LOCAL_RANK"] = DS_LOCAL_RANK
    os.environ["WORLD_SIZE"] = DS_WORLD_SIZE

    ds_model = deepspeed.init_inference(
        model=model,
        mp_size=1,
        dtype=torch.float16,
        replace_method="auto",
        replace_with_kernel_inject=True,
    )
    print(f"model is loaded on device {ds_model.module.device}")
    return ds_model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def run_inference(model, tokenizer: AutoTokenizer, prompt: str, device) -> str:
    """Run a single generation pass and return the decoded prediction text."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    logits = model.generate(
        input_ids,
        do_sample=True,
        num_beams=1,
        min_length=128,
        max_new_tokens=128,
        pad_token_id=PAD_TOKEN_ID,
    )
    return tokenizer.decode(logits[0].tolist()[len(input_ids[0]):])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Orchestrate vanilla vs. DeepSpeed-optimised GPT-Neo benchmarking."""
    model, tokenizer = load_base_model(MODEL_ID)

    # Verify that the vanilla model works correctly
    print(f"prediction:\n\n{run_inference(model, tokenizer, EXAMPLE_PROMPT, model.device)}")

    # Benchmark the vanilla (unoptimised) model
    generation_args = dict(
        do_sample=True,
        max_length=GENERATION_MAX_LENGTH,
        pad_token_id=PAD_TOKEN_ID,
        use_cache=True,
    )
    vanilla_results = measure_latency(
        model, tokenizer, EXAMPLE_PROMPT, model.device, generation_args
    )
    print(f"Vanilla model: {vanilla_results[0]}")

    # Optimise with DeepSpeed; inspect the replaced layer architecture
    ds_model = init_deepspeed_model(model)
    print(ds_model)  # Shows which layers DeepSpeed replaced with optimised kernels

    # Verify that the DeepSpeed-optimised model works correctly
    input_ids = tokenizer(EXAMPLE_PROMPT, return_tensors="pt").input_ids.to(model.device)
    logits = ds_model.generate(
        input_ids,
        do_sample=True,
        num_beams=1,
        min_length=128,
        max_new_tokens=128,
        pad_token_id=PAD_TOKEN_ID,
        use_cache=False,
    )
    print(tokenizer.decode(logits[0].tolist()))

    # Benchmark the DeepSpeed-optimised model
    generation_args = dict(
        do_sample=True,
        max_length=GENERATION_MAX_LENGTH,
        pad_token_id=PAD_TOKEN_ID,
        use_cache=True,
    )
    ds_results = measure_latency(
        ds_model, tokenizer, EXAMPLE_PROMPT, ds_model.module.device, generation_args
    )
    print(f"DeepSpeed model: {ds_results[0]}")


if __name__ == "__main__":
    main()
