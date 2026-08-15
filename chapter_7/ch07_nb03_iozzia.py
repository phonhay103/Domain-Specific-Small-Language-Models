"""
Benchmarking Python Code Generation with Vanilla and 8-bit Quantized StarCoder2 Models.

Companion to Chapter 7 of "Domain Specific LLMs in Action" by Guglielmo Iozzia,
Manning Publications, 2024.

Benchmarks inference performance (latency and throughput) when generating Python code
using a vanilla StarCoder2-3B model and after 8-bit quantization of the same model.
GPU acceleration is required.

Setup instructions (run once before executing this script):
    pip install optimum[onnxruntime-gpu]==1.21.2
    pip install -U bitsandbytes
    pip install -U numpy transformers
"""

import gc
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from tqdm import trange

# Model and checkpoint constants
MODEL_ID = "bigcode/starcoder2-3b"
CHECKPOINT_SAVE_DIR = "local-pt-checkpoint"
CHECKPOINT_8BIT_SAVE_DIR = "local-8bit-checkpoint"

# Benchmark configuration
BENCHMARK_PROMPT = "def print_hello_world():"
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

# Pipeline inference parameters
PIPELINE_TEMPERATURE = 0.2
PIPELINE_TOP_P = 0.95
PIPELINE_MAX_LENGTH = 14

# Provider sets
VANILLA_PROVIDERS = {("gpu", "PyTorch GPU")}
QUANTIZED_PROVIDERS = {("ort", "Quant GPU")}

# Percentile thresholds for latency reporting
LATENCY_PERCENTILES = [50, 75, 90, 95, 99]
PERF_INDEX_LABELS = [
    "Average_latency (ms)",
    "Latency_P50",
    "Latency_P75",
    "Latency_P90",
    "Latency_P95",
    "Latency_P99",
    "Throughput",
]


# ---- Benchmarking utilities ----

@contextmanager
def track_infer_time(time_buffer: List[float]):
    """Context manager that appends elapsed wall-clock time to *time_buffer*."""
    start_time = perf_counter()
    yield
    end_time = perf_counter()
    time_buffer.append(end_time - start_time)


@dataclass
class BenchmarkInferenceResult:
    """Holds raw inference times and an optional path to the optimized model."""
    model_inference_time: List[float]
    optimized_model_path: str


def benchmark_inference(
    providers_dict,
    pipe,
    prompt: str,
    results: Dict[str, BenchmarkInferenceResult],
) -> Dict[str, BenchmarkInferenceResult]:
    """Run warmup + timed inference passes and store results by provider label."""
    for device, label in providers_dict:
        for _ in trange(WARMUP_ITERATIONS, desc="Warming up"):
            pipe(prompt)

        time_buffer: List[float] = []
        for _ in trange(BENCHMARK_ITERATIONS, desc=f"Tracking inference time ({label})"):
            with track_infer_time(time_buffer):
                pipe(prompt)

        results[label] = BenchmarkInferenceResult(time_buffer, None)

    return results


# ---- Model loading helpers ----

def load_vanilla_model(model_id: str, device: str):
    """Download and load the vanilla StarCoder2 model (bfloat16) and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return model, tokenizer


def save_checkpoint(model, tokenizer, save_dir: str) -> None:
    """Persist model and tokenizer weights to *save_dir* for later reuse."""
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)


def free_model_memory(model, pipe) -> None:
    """Move model to CPU, delete references, and clear GPU cache to prevent OOM."""
    model.cpu()
    del model
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def load_quantized_model(checkpoint_dir: str, save_dir: str):
    """Apply 8-bit quantization to a saved checkpoint and persist the result."""
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    quantized_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, quantization_config=quantization_config
    )
    quantized_model.eval()
    quantized_model.save_pretrained(save_dir)
    # Reload from disk to mirror the original notebook flow
    quantized_model_loaded = AutoModelForCausalLM.from_pretrained(save_dir)
    quantized_model_loaded.eval()
    return quantized_model_loaded, tokenizer


def build_pipeline(model, tokenizer):
    """Create a Transformers text-generation pipeline for a given model."""
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        use_cache=True,
        temperature=PIPELINE_TEMPERATURE,
        top_p=PIPELINE_TOP_P,
        max_length=PIPELINE_MAX_LENGTH,
    )


# ---- Results reporting ----

def plot_avg_inference_time(results: Dict[str, BenchmarkInferenceResult]) -> None:
    """Bar chart of average inference time (ms) across providers."""
    time_results = {k: np.mean(v.model_inference_time) * 1e3 for k, v in results.items()}
    fig = px.bar(
        x=time_results.keys(),
        y=time_results.values(),
        title="Average inference time (ms) for each provider",
        labels={"x": "Provider", "y": "Avg Inference time (ms)"},
        text_auto=".2s",
    )
    fig.show()


def compute_perf_dataframe(results: Dict[str, BenchmarkInferenceResult]) -> pd.DataFrame:
    """Build a DataFrame of latency percentiles and throughput across providers."""
    time_results = {k: np.mean(v.model_inference_time) * 1e3 for k, v in results.items()}

    perf_results = {}
    for k, v in results.items():
        latency_list = v.model_inference_time
        pcts = [np.percentile(latency_list, p) * 1e3 for p in LATENCY_PERCENTILES]
        average_latency = np.mean(latency_list) * 1e3
        throughput = 1 * (1000 / average_latency)
        perf_results[k] = (average_latency, *pcts, throughput)

    return pd.DataFrame(data=perf_results, index=PERF_INDEX_LABELS)


def plot_inference_duration_distribution(results: Dict[str, BenchmarkInferenceResult]) -> None:
    """Box plot comparing inference duration distributions across providers."""
    results_df = pd.DataFrame(columns=["Provider", "Inference_time"])
    for k, v in results.items():
        for i in range(len(v.model_inference_time)):
            results_df.loc[len(results_df.index)] = [k, v.model_inference_time[i] * 1e3]

    fig = px.box(
        results_df,
        x="Provider",
        y="Inference_time",
        points="all",
        labels={"Provider": "Provider", "Inference_time": "Inference durations (ms)"},
    )
    fig.show()


# ---- Main orchestration ----

def main() -> None:
    """
    Orchestrate:
      1. Load and benchmark the vanilla StarCoder2-3B model.
      2. Quantize to 8-bit, benchmark again.
      3. Display comparative latency/throughput results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: Dict[str, BenchmarkInferenceResult] = {}

    # --- Vanilla model ---
    model, tokenizer = load_vanilla_model(MODEL_ID, device)

    # Verify model loaded correctly before benchmarking
    inputs = tokenizer.encode(BENCHMARK_PROMPT, return_tensors="pt").to(device)
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))

    pipe = build_pipeline(model, tokenizer)

    # Quick sanity-check of the pipeline
    result = pipe(BENCHMARK_PROMPT)
    print(result[0]["generated_text"])

    # Save checkpoint for later reuse during quantization
    save_checkpoint(model, tokenizer, CHECKPOINT_SAVE_DIR)

    results = benchmark_inference(VANILLA_PROVIDERS, pipe, BENCHMARK_PROMPT, results)

    # --- 8-bit quantized model ---
    # Free GPU/RAM before quantization to avoid OOM
    free_model_memory(model, pipe)

    quantized_model_loaded, tokenizer = load_quantized_model(
        CHECKPOINT_SAVE_DIR, CHECKPOINT_8BIT_SAVE_DIR
    )

    # Verify quantized model works
    inputs = tokenizer.encode(BENCHMARK_PROMPT, return_tensors="pt").to(device)
    outputs = quantized_model_loaded.generate(inputs)
    print(tokenizer.decode(outputs[0]))

    pipe = build_pipeline(quantized_model_loaded, tokenizer)

    # Sanity-check the quantized pipeline
    result = pipe(BENCHMARK_PROMPT)
    print(result[0]["generated_text"])

    results = benchmark_inference(QUANTIZED_PROVIDERS, pipe, BENCHMARK_PROMPT, results)

    # --- Results ---
    plot_avg_inference_time(results)

    perf_df = compute_perf_dataframe(results)
    print(perf_df)

    plot_inference_duration_distribution(results)


if __name__ == "__main__":
    main()
