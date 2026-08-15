"""Benchmarking Python Code Generation with Vanilla, ONNX Converted and Quantized CodeGen Models.

This script is a companion of chapter 7 of the "Domain Specific LLMs in Action"
book, author Guglielmo Iozzia, Manning Publications, 2024.
It benchmarks inference performance (latency and throughput) when generating
Python code using a Vanilla CodeGen 350M mono model, after ONNX conversion of
the same model, and after 8-bit quantization. It doesn't require hardware
acceleration.
More details about the code can be found in the related book's chapter.

# Install notes (run once in your environment):
#   pip install optimum[onnxruntime]==1.21.2
#   pip install -U transformers
"""

# --- stdlib ---
import gc
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Generator

# --- third-party ---
import numpy as np
import pandas as pd
import plotly.express as px
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer, CodeGenForCausalLM, pipeline
from tqdm import trange

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "Salesforce/codegen-350M-mono"
LOCAL_CHECKPOINT_DIR = "local-pt-checkpoint"
ONNX_PATH = Path("onnx")
QUANTIZED_MODEL_FILE = "model_quantized.onnx"
DEVICE = "cpu"
PAD_TOKEN_ID = 50256
MAX_GEN_LENGTH = 12
BENCHMARK_PROMPT = "def hello_world():"
WARMUP_RUNS = 10
BENCHMARK_RUNS = 100

# Providers for each benchmark phase
PYTORCH_PROVIDERS = {("cpu", "PyTorch CPU")}
ONNX_PROVIDERS = {("CPUExecutionProvider", "ONNX CPU")}
ONNX_QUANT_PROVIDERS = {("CPUExecutionProvider", "ONNX Quant CPU")}

# Performance table column labels
PERF_INDEX_LABELS = [
    "Average_latency (ms)", "Latency_P50", "Latency_P75",
    "Latency_P90", "Latency_P95", "Latency_P99", "Throughput",
]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkInferenceResult:
    """Stores raw inference times and optional optimized model path for one run."""

    model_inference_time: list
    optimized_model_path: str


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

@contextmanager
def track_infer_time(time_buffer: list) -> Generator:
    """Context manager that appends elapsed wall-clock time (s) to *time_buffer*."""
    start_time = perf_counter()
    yield
    time_buffer.append(perf_counter() - start_time)


def benchmark_inference(
    providers_dict: set,
    pipe,
    prompt: str,
    results: dict,
) -> dict:
    """Warm up then benchmark *pipe* for each provider label; store results in *results*."""
    for device, label in providers_dict:
        for _ in trange(WARMUP_RUNS, desc="Warming up"):
            pipe(prompt)

        time_buffer = []
        for _ in trange(BENCHMARK_RUNS, desc=f"Tracking inference time ({label})"):
            with track_infer_time(time_buffer):
                pipe(prompt)

        results[label] = BenchmarkInferenceResult(time_buffer, None)

    return results


def make_pipeline(model, tokenizer) -> pipeline:
    """Build a text-generation pipeline with shared settings."""
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=PAD_TOKEN_ID,
        truncation=True,
        max_length=MAX_GEN_LENGTH,
    )


# ---------------------------------------------------------------------------
# Benchmark phases
# ---------------------------------------------------------------------------

def run_vanilla_benchmark(tokenizer, results: dict) -> tuple:
    """Download CodeGen, verify generation, save checkpoint, and run PyTorch benchmark."""
    model = CodeGenForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()

    # Verify the model generates correctly before benchmarking
    input_ids = tokenizer(BENCHMARK_PROMPT, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=MAX_GEN_LENGTH)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True, pad_token_id=PAD_TOKEN_ID))

    pipe = make_pipeline(model, tokenizer)
    result = pipe(BENCHMARK_PROMPT)
    print(result[0]["generated_text"])

    # Save checkpoint for later ONNX conversion reference
    tokenizer.save_pretrained(LOCAL_CHECKPOINT_DIR)
    model.save_pretrained(LOCAL_CHECKPOINT_DIR)

    results = benchmark_inference(PYTORCH_PROVIDERS, pipe, BENCHMARK_PROMPT, results)

    # Free memory before ONNX conversion
    del model
    gc.collect()

    return pipe, results


def run_onnx_benchmark(tokenizer, results: dict) -> tuple:
    """Convert CodeGen to ONNX, save to disk, and run ONNX CPU benchmark."""
    model = ORTModelForCausalLM.from_pretrained(
        MODEL_ID, export=True, provider="CPUExecutionProvider"
    )
    model.save_pretrained(ONNX_PATH)

    pipe = make_pipeline(model, tokenizer)
    result = pipe(BENCHMARK_PROMPT)
    print(result)

    results = benchmark_inference(ONNX_PROVIDERS, pipe, BENCHMARK_PROMPT, results)

    return model, pipe, results


def run_quantized_benchmark(onnx_model, tokenizer, results: dict) -> dict:
    """Dynamically quantize the ONNX model and run the quantized benchmark."""
    dynamic_quantizer = ORTQuantizer.from_pretrained(onnx_model)
    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    dynamic_quantizer.quantize(save_dir=ONNX_PATH, quantization_config=dqconfig)

    quantized_model = ORTModelForCausalLM.from_pretrained(
        str(ONNX_PATH), file_name=QUANTIZED_MODEL_FILE
    )
    pipe = make_pipeline(quantized_model, tokenizer)
    result = pipe(BENCHMARK_PROMPT)
    print(result)

    results = benchmark_inference(ONNX_QUANT_PROVIDERS, pipe, BENCHMARK_PROMPT, results)
    return results


# ---------------------------------------------------------------------------
# Results reporting
# ---------------------------------------------------------------------------

def plot_average_inference_times(results: dict) -> None:
    """Bar chart of average inference time (ms) across providers."""
    time_results = {k: np.mean(v.model_inference_time) * 1e3 for k, v in results.items()}
    fig = px.bar(
        x=list(time_results.keys()),
        y=list(time_results.values()),
        title="Average inference time (ms) for each provider",
        labels={"x": "Provider", "y": "Avg Inference time (ms)"},
        text_auto=".2s",
    )
    fig.show()


def build_perf_dataframe(results: dict) -> pd.DataFrame:
    """Compute latency percentiles and throughput and return as a DataFrame."""
    perf_results = {}
    for k, v in results.items():
        latency_list = v.model_inference_time
        average_latency = np.mean(latency_list) * 1e3
        throughput = 1 * (1000 / average_latency)
        perf_results[k] = (
            average_latency,
            np.percentile(latency_list, 50) * 1e3,
            np.percentile(latency_list, 75) * 1e3,
            np.percentile(latency_list, 90) * 1e3,
            np.percentile(latency_list, 95) * 1e3,
            np.percentile(latency_list, 99) * 1e3,
            throughput,
        )
    return pd.DataFrame(data=perf_results, index=PERF_INDEX_LABELS)


def plot_inference_duration_box(results: dict) -> None:
    """Box-plot of inference durations across all runs for each provider."""
    results_df = pd.DataFrame(columns=["Provider", "Inference_time"])
    for k, v in results.items():
        for t in v.model_inference_time:
            results_df.loc[len(results_df.index)] = [k, t * 1e3]

    fig = px.box(
        results_df,
        x="Provider",
        y="Inference_time",
        points="all",
        labels={"Provider": "Provider", "Inference_time": "Inference durations (ms)"},
    )
    fig.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate vanilla → ONNX → quantized benchmarks and display results."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    results: dict = {}

    # Phase 1: Vanilla PyTorch model
    _vanilla_pipe, results = run_vanilla_benchmark(tokenizer, results)

    # Phase 2: ONNX converted model
    onnx_model, onnx_pipe, results = run_onnx_benchmark(tokenizer, results)

    # Free the ONNX pipeline before quantization to avoid OOM
    del onnx_pipe
    gc.collect()

    # Phase 3: 8-bit quantized ONNX model
    results = run_quantized_benchmark(onnx_model, tokenizer, results)

    # Visualise and report
    plot_average_inference_times(results)

    perf_df = build_perf_dataframe(results)
    print(perf_df)

    plot_inference_duration_box(results)


if __name__ == "__main__":
    main()
