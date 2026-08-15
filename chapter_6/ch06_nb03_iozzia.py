"""Quantization of a Finetuned BERT Model with HF's Optimum.

This script is a companion of chapter 6 of the "Domain Specific LLMs in Action"
book, author Guglielmo Iozzia, Manning Publications, 2024.
It introduces readers to the quantization of an encoder-only language model,
distilbert-base-uncased-finetuned-banking77, using the Hugging Face Optimum
library. It doesn't require hardware acceleration.
More details about the code can be found in the related book's chapter.

# Install notes (run once in your environment):
#   pip install optimum[onnxruntime] evaluate
#   pip install --force-reinstall datasets==3.6.0
"""

# --- stdlib ---
import os
from pathlib import Path
from time import perf_counter

# --- third-party ---
import numpy as np
from datasets import load_dataset
from evaluate import evaluator
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer, pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "optimum/distilbert-base-uncased-finetuned-banking77"
DATASET_ID = "PolyAI/banking77"
ONNX_PATH = Path("onnx")
ORIGINAL_MODEL_NAME = "model.onnx"
QUANTIZED_MODEL_NAME = "model_quantized.onnx"
BENCHMARK_PROMPT = (
    "Dear Sir/Madam, my name is William. I am getting in touch because I didn't "
    "get a response from you yet. What actions do I need to do to get my new card "
    "which I have requested 3 weeks ago? Please help me and answer this email as "
    "soon as possible. Have a nice rest of the day. Best Regards."
) * 2
VANILLA_ACCURACY_BASELINE = 0.925  # known fp32 accuracy: 92.5 %
WARMUP_RUNS = 10
BENCHMARK_RUNS = 300


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def load_and_export_model(model_id: str, onnx_path: Path) -> tuple:
    """Load the finetuned model from HF Hub, convert to ONNX fp32, and save to disk."""
    model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.save_pretrained(onnx_path)
    tokenizer.save_pretrained(onnx_path)
    return model, tokenizer


def build_vanilla_pipeline(model, tokenizer) -> pipeline:
    """Build a text-classification pipeline with the fp32 ONNX model."""
    return pipeline("text-classification", model=model, tokenizer=tokenizer)


def quantize_model(model, onnx_path: Path) -> Path:
    """Apply dynamic AVX512-VNNI quantization and return the quantized model path."""
    dynamic_quantizer = ORTQuantizer.from_pretrained(model)
    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    model_quantized_path = dynamic_quantizer.quantize(
        save_dir=onnx_path,
        quantization_config=dqconfig,
    )
    return model_quantized_path


def load_quantized_pipeline(onnx_path: Path, quantized_model_name: str, tokenizer) -> pipeline:
    """Load the quantized ONNX model and return a text-classification pipeline."""
    q_model = ORTModelForSequenceClassification.from_pretrained(
        onnx_path, file_name=quantized_model_name
    )
    tokenizer_q = AutoTokenizer.from_pretrained(onnx_path)
    return pipeline("text-classification", model=q_model, tokenizer=tokenizer_q), q_model


def compare_model_sizes(onnx_path: Path, original_name: str, quantized_name: str) -> None:
    """Print file sizes (MB) for the original and quantized ONNX models."""
    size = os.path.getsize(onnx_path / original_name) / (1024 * 1024)
    quantized_size = os.path.getsize(onnx_path / quantized_name) / (1024 * 1024)
    print(f"Original Model file size: {size:.2f} MB")
    print(f"Quantized Model file size: {quantized_size:.2f} MB")


def evaluate_model(clf_pipeline, dataset_id: str, label_mapping: dict) -> dict:
    """Evaluate a classification pipeline on the Banking77 test set and return results."""
    eval = evaluator("text-classification")
    eval_dataset = load_dataset(dataset_id, split="test")
    results = eval.compute(
        model_or_pipeline=clf_pipeline,
        data=eval_dataset,
        metric="accuracy",
        input_column="text",
        label_column="label",
        label_mapping=label_mapping,
        strategy="simple",
    )
    return results


def measure_latency(payload_prompt: str, pipe) -> tuple:
    """Benchmark pipeline latency over many runs and return a summary string and P95 (ms)."""
    latencies = []
    # Warm up
    for _ in range(WARMUP_RUNS):
        _ = pipe(payload_prompt)
    # Effective runs
    for _ in range(BENCHMARK_RUNS):
        start_time = perf_counter()
        _ = pipe(payload_prompt)
        latencies.append(perf_counter() - start_time)

    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies, 95)

    summary = (
        f"P95 latency (ms) - {time_p95_ms}; "
        f"Average latency (ms) - {time_avg_ms:.2f} +\\- {time_std_ms:.2f};"
    )
    return summary, time_p95_ms


def main() -> None:
    """Orchestrate model loading, quantization, evaluation, and benchmarking."""
    # Load fp32 ONNX model and tokenizer
    model, tokenizer = load_and_export_model(MODEL_ID, ONNX_PATH)

    # Verify vanilla model works
    vanilla_clf = build_vanilla_pipeline(model, tokenizer)
    print(vanilla_clf("Could you assist me in checking my card validity?"))

    # Quantize the model
    quantize_model(model, ONNX_PATH)

    # Compare sizes
    compare_model_sizes(ONNX_PATH, ORIGINAL_MODEL_NAME, QUANTIZED_MODEL_NAME)

    # Load quantized pipeline
    q8_clf, q_model = load_quantized_pipeline(ONNX_PATH, QUANTIZED_MODEL_NAME, tokenizer)
    print(q8_clf("Could you assist me in checking my card validity?"))

    # Evaluate quantized model accuracy
    results = evaluate_model(q8_clf, DATASET_ID, q_model.config.label2id)
    print(results)

    # Compare accuracy
    print(f"Vanilla model: {VANILLA_ACCURACY_BASELINE * 100:.1f}%")
    print(f"Quantized model: {results['accuracy'] * 100:.2f}%")
    print(
        f"The quantized model achieves "
        f"{round(results['accuracy'] / VANILLA_ACCURACY_BASELINE, 4) * 100:.2f}% "
        f"accuracy of the fp32 model"
    )

    # Benchmark prompt token count
    print(f"Prompt length: {len(tokenizer(BENCHMARK_PROMPT)['input_ids'])}")

    # Benchmark latency
    original_stats = measure_latency(BENCHMARK_PROMPT, vanilla_clf)
    quantized_stats = measure_latency(BENCHMARK_PROMPT, q8_clf)

    print(f"Vanilla model: {original_stats[0]}")
    print(f"Quantized model: {quantized_stats[0]}")
    print(f"Improvement through quantization: {round(original_stats[1] / quantized_stats[1], 2)}x")


if __name__ == "__main__":
    main()
