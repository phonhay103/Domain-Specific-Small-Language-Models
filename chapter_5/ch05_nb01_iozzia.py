"""ONNX Conversion of the BERT Base Uncased Model.

This script is a companion of chapter 5 of the "Domain Specific LLMs in Action"
book, author Guglielmo Iozzia, Manning Publications, 2024.
The code introduces readers to the ONNX format and ONNX Runtime with the
BERT Base Uncased model. It can be executed with hardware acceleration (GPU).
More details about the code can be found in the related book's chapter.

# Install the missing requirements before running:
#   pip install onnx onnxruntime datasets
"""

import os
import time

import numpy
import onnxruntime
import torch
from datasets import load_dataset
from onnx.checker import check_model
from onnxruntime.transformers import optimizer
from transformers import AutoModelForQuestionAnswering, BertTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "google-bert/bert-base-uncased"
SQUAD_SPLIT = "validation"
SAMPLES_COUNT = 200
MAX_SEQ_LENGTH = 128
BERT_NUM_HEADS = 12
BERT_HIDDEN_SIZE = 768
ONNX_OPSET_VERSION = 15
CORRECTNESS_SAMPLE_RANGE = 2

OUTPUT_DIR = os.path.join(".", "onnx_models")
EXPORT_MODEL_PATH = os.path.join(OUTPUT_DIR, "bert-base-uncased.onnx")
OPTIMIZED_MODEL_PATH = os.path.join(OUTPUT_DIR, "bert-base-uncased.onnx_opt_cpu.onnx")


# ---------------------------------------------------------------------------
# Model and data loading
# ---------------------------------------------------------------------------
def load_model(model_id: str) -> tuple[AutoModelForQuestionAnswering, BertTokenizer]:
    """Download and load BERT Base Uncased QA model and tokenizer from HF Hub."""
    tokenizer = BertTokenizer.from_pretrained(model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    model.eval()
    return model, tokenizer


def load_squad_subset(samples_count: int):
    """Download the first *samples_count* examples from the SQuAD validation set."""
    return load_dataset("squad", split=f"{SQUAD_SPLIT}[:{samples_count}]")


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------
def benchmark_pytorch(
    model: AutoModelForQuestionAnswering,
    tokenizer: BertTokenizer,
    squad,
    samples_count: int,
) -> None:
    """Measure and print average PyTorch (CPU) inference latency over the dataset."""
    latency: list[float] = []
    with torch.no_grad():
        for i in range(samples_count):
            inputs = tokenizer(
                squad["question"][i], squad["context"][i], return_tensors="pt"
            )
            start = time.time()
            model(**inputs)
            latency.append(time.time() - start)
    avg_ms = format(sum(latency) * 1000 / len(latency), ".2f")
    print(f"PyTorch CPU Average inference time = {avg_ms} ms")


def benchmark_onnx(
    session: onnxruntime.InferenceSession,
    tokenizer: BertTokenizer,
    squad,
    samples_count: int,
    label: str = "OnnxRuntime cpu",
) -> list[float]:
    """Measure and print average ONNX Runtime inference latency.

    Returns the raw latency list (in seconds) for further analysis.
    """
    latency: list[float] = []
    ort_outputs = None
    for i in range(samples_count):
        full_inputs = tokenizer(
            squad["question"][i], squad["context"][i], return_tensors="np"
        )
        ort_inputs = {
            "input_ids": full_inputs["input_ids"],
            "input_mask": full_inputs["attention_mask"],
            "segment_ids": full_inputs["token_type_ids"],
        }
        start = time.time()
        ort_outputs = session.run(None, ort_inputs)
        latency.append(time.time() - start)
    avg_ms = format(sum(latency) * 1000 / len(latency), ".2f")
    print(f"{label} Average inference time = {avg_ms} ms")
    return ort_outputs


# ---------------------------------------------------------------------------
# ONNX export and optimisation
# ---------------------------------------------------------------------------
def export_to_onnx(
    model: AutoModelForQuestionAnswering,
    tokenizer: BertTokenizer,
    squad,
    export_path: str,
) -> None:
    """Export the PyTorch model to ONNX format using a sample SQuAD input."""
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    tokenized_inputs = tokenizer(
        squad["question"][0], squad["context"][0], return_tensors="pt"
    )
    inputs = {
        "input_ids": tokenized_inputs["input_ids"],
        "input_mask": tokenized_inputs["attention_mask"],
        "segment_ids": tokenized_inputs["token_type_ids"],
    }

    symbolic_names = {0: "batch_size", 1: "max_seq_len"}
    with torch.no_grad():
        torch.onnx.export(
            model,
            args=tuple(inputs.values()),
            f=export_path,
            opset_version=ONNX_OPSET_VERSION,
            do_constant_folding=True,
            input_names=["input_ids", "input_mask", "segment_ids"],
            output_names=["start", "end"],
            dynamic_axes={
                "input_ids": symbolic_names,
                "input_mask": symbolic_names,
                "segment_ids": symbolic_names,
                "start": symbolic_names,
                "end": symbolic_names,
            },
        )
    print("Model exported at", export_path)


def create_onnx_session(
    export_model_path: str,
    optimized_filepath: str,
    providers: list[str] | None = None,
) -> onnxruntime.InferenceSession:
    """Create an ONNX Runtime inference session with optional graph optimisation."""
    if providers is None:
        providers = ["CPUExecutionProvider"]
    sess_options = onnxruntime.SessionOptions()
    sess_options.optimized_model_filepath = optimized_filepath
    return onnxruntime.InferenceSession(export_model_path, sess_options, providers=providers)


def optimise_onnx_model(export_path: str, optimized_path: str) -> None:
    """Apply ONNX Runtime transformer optimisations and save to *optimized_path*."""
    opt_model = optimizer.optimize_model(
        export_path,
        model_type="bert",
        num_heads=BERT_NUM_HEADS,
        hidden_size=BERT_HIDDEN_SIZE,
    )
    opt_model.save_model_to_file(optimized_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Orchestrate BERT ONNX export, validation, and benchmarking."""
    model, tokenizer = load_model(MODEL_ID)
    squad = load_squad_subset(SAMPLES_COUNT)

    # Display one sample to confirm the dataset loaded correctly
    print(squad[0])

    # Benchmark the original PyTorch model
    benchmark_pytorch(model, tokenizer, squad, SAMPLES_COUNT)

    # Export to ONNX
    export_to_onnx(model, tokenizer, squad, EXPORT_MODEL_PATH)

    # Validate the exported model graph
    check_model(EXPORT_MODEL_PATH, full_check=True)

    # Benchmark the exported ONNX model on CPU
    session = create_onnx_session(
        EXPORT_MODEL_PATH,
        optimized_filepath=os.path.join(OUTPUT_DIR, "bert-base-uncased.onnx"),
    )
    ort_outputs = benchmark_onnx(session, tokenizer, squad, SAMPLES_COUNT)

    # Verify correctness: compare PyTorch and ONNX Runtime outputs numerically
    print("***** Verifying correctness *****")
    with torch.no_grad():
        sample_inputs = tokenizer(
            squad["question"][0], squad["context"][0], return_tensors="pt"
        )
        torch_outputs = model(**sample_inputs)
    for i in range(CORRECTNESS_SAMPLE_RANGE):
        match = numpy.allclose(ort_outputs[i], torch_outputs[i].cpu(), rtol=1e-05, atol=1e-04)
        print(f"PyTorch and ONNX Runtime output {i} are close: {match}")

    # Optimise the ONNX model and benchmark again
    optimise_onnx_model(EXPORT_MODEL_PATH, OPTIMIZED_MODEL_PATH)

    session_opt = create_onnx_session(
        EXPORT_MODEL_PATH,
        optimized_filepath=os.path.join(OUTPUT_DIR, "bert-base-uncased.onnx_opt_cpu.onnx"),
    )
    benchmark_onnx(session_opt, tokenizer, squad, SAMPLES_COUNT, label="OnnxRuntime cpu (optimised)")


if __name__ == "__main__":
    main()
