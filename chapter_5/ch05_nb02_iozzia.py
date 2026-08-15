"""ONNX Conversion of the GPT-2 Small Model.

Companion script for Chapter 5 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2025).

Introduces the ONNX format and ONNX Runtime on GPU with the GPT-2 Small model.
Requires hardware acceleration (GPU).

NOTE: This script is not compatible with PyTorch 2.1+ or the corresponding
HF Transformers releases. Downgrade first:
    pip install torch==2.0.1 transformers==4.31.0
    pip install onnx onnxruntime-gpu
"""

# Standard library
import os
import time

# Third-party
import numpy  # noqa: F401  (required by onnxruntime internals)
import onnxruntime
import torch
from onnxruntime.transformers import optimizer
from transformers import AutoModelForCausalLM, GPT2Tokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "openai-community/gpt2"
SAMPLE_PROMPT = "The story so far: in the beginning, the universe was created."
ONNX_OUTPUT_DIR = os.path.join(".", "onnx_models")
ONNX_MODEL_PATH = os.path.join(ONNX_OUTPUT_DIR, "gpt-2.onnx")
ONNX_OPT_MODEL_PATH = os.path.join(ONNX_OUTPUT_DIR, "gpt-2-onnx_opt_gpu.onnx")

# GPT-2 Small architecture hyper-parameters used during ONNX optimisation
GPT2_NUM_HEADS = 12
GPT2_HIDDEN_SIZE = 768
ONNX_OPSET_VERSION = 15

# Inference benchmarking settings
WARMUP_RUNS = 2
BENCHMARK_RUNS = 10
MAX_BENCHMARK_LENGTH = 256
MAX_PREVIEW_LENGTH = 64
PAD_TOKEN_ID = 50256


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_id: str, device: torch.device):
    """Download (or load from cache) GPT-2 and move it to *device*."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval().to(device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_model(model, tokenizer, prompt: str, device: torch.device) -> None:
    """Run a single forward pass and print input/output shapes."""
    inputs = tokenizer(prompt, return_attention_mask=False, return_tensors="pt")
    inputs = inputs.to(device)
    print("input tensors")
    print(inputs)
    print("input tensor shape")
    print(inputs["input_ids"].size())

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    print("output tensor")
    print(logits)
    print("output shape")
    print(logits.shape)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_to_onnx(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    output_dir: str,
    export_path: str,
) -> None:
    """Convert *model* to ONNX and save it to *export_path*."""
    os.makedirs(output_dir, exist_ok=True)

    tokenized = tokenizer(prompt, return_attention_mask=False, return_tensors="pt")
    tokenized.to(device)
    inputs_sample = {"input_ids": tokenized["input_ids"]}

    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs_sample,
            export_path,
            export_params=True,
            opset_version=ONNX_OPSET_VERSION,
            do_constant_folding=True,
            input_names=["input_ids"],
        )
    print(f"ONNX model exported to: {export_path}")


# ---------------------------------------------------------------------------
# ONNX optimisation
# ---------------------------------------------------------------------------
def optimize_onnx_model(export_path: str, optimized_path: str) -> None:
    """Apply GPU-targeted transformer optimisations and save the result."""
    optimized_model = optimizer.optimize_model(
        export_path,
        model_type="gpt2",
        use_gpu=True,
        num_heads=GPT2_NUM_HEADS,
        hidden_size=GPT2_HIDDEN_SIZE,
        verbose=True,
    )
    optimized_model.save_model_to_file(optimized_path)
    print(f"Optimised ONNX model saved to: {optimized_path}")


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------
def benchmark_pytorch(model, inputs, tokenizer, device: torch.device) -> None:
    """Warm up then time *BENCHMARK_RUNS* PyTorch inference calls."""
    with torch.inference_mode():
        # Preview generation
        sample_output = model.generate(
            inputs.input_ids, max_length=MAX_PREVIEW_LENGTH, pad_token_id=PAD_TOKEN_ID
        )
        print(tokenizer.decode(sample_output[0], skip_special_tokens=False))

        # Warm-up
        for _ in range(WARMUP_RUNS):
            model.generate(
                inputs.input_ids, max_length=MAX_BENCHMARK_LENGTH, pad_token_id=PAD_TOKEN_ID
            )
            torch.cuda.synchronize()

        # Timed runs
        start = time.time()
        for _ in range(BENCHMARK_RUNS):
            model.generate(
                inputs.input_ids, max_length=MAX_BENCHMARK_LENGTH, pad_token_id=PAD_TOKEN_ID
            )
            torch.cuda.synchronize()

    elapsed = (time.time() - start) / BENCHMARK_RUNS
    print(f"----\nPyTorch: {elapsed:.2f}s/sequence")


def benchmark_onnx_session(
    session: onnxruntime.InferenceSession,
    ort_inputs: dict,
    label: str,
) -> None:
    """Warm up then time *BENCHMARK_RUNS* ORT inference calls."""
    # Warm-up
    for _ in range(WARMUP_RUNS):
        session.run(None, ort_inputs)

    start = time.time()
    for _ in range(BENCHMARK_RUNS):
        session.run(None, ort_inputs)

    elapsed = (time.time() - start) / BENCHMARK_RUNS
    print(f"----\n{label}: {elapsed:.2f}s/sequence")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    """Run end-to-end ONNX conversion and benchmarking for GPT-2 Small."""
    device = torch.device("cuda")

    # Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_ID, device)

    # Verify the downloaded model works
    verify_model(model, tokenizer, SAMPLE_PROMPT, device)

    # Export to ONNX
    export_to_onnx(model, tokenizer, SAMPLE_PROMPT, device, ONNX_OUTPUT_DIR, ONNX_MODEL_PATH)

    # Optimise the exported ONNX model
    optimize_onnx_model(ONNX_MODEL_PATH, ONNX_OPT_MODEL_PATH)

    # Prepare tokenised inputs for benchmarking (PyTorch)
    pt_inputs = tokenizer(SAMPLE_PROMPT, return_attention_mask=False, return_tensors="pt")
    pt_inputs = pt_inputs.to(device)

    # Benchmark: original PyTorch model
    benchmark_pytorch(model, pt_inputs, tokenizer, device)
    model.cpu()  # Free GPU memory before ORT sessions

    # Prepare tokenised inputs for ORT (numpy tensors)
    np_input_ids = tokenizer(SAMPLE_PROMPT, return_attention_mask=False, return_tensors="np")
    ort_inputs = {"input_ids": np_input_ids["input_ids"]}

    # Benchmark: plain ONNX model
    session = onnxruntime.InferenceSession(
        ONNX_MODEL_PATH, providers=["CUDAExecutionProvider"]
    )
    benchmark_onnx_session(session, ort_inputs, label="ONNX")

    # Benchmark: optimised ONNX model
    opt_session = onnxruntime.InferenceSession(
        ONNX_OPT_MODEL_PATH, providers=["CUDAExecutionProvider"]
    )
    benchmark_onnx_session(opt_session, ort_inputs, label="ONNX-Optimised")


if __name__ == "__main__":
    main()
