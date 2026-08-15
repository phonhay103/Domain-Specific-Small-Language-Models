"""
Benchmarking Different Versions of a Small Language Model Before Deployment on an Endpoint.

This script is a companion of chapter 11 of the "Domain Specific LLMs in Action" book,
author Guglielmo Iozzia, Manning Publications, 2024.

The code shows how to benchmark different versions of the GPT-2 small model to assess
which one would be the most performant and the final candidate for deployment on a
FastAPI endpoint. The same code applies to any other Open Source LLM hosted in the
HF Hub by replacing the model id. No hardware acceleration is needed for this model.
Depending on the model under benchmark a GPU would be required.

More details about the code can be found in the related book's chapter.
"""

# stdlib
import gc
import timeit
from copy import deepcopy
from pathlib import Path

# third-party
import numpy as np
import torch
from onnxruntime import InferenceSession
from onnxruntime.transformers.optimizer import optimize_model
from transformers import BatchEncoding, GPT2Model, GPT2Tokenizer

# Install missing dependencies if needed:
# !pip install onnxruntime onnx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = "cpu"
MODEL_ID = "openai-community/gpt2"
MODEL_SAVE_PATH = Path("gpt2")
ONNX_MODEL_PATH = "gpt2_onnx.onnx"
OPTIMIZED_ONNX_PATH = "gpt2_optimized.onnx"
OPTIMIZED_FP16_MODEL_PATH = "optimized_fp16.onnx"
BENCHMARK_PROMPT = "Today is Saturday and"
MAX_SEQUENCE_LENGTH = 1024
ORT_PROVIDERS = ["CPUExecutionProvider"]
BENCHMARK_SEQUENCE_LENGTHS = [1, 4, 64, 256, 512, 1024]
BENCHMARK_WARMUP_RUNS = 10
BENCHMARK_TIMED_RUNS = 100
ONNX_OPSET_VERSION = 18


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def benchmark(f, name: str = "") -> None:
    """Warm up then time *f* over many runs and print average latency in ms."""
    for _ in range(BENCHMARK_WARMUP_RUNS):
        f()
    seconds_per_iter = timeit.timeit(f, number=BENCHMARK_TIMED_RUNS) / BENCHMARK_TIMED_RUNS
    print(f"{name}: {seconds_per_iter * 1000:.3f} ms")


def load_model(model_id: str, device: str) -> tuple[GPT2Tokenizer, GPT2Model]:
    """Load the vanilla GPT-2 small model and tokenizer from the HF Hub."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    model = GPT2Model.from_pretrained(model_id)
    model.eval()
    model.save_pretrained(MODEL_SAVE_PATH)
    return tokenizer, model


def export_to_onnx(model: GPT2Model, tokenizer: GPT2Tokenizer, text: str, path: str) -> None:
    """Convert the vanilla model to ONNX format."""
    input_ids: BatchEncoding = tokenizer(
        text, add_special_tokens=True, return_attention_mask=False, return_tensors="pt"
    )
    for k, v in input_ids.items():
        input_ids[k] = v.type(dtype=torch.int32)
    input_tensor = input_ids["input_ids"]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }
    torch.onnx.export(
        model,
        f=path,
        args=(input_tensor,),
        input_names=["input_ids"],
        output_names=["logits"],
        quantization=False,
        var_output_seq=True,
        do_constant_folding=True,
        opset_version=ONNX_OPSET_VERSION,
        dynamic_axes=dynamic_axes,
    )


def build_ort_inputs(tokenizer: GPT2Tokenizer, text: str) -> dict:
    """Encode *text* into ORT-ready int32 numpy inputs."""
    encodings_dict = tokenizer.batch_encode_plus([text])
    input_ids = torch.tensor(encodings_dict["input_ids"], dtype=torch.int32)
    return {"input_ids": input_ids.cpu().numpy()}


def run_sequence_length_benchmarks(
    tokenizer: GPT2Tokenizer,
    sess: InferenceSession,
    optimized_sess: InferenceSession,
    optimized_fp16_sess: InferenceSession,
) -> None:
    """Run latency benchmarks for each model variant across several token counts."""
    tokenizer.pad_token = tokenizer.eos_token
    for n in BENCHMARK_SEQUENCE_LENGTHS:
        print(f"====== Tokens {n} ======")
        txt = " ".join(["word"] * n)

        ort_inputs = dict(
            tokenizer(
                txt,
                max_length=MAX_SEQUENCE_LENGTH,
                return_tensors="np",
                return_attention_mask=False,
            )
        )
        ort_inputs["input_ids"] = ort_inputs["input_ids"].astype(np.int32)

        benchmark(
            lambda: sess.run(None, {"input_ids": ort_inputs["input_ids"]}),
            f"ONNX ({n} tokens)",
        )
        benchmark(lambda: optimized_sess.run(None, ort_inputs), f"ONNX optimized ({n} tokens)")
        benchmark(
            lambda: optimized_fp16_sess.run(None, ort_inputs),
            f"ONNX optimized fp16 ({n} tokens)",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate loading, exporting, optimising and benchmarking GPT-2."""
    # Load vanilla model
    tokenizer, model = load_model(MODEL_ID, DEVICE)
    num_layer = model.config.n_layer
    num_attention_heads = model.config.n_head
    hidden_size = model.config.n_embd

    # Tokenize prompt and benchmark vanilla PyTorch model
    inputs_base = tokenizer(BENCHMARK_PROMPT, return_tensors="pt").to(DEVICE)
    benchmark(lambda: model(**inputs_base), "PyTorch")

    # Export to ONNX
    export_to_onnx(model, tokenizer, BENCHMARK_PROMPT, ONNX_MODEL_PATH)

    # Free vanilla model; no longer needed
    del model
    gc.collect()

    # Benchmark base ONNX model
    sess = InferenceSession(ONNX_MODEL_PATH, providers=ORT_PROVIDERS)
    ort_inputs = build_ort_inputs(tokenizer, BENCHMARK_PROMPT)
    benchmark(lambda: sess.run(None, ort_inputs), "ONNX")

    del sess
    gc.collect()

    # Optimise the ONNX model
    optimized_model = optimize_model(input=ONNX_MODEL_PATH, model_type="gpt2", use_gpu=False)
    optimized_model.save_model_to_file(OPTIMIZED_ONNX_PATH)

    optimized_sess = InferenceSession(OPTIMIZED_ONNX_PATH, providers=ORT_PROVIDERS)
    benchmark(lambda: optimized_sess.run(None, input_feed=ort_inputs), "ONNX optimized")

    del optimized_sess
    gc.collect()

    # Downsize optimised ONNX model to FP16
    optimized_fp16_model = deepcopy(optimized_model)
    optimized_fp16_model.convert_float_to_float16()
    optimized_fp16_model.save_model_to_file(OPTIMIZED_FP16_MODEL_PATH)

    del optimized_model
    gc.collect()

    optimized_fp16_sess = InferenceSession(OPTIMIZED_FP16_MODEL_PATH, providers=ORT_PROVIDERS)
    benchmark(lambda: optimized_fp16_sess.run(None, input_feed=ort_inputs), "ONNX optimized fp16")

    # Multi-length benchmark across all variants
    sess = InferenceSession(ONNX_MODEL_PATH, providers=ORT_PROVIDERS)
    optimized_sess = InferenceSession(OPTIMIZED_ONNX_PATH, providers=ORT_PROVIDERS)
    run_sequence_length_benchmarks(tokenizer, sess, optimized_sess, optimized_fp16_sess)


if __name__ == "__main__":
    main()
