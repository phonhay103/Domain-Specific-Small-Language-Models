# ==========================================
# Extracted from CH11_NB02_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Benchmarking Different Versions of a Small Language Model Before Deployment on an Endpoint.
# This notebook is a companion of chapter 11 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook shows how to benchmark different versions of the [GPT-2 small](https://huggingface.co/openai-community/gpt2) model to assess which one would be the most performant and the final candidate for deployment on a FastAPI endpoint. The same code applies to any other Open Source LLM hosted in the HF's Hub by replacing the model id. No hardware acceleration is needed for this model. Depending on the model under benchmark a GPU would be required.
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the missing dependencies in the Colab VM (only ONNX and the ONNX runtime).
# ------------------------------------------------------------

# !pip install onnxruntime onnx

# ------------------------------------------------------------
# Load the vanilla GPT-2 small model and related tokenizer from the HF's Hub.
# ------------------------------------------------------------

from pathlib import Path
from transformers import GPT2Tokenizer, GPT2Model

device = "cpu"
model_id = 'openai-community/gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_id)
model = GPT2Model.from_pretrained(model_id)
model.eval()
model_path = Path("gpt2")
model.save_pretrained(model_path)

# ------------------------------------------------------------
# Get some model architecture values, required to convert then to ONNX.
# ------------------------------------------------------------

num_layer = model.config.n_layer
num_attention_heads = model.config.n_head
hidden_size = model.config.n_embd

# ------------------------------------------------------------
# Define a very simple function to benchmark the different versions of the GPT-2 small model (latency only).
# ------------------------------------------------------------

import timeit

def benchmark(f, name=""):
    for _ in range(10):
        f()
    seconds_per_iter = timeit.timeit(f, number=100) / 100
    print(
        f"{name}:",
        f"{seconds_per_iter * 1000:.3f} ms",
    )

# ------------------------------------------------------------
# Set a prompt an tokenize it for text generation with the vanilla model.
# ------------------------------------------------------------

text = "Today is Saturday and"
inputs_base = tokenizer(text, return_tensors="pt").to(device)

# ------------------------------------------------------------
# Benchmark the vanilla model.
# 
# ------------------------------------------------------------

benchmark(lambda: model(**inputs_base), "PyTorch")

# ------------------------------------------------------------
# Convert the vanilla model to ONNX format
# ------------------------------------------------------------

import torch
from transformers import BatchEncoding

input_ids: BatchEncoding = tokenizer(
    text, add_special_tokens=True, return_attention_mask=False, return_tensors="pt"
)
for k, v in input_ids.items():
    input_ids[k] = v.type(dtype=torch.int32)
input_tensor = input_ids['input_ids']
onnx_model_path = "gpt2_onnx.onnx"
dynamic_axes = {
    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
    'logits': {0: 'batch_size', 1: 'sequence_length'},
}

torch.onnx.export(
    model,
    f=onnx_model_path,
    args= (input_tensor,),
    input_names=['input_ids'],
    output_names=['logits'],
    quantization=False,
    var_output_seq=True,
    do_constant_folding=True,
    opset_version=18,
    dynamic_axes=dynamic_axes
)

# ------------------------------------------------------------
# Just in case, if you need to focus only one version of the model and run larger benchmarks on it, and need to free memory, remove the vanilla model (not needed anymore).
# ------------------------------------------------------------

del model

import gc
gc.collect()

# ------------------------------------------------------------
# Create an inference session for the ONNX model.
# ------------------------------------------------------------

from onnxruntime import InferenceSession

providers=["CPUExecutionProvider"]
sess = InferenceSession(onnx_model_path,
                        providers=providers)

# ------------------------------------------------------------
# Prepare the input to benchmark this and the other ONNX version of the model that we are going to build.
# ------------------------------------------------------------

import numpy as np
import torch

encodings_dict = tokenizer.batch_encode_plus([text])
input_ids = torch.tensor(encodings_dict["input_ids"], dtype=torch.int32)
empty_past = []
batch_size = input_ids.size(0)
sequence_length = input_ids.size(1)
past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
for i in range(num_layer):
    empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))

ort_inputs = {
    "input_ids": input_ids.cpu().numpy()
}

# ------------------------------------------------------------
# Benchmark the ONNX model.
# ------------------------------------------------------------

benchmark(lambda: sess.run(None, ort_inputs), "ONNX")

# ------------------------------------------------------------
# To free memory, remove the ONNX model inference session (not needed anymore).
# ------------------------------------------------------------

del sess

import gc
gc.collect()

# ------------------------------------------------------------
# Optimize the ONNX model.
# ------------------------------------------------------------

from onnxruntime.transformers.optimizer import optimize_model

optimized_onnx_path = "gpt2_optimized.onnx"
optimized_model = optimize_model(input=onnx_model_path,
                                 model_type="gpt2",
                                 use_gpu=False)
optimized_model.save_model_to_file(optimized_onnx_path)

# ------------------------------------------------------------
# Create an inference session for the optimized ONNX model.
# ------------------------------------------------------------

optimized_sess = InferenceSession(optimized_onnx_path,
                                  providers=providers)

# ------------------------------------------------------------
# Benchmark the optimized ONNX model.
# ------------------------------------------------------------

benchmark(lambda: optimized_sess.run(None, input_feed=ort_inputs),
          "ONNX optimized")

# ------------------------------------------------------------
# Just in case, if you need to focus only one version of the model and run larger benchmarks on it, and need to free memory, remove the optimize ONNX model inference session (not needed anymore).
# ------------------------------------------------------------

del optimized_sess

import gc
gc.collect()

# ------------------------------------------------------------
# Downsize the optimized ONNX model to FP16.
# ------------------------------------------------------------

from copy import deepcopy

optimized_fp16_model_path = "optimized_fp16.onnx"
optimized_fp16_model = deepcopy(optimized_model)
optimized_fp16_model.convert_float_to_float16()
optimized_fp16_model.save_model_to_file(optimized_fp16_model_path)

# ------------------------------------------------------------
# To free memory, remove the optimized ONNX model (not needed anymore).
# ------------------------------------------------------------

del optimized_model

import gc
gc.collect()

# ------------------------------------------------------------
# Create an inference session for the optimized FP16 ONNX model.
# ------------------------------------------------------------

optimized_fp16_sess = InferenceSession(
    optimized_fp16_model_path, providers=providers
)

# ------------------------------------------------------------
# Benchmark the optimized FP16 ONNX model.
# ------------------------------------------------------------

benchmark(lambda: optimized_fp16_sess.run(None, input_feed=ort_inputs),
          "ONNX optimized fp16")

# ------------------------------------------------------------
# Run the benchmark for each version of the model using different values of `max_length` (1, 4, 64, 256, 512, 1024) and then compare the results.
# ------------------------------------------------------------

import numpy as np

tokenizer.pad_token = tokenizer.eos_token
MAX_SEQUENCE_LENGTH=1024
for n in [1, 4, 64, 256, 512, 1024]:
    print(f"====== Tokens {n} ======")
    txt = " ".join(["word"] * n)

    pt_inputs_base = tokenizer(txt,
                              max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt").to(device)
    pt_inputs = tokenizer(txt,
                              max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt").to(device)
    ort_inputs = dict(tokenizer(txt,
                              max_length=MAX_SEQUENCE_LENGTH,
                              return_tensors="np",
                              return_attention_mask=False))
    ort_inputs['input_ids'] = ort_inputs['input_ids'].astype(np.int32)

    benchmark(lambda: model(**pt_inputs), f"Pytorch ({n} tokens)")
    benchmark(lambda: sess.run(None, {'input_ids': ort_inputs['input_ids']}), f"ONNX ({n} tokens)")
    benchmark(lambda: optimized_sess.run(None, ort_inputs), f"ONNX optimized ({n} tokens)")
    benchmark(
        lambda: optimized_fp16_sess.run(None, ort_inputs),
        f"ONNX optimized fp16 ({n} tokens)",
    )

