# ==========================================
# Extracted from CH05_NB02_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # ONNX Conversion of the GPT-2 Small Model
# This notebook is a companion of chapter 5 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2025.  
# The code in this notebook is to introduce readers to the [ONNX](https://onnx.ai/) format and [ONNX Runtime](https://onnxruntime.ai/) on GPU with the [GPT-2 Small](https://huggingface.co/openai-community/gpt2) model. It requires hardware acceleration (GPU).  
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# *** **Update September 2025: the code in this notebook isn't anymore compatible with PyTorch 2.1 or later and the HF's Transformers releases that support the latest PyTorch. We need then to downgrade PyTorch and the Transformers packages.** ***
# ------------------------------------------------------------

# !pip install torch==2.0.1 transformers==4.31.0

# ------------------------------------------------------------
# Install the missing requirements (only ONNX and the ONNX runtime for GPUs).
# ------------------------------------------------------------

# !pip install onnx onnxruntime-gpu

# ------------------------------------------------------------
# Download the GPT-2 Small model from the Hugging Face Hub and load it into the GPU memory.
# ------------------------------------------------------------

import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM

model_id = 'openai-community/gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
device = torch.device("cuda")
model.eval().to(device)

# ------------------------------------------------------------
# Verify that the downloaded model works as expected.
# ------------------------------------------------------------

inputs = tokenizer("The story so far: in the beginning, the universe was created.", return_attention_mask=False, return_tensors="pt")
print("input tensors")
print(inputs.to(device))
print("input tensor shape")
print(inputs["input_ids"].size())

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
print("output tensor")
print(logits)
print("output shape")
print(logits.shape)

# ------------------------------------------------------------
# Create a directory where to store the ONNX converted model.
# ------------------------------------------------------------

import os

output_dir = os.path.join(".", "onnx_models")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
export_model_path = os.path.join(output_dir, 'gpt-2.onnx')

# ------------------------------------------------------------
# Create an input tensor to be used for model conversion.
# ------------------------------------------------------------

tokenized_inputs = tokenizer("The story so far: in the beginning, the universe was created.",
                             return_attention_mask=False,
                             return_tensors="pt")
tokenized_inputs.to(device)
inputs_sample = {
        'input_ids':  tokenized_inputs['input_ids']
    }

# ------------------------------------------------------------
# Convert the model to ONNX.
# ------------------------------------------------------------

with torch.no_grad():
  torch.onnx.export(model,
                    inputs_sample,
                    export_model_path,
                    export_params=True,
                    opset_version=15,
                    do_constant_folding=True,
                    input_names=['input_ids']
                    )

# ------------------------------------------------------------
# Optimize the exported model.
# ------------------------------------------------------------

from onnxruntime.transformers import optimizer

optimized_model_path = os.path.join(output_dir, 'gpt-2-onnx_opt_gpu.onnx')
optimized_model = optimizer.optimize_model(export_model_path,
                                           model_type='gpt2',
                                           use_gpu=True,
                                           num_heads=12,
                                           hidden_size=768,
                                           verbose=True)
optimized_model.save_model_to_file(optimized_model_path)

# ------------------------------------------------------------
# Benchmark inference with the original model.
# ------------------------------------------------------------

import time

with torch.inference_mode():
    sample_output = model.generate(inputs.input_ids, max_length=64, pad_token_id=50256)
    print(tokenizer.decode(sample_output[0], skip_special_tokens=False))
    for _ in range(2):
        _ = model.generate(inputs.input_ids, max_length=64, pad_token_id=50256)
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = model.generate(inputs.input_ids, max_length=256, pad_token_id=50256)
        torch.cuda.synchronize()
    print(f"----\nPytorch: {(time.time() - start)/10:.2f}s/sequence")
_ = model.cpu()

# ------------------------------------------------------------
# Benchmark inference with the ONNX converted model.
# ------------------------------------------------------------

import onnxruntime
import numpy

session = onnxruntime.InferenceSession(export_model_path, providers=["CUDAExecutionProvider"])
onnx_input_ids = tokenizer("The story so far: in the beginning, the universe was created.",
                           return_attention_mask=False,
                           return_tensors="np")
ort_inputs = {
    "input_ids": onnx_input_ids['input_ids']
}

for _ in range(2):
  ort_outputs = session.run(None, ort_inputs)
start = time.time()
for _ in range(10):
  ort_outputs = session.run(None, ort_inputs)
print(f"----\nPytorch: {(time.time() - start)/10:.2f}s/sequence")

# ------------------------------------------------------------
# Benchmark inference with the optimized ONNX model.
# ------------------------------------------------------------

import onnxruntime
import numpy

opt_session = onnxruntime.InferenceSession(optimized_model_path, providers=["CUDAExecutionProvider"])
onnx_input_ids = tokenizer("The story so far: in the beginning, the universe was created.",
                           return_attention_mask=False,
                           return_tensors="np")
ort_inputs = {
    "input_ids": onnx_input_ids['input_ids']
}

for _ in range(2):
  ort_outputs = opt_session.run(None, ort_inputs)
start = time.time()
for _ in range(10):
  ort_outputs = opt_session.run(None, ort_inputs)
print(f"----\nPytorch: {(time.time() - start)/10:.2f}s/sequence")

