# ==========================================
# Extracted from CH04_NB02_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Accelerating inference for GPT-Neo with DeepSpeed
# This notebook is a companion of chapter 4 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is to introduce readers to the [DeepSpeed](https://github.com/microsoft/DeepSpeed) library to accelerate inference for the [GPT-Neo model](https://github.com/EleutherAI/gpt-neo) for text generation tasks. It can be executed in the Colab free tier with hardware acceleration (GPU).  
# More details about the code can be found in the book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the missing dependencies in the Colab VM (DeepSpeed and HF's Accelerate only).
# ------------------------------------------------------------

# !pip install deepspeed accelerate

# ------------------------------------------------------------
# Before loading the model, let's define a custom function to be used for benchmarking (latency measurement).
# ------------------------------------------------------------

from time import perf_counter
import numpy as np

def measure_latency(model, tokenizer, payload, device, generation_args={}):
    input_ids = tokenizer(payload, return_tensors="pt").input_ids.to(device)
    latencies = []
    # Do GPU warm up before benchmarking
    for _ in range(2):
        _ =  model.generate(input_ids, **generation_args)
    # Runs used for measuring the latency
    for _ in range(20):
        start_time = perf_counter()
        _ = model.generate(input_ids, **generation_args)
        latency = perf_counter() - start_time
        latencies.append(latency)

    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies,95)

    return f"P95 latency (ms) - {time_p95_ms}; Average latency (ms) - {time_avg_ms:.2f} +\\- {time_std_ms:.2f};", time_p95_ms

# ------------------------------------------------------------
# Download the base GPT-Neo 2.7B model in half precision and the related tokenizer from the HF's Hub.
# ------------------------------------------------------------

import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM

model_id = "EleutherAI/gpt-neo-2.7B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                          torch_dtype=torch.float16,
                                          device_map="auto")
print(f"model is loaded on device {model.device.type}")

# ------------------------------------------------------------
# Do inference with the downloaded model to verify that everything is working as expected.
# ------------------------------------------------------------

example = "The story so far: in the beginning, the universe was created."

input_ids = tokenizer(example, return_tensors="pt").input_ids.to(model.device)
logits = model.generate(input_ids,
                        do_sample=True,
                        num_beams=1,
                        min_length=128,
                        max_new_tokens=128,
                        pad_token_id=50256)

print(f"prediction: \n \n {tokenizer.decode(logits[0].tolist()[len(input_ids[0]):])}")

# ------------------------------------------------------------
# Perform benchmark for the vanilla model. The previously defined ```measure_latency``` function is used.
# 
# ------------------------------------------------------------

generation_args = dict(do_sample=True,
                      max_length=300,
                      pad_token_id=50256,
                      use_cache=True
)
vanilla_results = measure_latency(model, tokenizer, example,
                                  model.device, generation_args)

print(f"Vanilla model: {vanilla_results[0]}")

# ------------------------------------------------------------
# Let's now optimize the base GPT-Neo 2.7B model for inference on GPU with DeepSpeed. The decision about which of the original model's layers have to be replaced is left to DeepSpeed itself here.
# ------------------------------------------------------------

import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9999'
os.environ['RANK'] = "0"
os.environ['LOCAL_RANK'] = "0"
os.environ['WORLD_SIZE'] = "1"

import deepspeed

ds_model = deepspeed.init_inference(
    model=model,
    mp_size=1,
    dtype=torch.float16,
    replace_method="auto",
    replace_with_kernel_inject=True,
)
print(f"model is loaded on device {ds_model.module.device}")

# ------------------------------------------------------------
# Print the optimized model's architecture to this cell output to verify that some of the original model's layers have been replaced with DeepSpeed optimized kernel implementations.
# ------------------------------------------------------------

ds_model

# ------------------------------------------------------------
# Do inference with the optimized model to verify that everything is working as expected.
# ------------------------------------------------------------

input_ids = tokenizer(example, return_tensors="pt").input_ids.to(model.device)
logits = ds_model.generate(input_ids,
                           do_sample=True,
                           num_beams=1,
                           min_length=128,
                           max_new_tokens=128,
                           pad_token_id=50256,
                           use_cache=False
                          )
print(tokenizer.decode(logits[0].tolist()))

# ------------------------------------------------------------
# Perform now benchmark for the DeepSpeed optimized model. The previously defined ```measure_latency``` function is used.
# ------------------------------------------------------------

generation_args = dict(do_sample=True,
                       max_length=300,
                       pad_token_id=50256,
                       use_cache=True)
ds_results = measure_latency(ds_model, tokenizer, example,
                             ds_model.module.device, generation_args)

print(f"DeepSpeed model: {ds_results[0]}")

