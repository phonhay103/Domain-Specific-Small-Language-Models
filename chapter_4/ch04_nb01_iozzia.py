# ==========================================
# Extracted from CH04_NB01_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # GPT-Neo inference with the HF's Transformers Library
# This notebook is a companion of chapter 4 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is to introduce readers to the inference (text generation) with the [GPT-Neo model](https://github.com/EleutherAI/gpt-neo) using the Hugging Face's [Transformers library](https://github.com/huggingface/transformers). It can be executed in the Colab free tier with hardware acceleration (GPU).  
# More details about the code can be found in the book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the missing requirements in the Colab VM (HF's Accelerate only).
# ------------------------------------------------------------

# !pip install accelerate

# ------------------------------------------------------------
# Download the GPT-Neo 2.7B model and the associated tokenizer from the HF's Hub. The model is loaded in full precision and is then loaded into the GPU.
# ------------------------------------------------------------

import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "EleutherAI/gpt-neo-2.7B"
tokenizer = GPT2Tokenizer.from_pretrained(model_id)
model = GPTNeoForCausalLM.from_pretrained(model_id, device_map="auto")
model.to(device)

# ------------------------------------------------------------
# Verify where the model layers have been loaded (all in the GPU memory or also RAM and/or disk).
# ------------------------------------------------------------

model.hf_device_map

# ------------------------------------------------------------
# Perform standard inference (text completion).
# ------------------------------------------------------------

prompt = "The story so far: in the beginning, the universe was created."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generated_ids = model.generate(input_ids,
                               do_sample=True,
                               temperature=0.9,
                               max_length=200,
                               pad_token_id=50256)
generated_text = tokenizer.decode(generated_ids[0])
print(generated_text)

# ------------------------------------------------------------
# Do few-shot text classification (the model can generalize learning from few new and unseen examples.
# ------------------------------------------------------------

prompt = """
Sentence: This movie is very nice.
Sentiment: positive

#####

Sentence: I hated this movie, it sucks.
Sentiment: negative

#####

Sentence: This movie was actually pretty funny.
Sentiment: positive

#####

Sentence: This movie could have been better.
Sentiment: neutral
"""
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generated_ids = model.generate(input_ids,
                               do_sample=True,
                               temperature=0.9,
                               max_length=200,
                               pad_token_id=50256)
generated_text = tokenizer.decode(generated_ids[0])
print(generated_text)

# ------------------------------------------------------------
# Do Python code generation.
# ------------------------------------------------------------

prompt = """Instruction: Generate a Python function that lets you reverse a list of integers.

Answer: """
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generated_ids = model.generate(input_ids,
                               do_sample=True,
                               temperature=0.9,
                               max_length=200,
                               pad_token_id=50256
                               )
generated_text = tokenizer.decode(generated_ids[0])
print(generated_text)

# ------------------------------------------------------------
# Do batch text completion.
# ------------------------------------------------------------

texts = ["Once there was a man ", "The weather today will be ", "A great soccer player must "]

tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)
with torch.no_grad():
    generated_ids = model.generate(**encoding,
                                   do_sample=True,
                                   temperature=0.9,
                                   max_length=50,
                                   pad_token_id=50256)
generated_texts = tokenizer.batch_decode(
    generated_ids, skip_special_tokens=True)

for text in generated_texts:
  print("---------")
  print(text)

# ------------------------------------------------------------
# Benchmarking the model on text completion: comparing the cases where the KV cache is used to those where it isn't.
# ------------------------------------------------------------

import time
import numpy as np

prompt = "The story so far: in the beginning, the universe was created."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

for use_cache in (True, False):
  times = []
  for _ in range(20):
    start = time.time()
    generated_ids = model.generate(input_ids,
                                  do_sample=True,
                                  temperature=0.9,
                                  max_length=200,
                                  pad_token_id=50256,
                                  use_cache=use_cache)
    times.append(time.time() - start)
  print(f"{'Using' if use_cache else 'No'} KV cache: {round(np.mean(times), 3)} +- {round(np.std(times), 3)} seconds")

# ------------------------------------------------------------
# Benchmarking the model's total generation time.
# ------------------------------------------------------------

import time
import numpy as np

prompt = "The story so far: in the beginning, the universe was created."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

max_length = 300
times = []
inference_runs = 21
for _ in range(inference_runs):
  start = time.time()
  generated_ids = model.generate(input_ids,
                                do_sample=True,
                                temperature=0.9,
                                max_length=max_length,
                                pad_token_id=50256,
                                )
  times.append(time.time() - start)
print(f"Average Total Generation time: {round(np.mean(times[1:]), 3)} +- {round(np.std(times[1:]), 3)} seconds")

