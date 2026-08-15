# ==========================================
# Extracted from CH06_NB04_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # 4-bit Quantization of GPT-2 with Auto-GPTQ
# This notebook is a companion of chapter 6 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is to introduce readers to 4-bit quantization of a decoder-only language model, [GPT-2](https://huggingface.co/openai-community/gpt2), using the [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) library. It requires hardware acceleration.  
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the missing dependecies (AutGPTQ only).
# ------------------------------------------------------------

# !export BUILD_CUDA_EXT=0
# !pip install -q auto-gptq

# ------------------------------------------------------------
# Force the upgrade to the latest HF's Dataset package. A runtime restart would be probably needed when completed.
# ------------------------------------------------------------

# !pip install --force-reinstall datasets

# ------------------------------------------------------------
# Import the required classes/packages
# ------------------------------------------------------------

import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import TextGenerationPipeline

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# ------------------------------------------------------------
# Define a function to load and prepare the test set to be used for model quantization and validation.
# ------------------------------------------------------------

def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # load dataset and preprocess
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset, testenc

# ------------------------------------------------------------
# Specify the model ID in the HF's Hub and the destination directory where to save the quantized model.
# ------------------------------------------------------------

model_id = "openai-community/gpt2"
quantized_model_dir = "gpt-2-4bit"

# ------------------------------------------------------------
# Download the model tokenizer from the HF's Hub.
# ------------------------------------------------------------

from transformers import AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# ------------------------------------------------------------
# Set the quantization configuration.
# ------------------------------------------------------------

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
)

# ------------------------------------------------------------
# Download the un-quantized model (it is forced to be loaded into CPU). Then get the maximum sequence lenght for it.
# ------------------------------------------------------------

model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)
model_config = model.config.to_dict()
seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
if any(k in model_config for k in seq_len_keys):
    for key in seq_len_keys:
        if key in model_config:
            model.seqlen = model_config[key]
            break
else:
    print("The model's sequence length cannot be retrieved from its configuration. It will then be set to 2048.")
    model.seqlen = 2048

# ------------------------------------------------------------
# Load and prepare the dataset for the quantization process.
# ------------------------------------------------------------

traindataset, testenc = get_wikitext2(128, 0, model.seqlen, tokenizer)

# ------------------------------------------------------------
# Quantize the model. The examples used should be a list of dict whose keys contains "input_ids" and "attention_mask".
# ------------------------------------------------------------

model.quantize(traindataset, use_triton=False)

# ------------------------------------------------------------
# Save the quantized model to disk.
# ------------------------------------------------------------

model.save_quantized(quantized_model_dir, use_safetensors=True)

# ------------------------------------------------------------
# The size of the saved safetensors is 1.02 GB.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Load the quantized model.
# ------------------------------------------------------------

quantized_model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir,
                                           device="cuda:0", use_triton=False)

# ------------------------------------------------------------
# Do inference with the quantized model.
# ------------------------------------------------------------

prompt = "Auto-GPTQ is"
output = tokenizer.decode(
    quantized_model.generate(**tokenizer(prompt, return_tensors="pt").to("cuda:0"))[0])
print(output)

# ------------------------------------------------------------
# HF Transformers pipelines are supported too for inference with the 4-bit quantized model.
# ------------------------------------------------------------

pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device="cuda:0")
print(pipeline(prompt)[0]["generated_text"])

# ------------------------------------------------------------
# ### Weight Comparison.
# The following 5 code cells are meant to display the weights of the original and the 4-bit quantized model in a histogram chart, same way as for the 8-bit quantization case presented in the CH05_NB02_Iozzia.ipynb notebook. Please refer to it for more details.
# ------------------------------------------------------------

model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)

weights = [param.data.clone() for param in model.parameters()]
weights_int8 = [param.data.clone() for param in quantized_model.parameters()]

weights = np.concatenate([t.cpu().numpy().flatten() for t in weights])
weights_int8 = np.concatenate([t.cpu().numpy().flatten() for t in weights_int8])

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set background style
plt.style.use('ggplot')

# Create figure and axes
fig, axs = plt.subplots(1, figsize=(10,10), dpi=300, sharex=True)

# Plot the histograms for original and zero-point weights
axs.hist(weights, bins=150, alpha=0.5, label='Original weights', color='yellow', range=(-0.5, 0.5))
axs.hist(weights_int8, bins=150, alpha=0.5, label='LLM.int8() weights', color='blue', range=(-0.5, 0.5))

# Add grid
axs.grid(True, linestyle='--', alpha=0.6)

# Add legend
axs.legend()

# Add title and labels
axs.set_title('Comparison of Original and LLM.int8() Weights', fontsize=16)

axs.set_xlabel('Weights', fontsize=14)
axs.set_ylabel('Count', fontsize=14)
axs.yaxis.set_major_formatter(ticker.EngFormatter()) # Make y-ticks more human readable

# Improve font
plt.rc('font', size=12)

plt.tight_layout()
plt.show()

