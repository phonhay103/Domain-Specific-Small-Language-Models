# ==========================================
# Extracted from CH06_NB01_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Quantization of the GPT-2 Small Model
# This notebook is a companion of chapter 6 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is to introduce readers to the quantization of a decoder-only language model, [GPT-2 Small](https://huggingface.co/openai-community/gpt2). It doesn't require hardware acceleration.  
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Import the required packages and classes.
# ------------------------------------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------
# Download the GPT-2 Small model and associated tokenizer from the HF's Hub and load it to CPU. Finally print the size (in bytes) of the model in memory.
# ------------------------------------------------------------

device = 'cpu'

model_id = 'openai-community/gpt2'
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"Model size: {model.get_memory_footprint():,} bytes")

# ------------------------------------------------------------
# Define a custom function to perform *absmax* quantization and dequantization.
# ------------------------------------------------------------

def absmax_quantize(X):
    # Calculate scale
    scale = 127 / torch.max(torch.abs(X))

    # Quantize
    X_quant = (scale * X).round()

    # Dequantize
    X_dequant = X_quant / scale

    return X_quant.to(torch.int8), X_dequant

# ------------------------------------------------------------
# Clone the source model and apply the previously defined quantization function to all the weights of the cloned copy.
# ------------------------------------------------------------

import numpy as np
from copy import deepcopy

weights = [param.data.clone() for param in model.parameters()]

model_abs = deepcopy(model)

weights_abs = []
for param in model_abs.parameters():
    _, dequantized = absmax_quantize(param.data)
    param.data = dequantized
    weights_abs.append(dequantized)

# ------------------------------------------------------------
# Using the matplotlib library, plot the distribution of the weights for the source model and the quantized version both on the same histogram chart.
# ------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

weights = np.concatenate([t.cpu().numpy().flatten() for t in weights])
weights_abs = np.concatenate([t.cpu().numpy().flatten() for t in weights_abs])

# Set background style
plt.style.use('ggplot')

# Create figure and axes
fig, axs = plt.subplots(1, figsize=(10,10), dpi=300, sharex=True)

# Plot the histograms for original and zero-point weights
axs.hist(weights, bins=150, alpha=0.5, label='Original weights', color='blue', range=(-2, 2))
axs.hist(weights_abs, bins=150, alpha=0.5, label='Absmax weights', color='yellow', range=(-2, 2))

# Add grid
axs.grid(True, linestyle='--', alpha=0.6)

# Add legend
axs.legend()

# Add title and labels
axs.set_title('Comparison of Original and Absmax Quantized Weights', fontsize=16)

axs.set_xlabel('Weights', fontsize=14)
axs.set_ylabel('Count', fontsize=14)
axs.yaxis.set_major_formatter(ticker.EngFormatter()) # Make y-ticks more human readable

# Improve font
plt.rc('font', size=12)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Define a function to generate text, whatever the model (original or quantized).
# ------------------------------------------------------------

def generate_text(model, input_text, max_length=100):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    output = model.generate(inputs=input_ids,
                            max_length=max_length,
                            do_sample=True,
                            top_k=30,
                            pad_token_id=tokenizer.eos_token_id,
                            attention_mask=input_ids.new_ones(input_ids.shape))
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ------------------------------------------------------------
# Use the text generation function defined in the previous code cell to generate text with both model versions (the original and the one after quantization).
# ------------------------------------------------------------

prompt = 'My favourite school subject is'
original_text = generate_text(model, prompt)
absmax_text   = generate_text(model_abs, prompt)

print(f"Original model:\n{original_text}")
print(f"Absmax model:\n{absmax_text}")

# ------------------------------------------------------------
# Define a function to calculate the perplexity score.
# ------------------------------------------------------------

def calculate_perplexity(model, text, device):
    encodings = tokenizer(text, return_tensors='pt').to(device)

    input_ids = encodings.input_ids
    target_ids = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

    neg_log_likelihood = outputs.loss

    perplexity = torch.exp(neg_log_likelihood)

    return perplexity

# ------------------------------------------------------------
# Calculate the perplexity score for both versions of the model, using the text results previously generated by both.
# ------------------------------------------------------------

perplexity = calculate_perplexity(model, original_text, device)
perplexity_absmax = calculate_perplexity(model_abs, absmax_text, device)

print(f"Original perplexity:  {perplexity.item():.2f}")
print(f"Absmax perplexity:    {perplexity_absmax.item():.2f}")

