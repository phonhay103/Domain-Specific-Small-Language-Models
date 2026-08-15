# ==========================================
# Extracted from CH06_NB02_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Quantization of the GPT-2 Small Model with LLM.int8()
# This notebook is a companion of chapter 6 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is to introduce readers to the quantization of a decoder-only language model, [GPT-2 Small](https://huggingface.co/openai-community/gpt2) using [LLM.int8()](https://arxiv.org/abs/2208.07339). It requires hardware acceleration (GPU).  
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the missing requirements (HF's Accelerate and Bitsandbytes).
# ------------------------------------------------------------

# !pip install accelerate bitsandbytes

# ------------------------------------------------------------
# Import the required packages and classes.
# ------------------------------------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------
# Download the GPT-2 model and associated tokenizer from the HF's Hub and load it to GPU. Finally print the size (in bytes) of the model in memory.
# ------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_id = 'openai-community/gpt2'
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"Model size: {model.get_memory_footprint():,} bytes")

# ------------------------------------------------------------
# Download the GPT-2 model in 8-bit from the HF's Hub and load it to GPU. Finally print the size (in bytes) of the model in memory.
# ------------------------------------------------------------

model_int8 = AutoModelForCausalLM.from_pretrained(model_id,
                                                device_map='auto',
                                             load_in_8bit=True,
                                             )

print(f"Model size: {model_int8.get_memory_footprint():,} bytes")

# ------------------------------------------------------------
# Get the original model's and 8-bit model's weights and prepare them for visualization in a histogram chart.
# ------------------------------------------------------------

import numpy as np

weights = [param.data.clone() for param in model.parameters()]
weights = np.concatenate([t.cpu().numpy().flatten() for t in weights])
weights_int8 = [param.data.clone() for param in model_int8.parameters()]
weights_int8 = np.concatenate([t.cpu().numpy().flatten() for t in weights_int8])

# ------------------------------------------------------------
# Using the matplotlib library, plot the distribution of the weights for the original model and the 8-bit version both on the same histogram chart.
# ------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set background style
plt.style.use('ggplot')

# Create figure and axes
fig, axs = plt.subplots(1, figsize=(10,10), dpi=300, sharex=True)

# Plot the histograms for original and zero-point weights
axs.hist(weights, bins=150, alpha=0.5, label='Original weights', color='blue', range=(-2, 2))
axs.hist(weights_int8, bins=150, alpha=0.5, label='LLM.int8() weights', color='yellow', range=(-2, 2))

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
# Use the text generation function defined in the previous code cell to generate text with both model version (the original and the 8-bit quantizated one).
# ------------------------------------------------------------

prompt = 'My favourite school subject is'
original_text = generate_text(model, prompt)
text_int8 = generate_text(model_int8, prompt)

print(f"Original model:\n{original_text}")
print(f"LLM.int8() model:\n{text_int8}")

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
perplexity_int8 = calculate_perplexity(model_int8, text_int8, device)
print(f"Original Perplexity:   {perplexity.item():.2f}")
print(f"LLM.int8() perplexity: {perplexity_int8.item():.2f}")

