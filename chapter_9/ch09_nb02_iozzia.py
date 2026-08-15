# ==========================================
# Extracted from CH09_NB02_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Using SmoothQuant on OPT large models
# This notebook is a companion of chapter 9 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is to show evidence that for LLMs having more 6 or more billion parameters, systematic outliers in a model's activations lead to a degradation in accuracy after quantization, and that the application of the [SmoothQuant](https://github.com/mit-han-lab/smoothquant) technique mitigates that risk. While the code refers to the Meta AI's [OPT 6.7 B](https://huggingface.co/facebook/opt-6.7b) model, the same applies to other models too. It requires hardware acceleration to be executed.  
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Force the upgrade of the HF's Datasets library to the latest version. Restart the runtime at the end of this upgrade and before moving on with other cells code execution.
# ------------------------------------------------------------

# !pip install --force-reinstall datasets

# ------------------------------------------------------------
# Install SmoothQuant from source.
# ------------------------------------------------------------

# !pip install git+https://github.com/mit-han-lab/smoothquant.git

# ------------------------------------------------------------
# Import the required dependencies.
# ------------------------------------------------------------

import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear

# ------------------------------------------------------------
# Define a custom finction to quantize a model (weights and activations) in INT8 precision.
# ------------------------------------------------------------

def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant,
                                          act_quant=act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant,
                                          act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant,
                quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant,
                quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant,
                quantize_output=quantize_bmm_input)
            m.out_proj = W8A8Linear.from_float(m.out_proj,
                                               weight_quant=weight_quant, act_quant=act_quant)
    return model

# ------------------------------------------------------------
# Implementa a class to evaluate an LLM given a test dataset.
# ------------------------------------------------------------

class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc

# ------------------------------------------------------------
# Download a subset (1000 samples in this case) of the LAMBADA dataset and the Meta AI OPT 6.7B model's tokenizer from the Hugging Face's Hub and then create an instance of the Evaluator class using them. Everything goes to GPU.
# ------------------------------------------------------------

from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch

model_id = 'facebook/opt-6.7b'
tokenizer = GPT2Tokenizer.from_pretrained(model_id)
dataset = load_dataset('cimec/lambada', split='validation[:1000]')
evaluator = Evaluator(dataset, tokenizer, 'cuda')

print("Dataset loaded and Evaluator initialized successfully.")

# ------------------------------------------------------------
# #### FP16 Model Accuracy
# ------------------------------------------------------------

# ------------------------------------------------------------
# Download the Meta AI OPT 6.7B model in FP16 from the HF's Hub.
# ------------------------------------------------------------

model_fp16 = OPTForCausalLM.from_pretrained(model_id,
                                            torch_dtype=torch.float16,
                                            device_map='auto',
                                            offload_folder='.')
model_fp16.eval()

# ------------------------------------------------------------
# Evaluate the model on the 1000 samples from the LAMBADA dataset.
# ------------------------------------------------------------

acc_fp16 = evaluator.evaluate(model_fp16)
print(f'Original model (fp16) accuracy: {acc_fp16}')

# ------------------------------------------------------------
# #### Naive W8A8 Quantized Model Accuracy
# ------------------------------------------------------------

# ------------------------------------------------------------
# Quantize weights and activation of the vanilla model (no SmoothQuant).
# ------------------------------------------------------------

model_w8a8 = quantize_model(model_fp16)
print(model_w8a8)

# ------------------------------------------------------------
# Evaluate the quantized model on the 1000 samples from the LAMBADA dataset.
# ------------------------------------------------------------

acc_w8a8 = evaluator.evaluate(model_w8a8)
print(f'Naive W8A8 quantized model accuracy: {acc_w8a8}')

# ------------------------------------------------------------
# #### SmoothQuant W8A8 Quantized Model Accuracy
# ------------------------------------------------------------

# ------------------------------------------------------------
# **To save time and free GPU memory to evaluate the model after applying SmoothQuant, a runtime restart is recommended at this time, before proceeding further.**
# ------------------------------------------------------------

# ------------------------------------------------------------
# Download the specific model's scales from the HF's Hub (mandatory to apply SmoothQuant).
# ------------------------------------------------------------

# !mkdir ./act_scales
# %cd act_scales
# !wget https://huggingface.co/mit-han-lab/smoothquant-scales/resolve/main/opt-6.7b.pt
# %cd ..

# ------------------------------------------------------------
# Apply SmoothQuant and after quantize the vanilla model's weights and activations in INT8 format.
# ------------------------------------------------------------

model_fp16 = OPTForCausalLM.from_pretrained(model_id,
                                            torch_dtype=torch.float16,
                                            device_map='auto',
                                            offload_folder='.')

act_scales = torch.load('./act_scales/opt-6.7b.pt')
smooth_lm(model_fp16, act_scales, 0.5)
model_smoothquant_w8a8 = quantize_model(model_fp16)
print(model_smoothquant_w8a8)

# ------------------------------------------------------------
# Evaluate the smooth quantized model on the 1000 samples from the LAMBADA dataset.
# ------------------------------------------------------------

model_smoothquant_w8a8.eval()

acc_smoothquant_w8a8 = evaluator.evaluate(model_smoothquant_w8a8)
print(f'SmoothQuant W8A8 quantized model accuracy: {acc_smoothquant_w8a8}')

# ------------------------------------------------------------
# The accuracy of the vanilla model and its smooth quantized version should be comparable, while there should be a significant drop (up to 40%) for the naive quantized model.
# ------------------------------------------------------------

