# ==========================================
# Extracted from CH05_NB01_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # ONNX Conversion of the BERT Base Uncased Model
# This notebook is a companion of chapter 5 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is to introduce readers to the [ONNX](https://onnx.ai/) format and [ONNX Runtime](https://onnxruntime.ai/) with the [BERT Base Uncased](https://huggingface.co/google-bert/bert-base-uncased) model. It can be executed in the Colab free tier with hardware acceleration (GPU).  
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# ### Settings
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the missing requirements in the Colab VM (ONNX, the ONNX runtime and the HF's Datasets).
# ------------------------------------------------------------

# !pip install onnx onnxruntime datasets

# ------------------------------------------------------------
# Download the BERT Base Uncased model (and associated tokenizer) from the Hugging Face Hub.
# ------------------------------------------------------------

from transformers import AutoModelForQuestionAnswering, BertTokenizer

model_id = 'google-bert/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_id)
model = AutoModelForQuestionAnswering.from_pretrained(model_id)
model.eval()

# ------------------------------------------------------------
# Download a subset of the SQuAD dataset from the Hugging Face Hub.
# ------------------------------------------------------------

from datasets import load_dataset

samples_count = 200
squad = load_dataset("squad", split="validation[:"+ str(samples_count) +"]")

# ------------------------------------------------------------
# Display one test sample.
# ------------------------------------------------------------

squad[0]

# ------------------------------------------------------------
# Benchmark the original model on the selected subset of the squad test set.
# ------------------------------------------------------------

import time
import torch

max_seq_length = 128
# Measure the latency.
latency = []
with torch.no_grad():
    for i in range(samples_count):
        inputs = tokenizer(squad["question"][i], squad["context"][i], return_tensors="pt")
        start = time.time()
        outputs = model(**inputs)
        latency.append(time.time() - start)
print("PyTorch {} Average inference time = {} ms".format('CPU', format(sum(latency) * 1000 / len(latency), '.2f')))

# ------------------------------------------------------------
# ### Convert the model to ONNX.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Create the directory to host the converted model.
# ------------------------------------------------------------

import os

output_dir = os.path.join(".", "onnx_models")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
export_model_path = os.path.join(output_dir, 'bert-base-uncased.onnx')

# ------------------------------------------------------------
# Pick up one sample from the test dataset.
# ------------------------------------------------------------

tokenized_inputs = tokenizer(squad["question"][0], squad["context"][0], return_tensors="pt")
inputs = {
        'input_ids':  tokenized_inputs['input_ids'],
        'input_mask': tokenized_inputs['attention_mask'],
        'segment_ids': tokenized_inputs['token_type_ids']
    }

# ------------------------------------------------------------
# Export the model to ONNX.
# ------------------------------------------------------------

with torch.no_grad():
    symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
    torch.onnx.export(model,
                      args=tuple(inputs.values()),
                      f=export_model_path,
                      opset_version=15,
                      do_constant_folding=True,
                      input_names=['input_ids',
                                       'input_mask',
                                       'segment_ids'],
                      output_names=['start', 'end'],
                      dynamic_axes={'input_ids': symbolic_names,
                                    'input_mask' : symbolic_names,
                                    'segment_ids' : symbolic_names,
                                    'start' : symbolic_names,
                                    'end' : symbolic_names})
    print("Model exported at ", export_model_path)

# ------------------------------------------------------------
# Validate the exported model.
# ------------------------------------------------------------

from onnx.checker import check_model

check_model(export_model_path, full_check=True)

# ------------------------------------------------------------
# Benchmark the exported model (CPUExecutionProvider).
# ------------------------------------------------------------

import onnxruntime
import numpy

sess_options = onnxruntime.SessionOptions()

sess_options.optimized_model_filepath = os.path.join(output_dir, "bert-base-uncased.onnx")

session = onnxruntime.InferenceSession(export_model_path, sess_options, providers=['CPUExecutionProvider'])

latency = []
for i in range(samples_count):
    full_inputs = tokenizer(squad["question"][i], squad["context"][i], return_tensors="np")
    ort_inputs = {
        'input_ids':  full_inputs['input_ids'],
        'input_mask': full_inputs['attention_mask'],
        'segment_ids': full_inputs['token_type_ids']
    }
    start = time.time()
    ort_outputs = session.run(None, ort_inputs)
    latency.append(time.time() - start)
print("OnnxRuntime cpu Average inference time = {} ms".format(format(sum(latency) * 1000 / len(latency), '.2f')))

# ------------------------------------------------------------
# Verify correctess of the exported model.
# ------------------------------------------------------------

print("***** Verifying correctness *****")
sample_range = 2
for i in range(sample_range):
    print('PyTorch and ONNX Runtime output {} are close:'.format(i), numpy.allclose(ort_outputs[i], outputs[i].cpu(), rtol=1e-05, atol=1e-04))

# ------------------------------------------------------------
# ### Model Optimization
# ------------------------------------------------------------

# ------------------------------------------------------------
# Optimize the exported model.
# ------------------------------------------------------------

from onnxruntime.transformers import optimizer

optimized_model_path = os.path.join(output_dir, 'bert-base-uncased.onnx_opt_cpu.onnx')
optimized_model = optimizer.optimize_model(export_model_path, model_type='bert', num_heads=12, hidden_size=768)
optimized_model.save_model_to_file(optimized_model_path)

# ------------------------------------------------------------
# Benchmark the optimized model (CPUExecutionProvider).
# ------------------------------------------------------------

sess_options_opt = onnxruntime.SessionOptions()

sess_options_opt.optimized_model_filepath = os.path.join(output_dir, "bert-base-uncased.onnx_opt_cpu.onnx")

session_opt = onnxruntime.InferenceSession(export_model_path, sess_options_opt, providers=['CPUExecutionProvider'])

latency_opt = []
for i in range(samples_count):
    full_inputs = tokenizer(squad["question"][i], squad["context"][i], return_tensors="np")
    ort_inputs = {
        'input_ids':  full_inputs['input_ids'],
        'input_mask': full_inputs['attention_mask'],
        'segment_ids': full_inputs['token_type_ids']
    }
    start = time.time()
    ort_outputs = session_opt.run(None, ort_inputs)
    latency_opt.append(time.time() - start)
print("OnnxRuntime cpu Average inference time = {} ms".format(format(sum(latency_opt) * 1000 / len(latency_opt), '.2f')))

