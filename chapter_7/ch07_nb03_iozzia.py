# ==========================================
# Extracted from CH07_NB03_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Benchmarking Python Code Generation with Vanilla and 8-bit Quantized StarCoder2 Models
# This notebook is a companion of chapter 7 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is to benchmark inference performance (latency and throughtput) when generating Python code using a vanilla [StarCoder2](https://huggingface.co/Salesforce/codegen-350M-mono) 2B model, and after 8-bit quantization of the same model. It reuqires hardware acceleration.  
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the missing requirements in the ColabVM (only HF's Optimum for the ONNX runtime and Bitsandbytes).
# ------------------------------------------------------------

# !pip install optimum[onnxruntime-gpu]==1.21.2
# !pip install -U bitsandbytes

# ------------------------------------------------------------
# Upgrade the Numpy and HF's Transformers packages to the latest version. A restart of the VM is needed after.
# ------------------------------------------------------------

# !pip install -U numpy transformers

# ------------------------------------------------------------
# ### Vanilla Model
# ------------------------------------------------------------

# ------------------------------------------------------------
# Download the StarCoder2-3B model (in bfloat16) and its tokenizer from the HF's Hub.
# ------------------------------------------------------------

from transformers import AutoTokenizer

model_id = "bigcode/starcoder2-3b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

import torch
from transformers import AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map='auto',
                                             torch_dtype=torch.bfloat16)
model.eval()

# ------------------------------------------------------------
# Set a text prompt (a Python function header) to be used across benchmarks.
# ------------------------------------------------------------

prompt = "def print_hello_world():"

# ------------------------------------------------------------
# The code in the following cell is just to verify that model and tokenizer have been downloaded properly. You can skip its execution.
# ------------------------------------------------------------

inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))

# ------------------------------------------------------------
# Setup a Transformers' pipeline for inference with the vanilla model.
# ------------------------------------------------------------

from transformers import pipeline

pipe = pipeline("text-generation",
            model=model,
            tokenizer=tokenizer,
            do_sample=True,
            use_cache=True,
            temperature=0.2,
            top_p=0.95,
            max_length=14
)

# ------------------------------------------------------------
# Test the pipeline.
# ------------------------------------------------------------

result = pipe(prompt)
print(result[0]['generated_text'])

# ------------------------------------------------------------
# Save the checkpoints locally, to be reused when quantizing it later.
# ------------------------------------------------------------

checkpoint_save_dir = 'local-pt-checkpoint'
tokenizer.save_pretrained(checkpoint_save_dir)
model.save_pretrained(checkpoint_save_dir)

# ------------------------------------------------------------
# Define some utils for benchmarking (more details about them in chapter 6 of the book).
# ------------------------------------------------------------

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

@contextmanager
def track_infer_time(time_buffer):
    start_time = perf_counter()
    yield
    end_time = perf_counter()

    time_buffer.append(end_time - start_time)

@dataclass
class BenchmarkInferenceResult:
    model_inference_time: [int]
    optimized_model_path: str

# ------------------------------------------------------------
# Define a custom funtion to be reused across benchmarks with the different versions of the model under evaluation.
# ------------------------------------------------------------

from tqdm import trange

def benchmark_inference(providers_dict, pipe, prompt, results):
  for device, label in PROVIDERS:
      for _ in trange(10, desc="Warming up"):
          pipe(prompt)

      time_buffer = []
      for _ in trange(100, desc=f"Tracking inference time ({label})"):
        with track_infer_time(time_buffer):
            pipe(prompt)

      results[label] = BenchmarkInferenceResult(
          time_buffer,
          None
      )

  return results

# ------------------------------------------------------------
# Execute the benchmarks for the StarCoder2 vanilla model.
# ------------------------------------------------------------

results = {}
PROVIDERS = {
    ("gpu", "PyTorch GPU"),
}
results = benchmark_inference(PROVIDERS, pipe, prompt, results)

# ------------------------------------------------------------
# ### 8-bit Quantization
# ------------------------------------------------------------

# ------------------------------------------------------------
# To prevent potential out of memory issues, let's do some VRAM and RAM cleanup.
# ------------------------------------------------------------

import gc

model.cpu()
del model
del pipe
gc.collect()
torch.cuda.empty_cache()

# ------------------------------------------------------------
# Do 8-bit quantization of the original model and save it to disk.
# ------------------------------------------------------------

from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(checkpoint_save_dir)
quantized_model = AutoModelForCausalLM.from_pretrained(checkpoint_save_dir,
                                        quantization_config=quantization_config)
quantized_model.eval()

# ------------------------------------------------------------
# The code in the following cell is just to verify that model and tokenizer have been downloaded properly. You can skip its execution.
# ------------------------------------------------------------

inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
outputs = quantized_model.generate(inputs)
print(tokenizer.decode(outputs[0]))

quantized_model.save_pretrained('local-8bit-checkpoint')

checkpoint_8bit_save_dir = 'local-8bit-checkpoint'

# Load the quantized model from the specified directory
quantized_model_loaded = AutoModelForCausalLM.from_pretrained(checkpoint_8bit_save_dir)
quantized_model_loaded.eval()

# ------------------------------------------------------------
# Setup the pipeline for inference with the quantized model.
# ------------------------------------------------------------

pipe = pipeline("text-generation",
            model=quantized_model_loaded,
            tokenizer=tokenizer,
            do_sample=True,
            use_cache=True,
            temperature=0.2,
            top_p=0.95,
            max_length=14,
)

# ------------------------------------------------------------
# Verify that the pipeline works as expected.
# ------------------------------------------------------------

result = pipe(prompt)
result

# ------------------------------------------------------------
# Repeat the benchmark on the quantized model.
# ------------------------------------------------------------

PROVIDERS = {
    ("ort", "Quant GPU"),
}
results = benchmark_inference(PROVIDERS, pipe, prompt, results)

# ------------------------------------------------------------
# ### Results of the Benchmarks
# ------------------------------------------------------------

# ------------------------------------------------------------
# Visually compare the average inference times across benchmarks for the 2 different versions of the model.
# ------------------------------------------------------------

import numpy as np
import plotly.express as px

# Compute average inference time
time_results = {k: np.mean(v.model_inference_time) * 1e3 for k, v in results.items()}

fig = px.bar(x=time_results.keys(), y=time_results.values(),
             title="Average inference time (ms) for each provider",
             labels={'x':'Provider', 'y':'Avg Inference time (ms)'},
             text_auto='.2s')
fig.show()

# ------------------------------------------------------------
# Calculate latency and throughput metrics for the 3 benchmark sets and put them into a Pandas DataFrame.
# ------------------------------------------------------------

time_results = {k: np.mean(v.model_inference_time) * 1e3 for k, v in results.items()}
time_results_std = {k: np.std(v.model_inference_time) * 1000 for k, v in results.items()}

perf_results = {}
for k, v in results.items():
  latency_list = v.model_inference_time
  latency_50 = np.percentile(latency_list, 50) * 1e3
  latency_75 = np.percentile(latency_list, 75) * 1e3
  latency_90 = np.percentile(latency_list, 90) * 1e3
  latency_95 = np.percentile(latency_list, 95) * 1e3
  latency_99 = np.percentile(latency_list, 99) * 1e3

  average_latency = np.mean(v.model_inference_time) * 1e3
  throughput = 1 * (1000 / average_latency)

  perf_results[k] = (
        average_latency,
        latency_50,
        latency_75,
        latency_90,
        latency_95,
        latency_99,
        throughput,
    )

import pandas as pd

index_labels = ['Average_latency (ms)', 'Latency_P50', 'Latency_P75',
                'Latency_P90', 'Latency_P95', 'Latency_P99', 'Throughput']
perf_df = pd.DataFrame(data=perf_results, index=index_labels)
perf_df

# ------------------------------------------------------------
# Visually compare inference durations across benchmarks for the 2 different versions of the model.
# ------------------------------------------------------------

results_df = pd.DataFrame(columns=['Provider', 'Inference_time'])
for k, v in results.items():
  for i in range(len(v.model_inference_time)):
    results_df.loc[len(results_df.index)] = [k, v.model_inference_time[i] * 1e3]

fig = px.box(results_df, x="Provider", y="Inference_time",
             points="all",
             labels={'Provider':'Provider', 'Inference_time':'Inference durations (ms)'})
fig.show()

