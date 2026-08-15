# ==========================================
# Extracted from CH07_NB01_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Benchmarking Python Code Generation with Vanilla, ONNX Converted and Quantized CodeGen Models
# This notebook is a companion of chapter 7 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is to benchmark inference performance (latency and throughtput) when generating Python code using a Vanilla [CodeGen](https://huggingface.co/Salesforce/codegen-350M-mono) 350M mono model, after ONNX conversion of the same model and after 8-bit quantization. It doesn't require hardware acceleration.  
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the missing requirements in the ColabVM (only Optimum for the ONNX runtime).
# ------------------------------------------------------------

# !pip install optimum[onnxruntime]==1.21.2

# ------------------------------------------------------------
# Update the Transformers library to the latest version. A runtime restart is needed after.
# ------------------------------------------------------------

# !pip install -U transformers

# ------------------------------------------------------------
# ### Vanilla Model
# ------------------------------------------------------------

# ------------------------------------------------------------
# Download the CodeGen 350 M mono model and its tokenizer from the HF's Hub.
# ------------------------------------------------------------

from transformers import AutoTokenizer

device = "cpu"
model_id = "Salesforce/codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(model_id)

from transformers import CodeGenForCausalLM

model = CodeGenForCausalLM.from_pretrained(model_id).to(device)
model.eval()

# ------------------------------------------------------------
# Set a text prompt (a Python function header) to be used across benchmarks.
# ------------------------------------------------------------

prompt = "def hello_world():"

# ------------------------------------------------------------
# The code in the following cell is just to verify that model and tokenizer have been downloaded properly. You can skip its execution.
# ------------------------------------------------------------

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
generated_ids = model.generate(input_ids, max_length=12)
print(tokenizer.decode(generated_ids[0],
                       skip_special_tokens=True,
                       pad_token_id=50256))

# ------------------------------------------------------------
# Setup a Transformers' pipeline for inference with the Vanilla model.
# ------------------------------------------------------------

from transformers import pipeline

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                pad_token_id=50256,
                truncation=True,
                max_length=12
      )

# ------------------------------------------------------------
# Test the pipeline.
# ------------------------------------------------------------

result = pipe(prompt)
print(result[0]['generated_text'])

tokenizer.save_pretrained("local-pt-checkpoint")
model.save_pretrained("local-pt-checkpoint")

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
# Execute the benchmarks for the CodeGen vanilla model.
# ------------------------------------------------------------

results = {}
PROVIDERS = {
    ("cpu", "PyTorch CPU"),
}
results = benchmark_inference(PROVIDERS, pipe, prompt, results)

# ------------------------------------------------------------
# ### ONNX Conversion
# ------------------------------------------------------------

# ------------------------------------------------------------
# To prevent potential out of memory issues, let's delete the original model from memory.
# ------------------------------------------------------------

import gc

del model
gc.collect()

# ------------------------------------------------------------
# Convert the CodeGen 350M mono model using the Optimum package.
# ------------------------------------------------------------

from optimum.onnxruntime import ORTModelForCausalLM

model_id = 'Salesforce/codegen-350M-mono'
model = ORTModelForCausalLM.from_pretrained(model_id,
                                            export=True,
                                            provider="CPUExecutionProvider")

# ------------------------------------------------------------
# Save the converted model to disk.
# ------------------------------------------------------------

from pathlib import Path

onnx_path = Path("onnx")
model.save_pretrained(onnx_path)

# ------------------------------------------------------------
# Setup a pipeline for inference with the ONNX converted CodeGen 350M mono model.
# ------------------------------------------------------------

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                pad_token_id=50256,
                truncation=True,
                max_length=12
                )

# ------------------------------------------------------------
# Verify that the pipeline works.
# ------------------------------------------------------------

result = pipe(prompt)
result

# ------------------------------------------------------------
# Repeat the benchmark on the ONNX converted model.
# ------------------------------------------------------------

PROVIDERS = {
    ("CPUExecutionProvider", "ONNX CPU"),
}
results = benchmark_inference(PROVIDERS, pipe, prompt, results)

# ------------------------------------------------------------
# ### 8-bit Quantization
# ------------------------------------------------------------

# ------------------------------------------------------------
# To prevent potential out of memory issues, let's delete the pipeline from memory.
# ------------------------------------------------------------

del pipe
gc.collect()

# ------------------------------------------------------------
# Do dynamic 8-bit quantization of the ONNX converted model and save it to disk.
# ------------------------------------------------------------

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

dynamic_quantizer = ORTQuantizer.from_pretrained(model)
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False,
                                              per_channel=False)

model_quantized_path = dynamic_quantizer.quantize(
    save_dir=onnx_path,
    quantization_config=dqconfig,
)

# ------------------------------------------------------------
# Load the quantized model in memory before setting the pipeline for it.
# ------------------------------------------------------------

quantized_model = ORTModelForCausalLM.from_pretrained("onnx", file_name="model_quantized.onnx")

# ------------------------------------------------------------
# Setup the pipeline for inference with the quantized model.
# ------------------------------------------------------------

pipe = pipeline("text-generation",
                model=quantized_model,
                tokenizer=tokenizer,
                pad_token_id=50256,
                truncation=True,
                max_length=12
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
    ("CPUExecutionProvider", "ONNX Quant CPU"),
}
results = benchmark_inference(PROVIDERS, pipe, prompt, results)

# ------------------------------------------------------------
# ### Results of the Benchmarks
# ------------------------------------------------------------

# ------------------------------------------------------------
# Visually compare the average inference times across benchmarks for the 3 different versions of the model.
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
# Visually compare inference durations across benchmarks for the 3 different versions of the model.
# ------------------------------------------------------------

results_df = pd.DataFrame(columns=['Provider', 'Inference_time'])
for k, v in results.items():
  for i in range(len(v.model_inference_time)):
    results_df.loc[len(results_df.index)] = [k, v.model_inference_time[i] * 1e3]

fig = px.box(results_df, x="Provider", y="Inference_time",
             points="all",
             labels={'Provider':'Provider', 'Inference_time':'Inference durations (ms)'})
fig.show()

