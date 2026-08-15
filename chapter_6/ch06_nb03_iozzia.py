# ==========================================
# Extracted from CH06_NB03_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Quantization of a Finetuned BERT Model with HF's Optimum
# This notebook is a companion of chapter 6 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is to introduce readers to the quantization of an encoder-only language model, [distilbert-base-uncased-finetuned-banking77](https://huggingface.co/optimum/distilbert-base-uncased-finetuned-banking77) using the Hugging Face's [Optimum](https://github.com/huggingface/optimum) library. It doesn't require hardware acceleration.  
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the missing requirements in the Colab VM (only the latest HF's Optimum for the ONNX runtime and Evaluate).
# ------------------------------------------------------------

# !pip install optimum[onnxruntime] evaluate

# ------------------------------------------------------------
# Force the downgrade of the HF's Dataset package to release 3.6.0 because of the following [bug](https://github.com/huggingface/datasets/issues/7693) introduced in the latest version 4.0.0 (installed by default in the Colab VMs). A runtime restart would be needed when completed.
# ------------------------------------------------------------

# !pip install --force-reinstall datasets==3.6.0

# ------------------------------------------------------------
# Import the required classes.
# ------------------------------------------------------------

from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# ------------------------------------------------------------
# Load the distilbert base uncased finetuned model from the HF Hub and convert it to ONNX (fp32). Then save it and the associated tokenizer to disk.
# ------------------------------------------------------------

model_id="optimum/distilbert-base-uncased-finetuned-banking77"
onnx_path = Path("onnx")

model = ORTModelForSequenceClassification.from_pretrained(model_id,
                                                          export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained(onnx_path)
tokenizer.save_pretrained(onnx_path)

# ------------------------------------------------------------
# Check that the downloaded model works as expected. Transformers' pipelines are supported in Optimum.
# ------------------------------------------------------------

from transformers import pipeline

vanilla_clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
vanilla_clf("Could you assist me in checking my card validity?")

# ------------------------------------------------------------
# Quantize the model dynamically. First create an ORTQuantizer instance and define the quantization configuration. Then apply the quantization configuration to the model.
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
# Compare the size of the downloaded ONNX model and its quantized version.
# ------------------------------------------------------------

import os

original_model_name = "model.onnx"
quantized_model_name = "model_quantized.onnx"
size = os.path.getsize(onnx_path / original_model_name)/(1024*1024)
quantized_model = os.path.getsize(onnx_path / quantized_model_name)/(1024*1024)

print(f"Original Model file size: {size:.2f} MB")
print(f"Quantized Model file size: {quantized_model:.2f} MB")

# ------------------------------------------------------------
# Check that inference with the quantized model works as expected.
# ------------------------------------------------------------

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import pipeline, AutoTokenizer

model = ORTModelForSequenceClassification.from_pretrained(onnx_path,
                                                file_name=quantized_model_name)
tokenizer = AutoTokenizer.from_pretrained(onnx_path)

q8_clf = pipeline("text-classification",model=model, tokenizer=tokenizer)

q8_clf("Could you assist me in checking my card validity?")

# ------------------------------------------------------------
# ### Models' performance evaluation
# ------------------------------------------------------------

# ------------------------------------------------------------
# Download the test set of the Banking 77 dataset (available in the HF's Hub).
# ------------------------------------------------------------

from evaluate import evaluator
from datasets import load_dataset

dataset_id="PolyAI/banking77"
eval = evaluator("text-classification")
eval_dataset = load_dataset(dataset_id, split="test")

# ------------------------------------------------------------
# Evaluate the quantized model towards the downloaded test set (the HF's Evalate library is used).
# ------------------------------------------------------------

results = eval.compute(
    model_or_pipeline=q8_clf,
    data=eval_dataset,
    metric="accuracy",
    input_column="text",
    label_column="label",
    label_mapping=model.config.label2id,
    strategy="simple",
)
print(results)

# ------------------------------------------------------------
# Compare the test scores across the original ONNX model and its quantized version.
# ------------------------------------------------------------

print(f"Vanilla model: 92.5%")
print(f"Quantized model: {results['accuracy']*100:.2f}%")
print(f"The quantized model achieves {round(results['accuracy']/0.925,4)*100:.2f}% accuracy of the fp32 model")

# ------------------------------------------------------------
# Define a function to benchmark the execution times for both models.
# ------------------------------------------------------------

from time import perf_counter
import numpy as np

def measure_latency(payload_prompt, pipe):
    latencies = []
    # Warm up
    for _ in range(10):
        _ = pipe(payload_prompt)
    # Effective runs
    for _ in range(300):
        start_time = perf_counter()
        _ =  pipe(payload_prompt)
        latency = perf_counter() - start_time
        latencies.append(latency)

    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies,95)

    return f"P95 latency (ms) - {time_p95_ms}; Average latency (ms) - {time_avg_ms:.2f} +\\- {time_std_ms:.2f};", time_p95_ms

# ------------------------------------------------------------
# Benchmark the two versions of the model and compare the results.
# ------------------------------------------------------------

prompt="Dear Sir/Madam, my name is William. I am getting in touch because I didn't get a response from you yet. What actions do I need to do to get my new card which I have requested 3 weeks ago? Please help me and answer this email as soon as possible. Have a nice rest of the day. Best Regards."*2
print(f'Prompt length: {len(tokenizer(prompt)["input_ids"])}')

original_model_stats = measure_latency(prompt, vanilla_clf)
quantized_model_stats = measure_latency(prompt, q8_clf)

print(f"Vanilla model: {original_model_stats[0]}")
print(f"Quantized model: {quantized_model_stats[0]}")
print(f"Improvement through quantization: {round(original_model_stats[1]/quantized_model_stats[1],2)}x")

