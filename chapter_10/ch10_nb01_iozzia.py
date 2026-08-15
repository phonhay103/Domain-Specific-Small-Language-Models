# ==========================================
# Extracted from CH10_NB01_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Profiling ONNX Models
# This notebook is a companion of chapter 10 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is about profiling and getting performance insights for a [GPT-2 small]((https://huggingface.co/openai-community/gpt2) model after conversion to the [ONNX](https://onnx.ai/) format and optimization. The same code applies to any other LLM and the insights building part is generic for any ML/DL ONNX model profiling analysis. No hardware acceleration needed.  
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the missing dependencies in the Colab VM (only ONNX and the ONNX runtime, plus mlprodict (for profiling data aggregation and clean up only). Please see note later in this notebook about the mlprodict package installation in later versions of the Colab runtime.
# ------------------------------------------------------------

# !pip install onnx onnxruntime

# ------------------------------------------------------------
# Import the required packages and classes.
# ------------------------------------------------------------

import logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding, GPT2LMHeadModel

# ------------------------------------------------------------
# Download the GPT-2 small model and companion tokenizer from the HF's Hub.
# ------------------------------------------------------------

model_name = "openai-community/gpt2"

model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.eos_token_id

# ------------------------------------------------------------
# Generate text to verify that the vanilla model has been downloaded and set up properly.
# ------------------------------------------------------------

sample_prompt = 'Here is some text to encode Hello World'
inputs = tokenizer(sample_prompt, return_tensors="pt")
print("input tensors")
print(inputs)
print("input tensor shape")
print(inputs["input_ids"].size())

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
print("output tensor")
print(logits)
print("output shape")
print(logits.shape)

# ------------------------------------------------------------
# Export the model to ONNX.
# ------------------------------------------------------------

input_ids: BatchEncoding = tokenizer(
    sample_prompt, add_special_tokens=True,
    return_attention_mask=False, return_tensors="pt"
)
for k, v in input_ids.items():
    input_ids[k] = v.type(dtype=torch.int32)
input_tensor = input_ids['input_ids']

onnx_model_path='gpt2onnx.onnx'
torch.onnx.export(
    model,
    f=onnx_model_path,
    args= (input_tensor,),
    input_names=['input_ids'],
    output_names=['logits'],
    quantization=False,
    var_output_seq=True,
    do_constant_folding=True,
    opset_version=18,
)
_ = model.eval()

# ------------------------------------------------------------
# Define a custom function to prepare the input for the ONNX model to run text generation for profiling.
# ------------------------------------------------------------

def get_example_inputs(prompt_text, tokenizer, num_layer, device='cpu'):
    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)

    input_ids = torch.tensor(encodings_dict["input_ids"], dtype=torch.int32)
    attention_mask = torch.tensor(encodings_dict["attention_mask"], dtype=torch.int32)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(position_ids < 0, 0)
    position_ids = position_ids.to(torch.int32)

    empty_past = []
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
    for i in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))

    return input_ids, attention_mask, position_ids, empty_past

# ------------------------------------------------------------
# Collect some vanilla model specs that are rquired for preparing the input for the ONNX version.
# ------------------------------------------------------------

num_layer = model.config.n_layer
num_attention_heads = model.config.n_head
hidden_size = model.config.n_embd

# ------------------------------------------------------------
# Run text generation using the ONNX model with profiling enabled.
# 
# ------------------------------------------------------------

import onnxruntime

tokenizer.pad_token = tokenizer.eos_token
input_ids, attention_mask, position_ids, empty_past = get_example_inputs(['Here is some text to encode Hello World'], tokenizer, num_layer)

so = onnxruntime.SessionOptions()
so.enable_profiling = True
session = onnxruntime.InferenceSession(onnx_model_path, so, providers=["CPUExecutionProvider"])
ort_inputs = {
    "input_ids": np.ascontiguousarray(input_ids.cpu().numpy()),
}
ort_outputs = session.run(None, ort_inputs)

# ------------------------------------------------------------
# Close the inference session and collect the profiling data. The `prof` variable contains the name of the generated JSON file.
# ------------------------------------------------------------

prof = session.end_profiling()

# ------------------------------------------------------------
# # Model Optimization
# ------------------------------------------------------------

# ------------------------------------------------------------
# Set up the logging level to see in the output which kind of optimizations are automatically applied.
# ------------------------------------------------------------

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# ------------------------------------------------------------
# Optimize the model using the ONNX's native optimizer.
# ------------------------------------------------------------

from onnxruntime.transformers import optimizer

onnx_optim_model_path="gpt2onnx-opt.onnx"
optimized_model = optimizer.optimize_model(onnx_model_path,
                                           model_type='gpt2',
                                           num_heads=num_attention_heads,
                                           hidden_size=hidden_size,
                                           use_gpu=False,
                                           opt_level=1,
                                           verbose=True)
optimized_model.convert_float_to_float16()
optimized_model.save_model_to_file(onnx_optim_model_path)

# ------------------------------------------------------------
# Run text generation using the ONNX optimized model with profiling enabled.
# ------------------------------------------------------------

import onnx

optimized_onnx_model = onnx.load(onnx_optim_model_path)

tokenizer.pad_token = tokenizer.eos_token
input_ids, attention_mask, position_ids, empty_past = get_example_inputs(
    ['Here is some text to encode Hello World'], tokenizer, num_layer)

so = onnxruntime.SessionOptions()
so.enable_profiling = True
session = onnxruntime.InferenceSession(onnx_optim_model_path, so,
                                       providers=["CPUExecutionProvider"])
ort_inputs = {
    "input_ids": np.ascontiguousarray(input_ids.cpu().numpy()),
}
ort_outputs = session.run(None, ort_inputs)
prof_optimized = session.end_profiling()

# ------------------------------------------------------------
# # Profiling Data Clean Up and Visualization
# ------------------------------------------------------------

# ------------------------------------------------------------
# Copying and pasting here the original *mlprodict*'s `OnnxWholeSession` class code as the installation of this package is failing on the latest version of the Colab runtime.
# ------------------------------------------------------------

import json
import numpy

class OnnxWholeSession:
    """
    Runs the prediction for a single :epkg:`ONNX`,
    it lets the runtime handle the graph logic as well.

    :param onnx_data: :epkg:`ONNX` model or data
    :param runtime: runtime to be used, mostly :epkg:`onnxruntime`
    :param runtime_options: runtime options
    :param device: device, a string `cpu`, `cuda`, `cuda:0`...

    .. versionchanged:: 0.8
        Parameter *device* was added.
    """

    def __init__(self, onnx_data, runtime, runtime_options=None, device=None):
        if runtime not in ('onnxruntime1', 'onnxruntime1-cuda'):
            raise NotImplementedError(  # pragma: no cover
                f"runtime '{runtime}' is not implemented.")

        from onnxruntime import (  # delayed
            InferenceSession, SessionOptions, RunOptions,
            GraphOptimizationLevel)
        from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
            Fail as OrtFail, InvalidGraph as OrtInvalidGraph,
            InvalidArgument as OrtInvalidArgument,
            NotImplemented as OrtNotImplemented,
            RuntimeException as OrtRuntimeException)

        onnx_data0 = onnx_data
        if hasattr(onnx_data, 'SerializeToString'):
            onnx_data = onnx_data.SerializeToString()
        if isinstance(runtime_options, SessionOptions):
            sess_options = runtime_options
            session_options = None
            runtime_options = None
        else:
            session_options = (
                None if runtime_options is None
                else runtime_options.get('session_options', None))
            self.runtime = runtime
            sess_options = session_options or SessionOptions()
        self.run_options = RunOptions()
        self.run_options.log_severity_level = 3
        self.run_options.log_verbosity_level = 1

        if session_options is None:
            if runtime_options is not None:
                if runtime_options.get('disable_optimisation', False):
                    sess_options.graph_optimization_level = (  # pragma: no cover
                        GraphOptimizationLevel.ORT_ENABLE_ALL)
                if runtime_options.get('enable_profiling', True):
                    sess_options.enable_profiling = True
                if runtime_options.get('log_severity_level', 2) != 2:
                    v = runtime_options.get('log_severity_level', 2)
                    sess_options.log_severity_level = v
                    self.run_options.log_severity_level = v
        elif runtime_options is not None and 'enable_profiling' in runtime_options:
            raise RuntimeError(  # pragma: no cover
                "session_options and enable_profiling cannot be defined at the "
                "same time.")
        elif runtime_options is not None and 'disable_optimisation' in runtime_options:
            raise RuntimeError(  # pragma: no cover
                "session_options and disable_optimisation cannot be defined at the "
                "same time.")
        elif runtime_options is not None and 'log_severity_level' in runtime_options:
            raise RuntimeError(  # pragma: no cover
                "session_options and log_severity_level cannot be defined at the "
                "same time.")
        providers = ['CPUExecutionProvider']
        if runtime == 'onnxruntime1-cuda':
            providers = ['CUDAExecutionProvider'] + providers
        try:
            self.sess = InferenceSession(onnx_data, sess_options=sess_options,
                                         device=device, providers=providers)
        except (OrtFail, OrtNotImplemented, OrtInvalidGraph,
                OrtInvalidArgument, OrtRuntimeException, RuntimeError) as e:
            raise RuntimeError(
                "Unable to create InferenceSession due to '{}'\n{}.".format(e)) from e
        self.output_names = [_.name for _ in self.sess.get_outputs()]

    def run(self, inputs):
        """
        Computes the predictions.

        @param      inputs      dictionary *{variable, value}*
        @return                 list of outputs
        """
        v = next(iter(inputs.values()))
        if isinstance(v, (numpy.ndarray, dict)):
            try:
                return self.sess._sess.run(
                    self.output_names, inputs, self.run_options)
            except ValueError as e:
                raise ValueError(
                    "Issue running inference inputs=%r, expected inputs=%r."
                    "" % (
                        list(sorted(inputs)),
                        [i.name for i in self.sess.get_inputs()])) from e
        try:
            return self.sess._sess.run_with_ort_values(
                inputs, self.output_names, self.run_options)
        except RuntimeError:
            return self.sess._sess.run_with_ort_values(
                {k: v._get_c_value() for k, v in inputs.items()},
                self.output_names, self.run_options)

    @staticmethod
    def process_profiling(js):
        """
        Flattens json returned by onnxruntime profiling.

        :param js: json
        :return: list of dictionaries
        """
        rows = []
        for row in js:
            if 'args' in row and isinstance(row['args'], dict):
                for k, v in row['args'].items():
                    row[f'args_{k}'] = v
                del row['args']
            rows.append(row)
        return rows

    def get_profiling(self):
        """
        Returns the profiling informations.
        """
        prof = self.sess.end_profiling()
        with open(prof, 'r') as f:
            content = f.read()
        js = json.loads(content)
        return OnnxWholeSession.process_profiling(js)

# ------------------------------------------------------------
# Define a custom function to put the raw ONNX profiling data in a more friendly and useful format.
# ------------------------------------------------------------

import json
import pandas as pd

def clean_up_profiling_data(prof):
  with open(prof, "r") as f:
      js = json.load(f)
  df = pd.DataFrame(OnnxWholeSession.process_profiling(js))

  return df

# ------------------------------------------------------------
# Define a custom function to do several profiling data aggregations (group by operator type and calculate the total duration for each one, count the number of occurrences for each one (and order them by duration), calculate the percentage of the total inference time for each one) that would be used to build some visualizations.
# ------------------------------------------------------------

def transform_profiling_data_for_visualization(df):
  gr_dur = df[['dur', "args_op_name"]].groupby("args_op_name").sum().sort_values('dur')

  gr_n = df[['dur', "args_op_name"]].groupby("args_op_name").count().sort_values('dur')
  gr_n = gr_n.loc[gr_dur.index, :]

  gr_dur_perc = gr_dur / gr_dur['dur'].sum()

  return gr_dur, gr_n, gr_dur_perc

# ------------------------------------------------------------
# Transform the profiling data for the ONNX model.
# ------------------------------------------------------------

gr_dur, gr_n, gr_dur_perc = transform_profiling_data_for_visualization(clean_up_profiling_data(prof))

# ------------------------------------------------------------
# Create visualizations for the ONNX model profiling data.
# ------------------------------------------------------------

import plotly.express as px

fig = px.bar(gr_dur, x='dur',
             labels={
                     "dur": "Duration (ms)",
                     "args_op_name": "Operation type",
                 },
             title='Duration')
fig.show()

fig = px.bar(gr_n, x='dur',
             labels={
                     "dur": "Op count",
                     "args_op_name": "Operation type",
                 },
             title='Occurrences')
fig.show()

fig = px.bar(gr_dur_perc, x='dur',
             labels={
                     "dur": "Duration (%)",
                     "args_op_name": "Operation type",
                 },
             title='Proportion')
fig.show()

# ------------------------------------------------------------
# Transform the profiling data for the optimized ONNX model.
# ------------------------------------------------------------

gr_dur, gr_n, gr_dur_perc = transform_profiling_data_for_visualization(clean_up_profiling_data(prof_optimized))

# ------------------------------------------------------------
# Create visualizations for the optimized ONNX model profiling data.
# ------------------------------------------------------------

fig = px.bar(gr_dur, x='dur',
             labels={
                     "dur": "Duration (ms)",
                     "args_op_name": "Operation type",
                 },
             title='Duration')
fig.show()

fig = px.bar(gr_n, x='dur',
             labels={
                     "dur": "Op count",
                     "args_op_name": "Operation type",
                 },
             title='Occurrences')
fig.show()

fig = px.bar(gr_dur_perc, x='dur',
             labels={
                     "dur": "Duration (%)",
                     "args_op_name": "Operation type",
                 },
             title='Proportion')
fig.show()

