"""
Profiling ONNX Models.

Companion script for chapter 10 of "Domain Specific LLMs in Action"
by Guglielmo Iozzia (Manning Publications, 2024).

Profiles and analyses performance insights for a GPT-2 small model
after conversion to ONNX format and optimization. The profiling
visualization code is generic for any ML/DL ONNX model. No GPU needed.

Setup (run once before executing this script):
    # pip install onnx onnxruntime
"""

import json
import logging

import numpy
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, GPT2LMHeadModel

import onnx
import onnxruntime
from onnxruntime.transformers import optimizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "openai-community/gpt2"
SAMPLE_PROMPT = "Here is some text to encode Hello World"
ONNX_MODEL_PATH = "gpt2onnx.onnx"
ONNX_OPTIM_MODEL_PATH = "gpt2onnx-opt.onnx"
OPSET_VERSION = 18
OPT_LEVEL = 1


# ---------------------------------------------------------------------------
# OnnxWholeSession (inlined from mlprodict; installation fails on newer runtimes)
# ---------------------------------------------------------------------------

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
        if runtime not in ("onnxruntime1", "onnxruntime1-cuda"):
            raise NotImplementedError(  # pragma: no cover
                f"runtime '{runtime}' is not implemented."
            )

        from onnxruntime import (  # delayed
            GraphOptimizationLevel,
            InferenceSession,
            RunOptions,
            SessionOptions,
        )
        from onnxruntime.capi._pybind_state import (  # pylint: disable=E0611
            Fail as OrtFail,
            InvalidArgument as OrtInvalidArgument,
            InvalidGraph as OrtInvalidGraph,
            NotImplemented as OrtNotImplemented,
            RuntimeException as OrtRuntimeException,
        )

        onnx_data0 = onnx_data
        if hasattr(onnx_data, "SerializeToString"):
            onnx_data = onnx_data.SerializeToString()
        if isinstance(runtime_options, SessionOptions):
            sess_options = runtime_options
            session_options = None
            runtime_options = None
        else:
            session_options = (
                None if runtime_options is None
                else runtime_options.get("session_options", None)
            )
            self.runtime = runtime
            sess_options = session_options or SessionOptions()
        self.run_options = RunOptions()
        self.run_options.log_severity_level = 3
        self.run_options.log_verbosity_level = 1

        if session_options is None:
            if runtime_options is not None:
                if runtime_options.get("disable_optimisation", False):
                    sess_options.graph_optimization_level = (  # pragma: no cover
                        GraphOptimizationLevel.ORT_ENABLE_ALL
                    )
                if runtime_options.get("enable_profiling", True):
                    sess_options.enable_profiling = True
                if runtime_options.get("log_severity_level", 2) != 2:
                    v = runtime_options.get("log_severity_level", 2)
                    sess_options.log_severity_level = v
                    self.run_options.log_severity_level = v
        elif runtime_options is not None and "enable_profiling" in runtime_options:
            raise RuntimeError(  # pragma: no cover
                "session_options and enable_profiling cannot be defined at the same time."
            )
        elif runtime_options is not None and "disable_optimisation" in runtime_options:
            raise RuntimeError(  # pragma: no cover
                "session_options and disable_optimisation cannot be defined at the same time."
            )
        elif runtime_options is not None and "log_severity_level" in runtime_options:
            raise RuntimeError(  # pragma: no cover
                "session_options and log_severity_level cannot be defined at the same time."
            )

        providers = ["CPUExecutionProvider"]
        if runtime == "onnxruntime1-cuda":
            providers = ["CUDAExecutionProvider"] + providers
        try:
            self.sess = InferenceSession(
                onnx_data, sess_options=sess_options, device=device, providers=providers
            )
        except (
            OrtFail, OrtNotImplemented, OrtInvalidGraph,
            OrtInvalidArgument, OrtRuntimeException, RuntimeError,
        ) as e:
            raise RuntimeError(
                f"Unable to create InferenceSession due to '{e}'\n{onnx_data0}."
            ) from e
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
                return self.sess._sess.run(self.output_names, inputs, self.run_options)
            except ValueError as e:
                raise ValueError(
                    "Issue running inference inputs=%r, expected inputs=%r."
                    % (list(sorted(inputs)), [i.name for i in self.sess.get_inputs()])
                ) from e
        try:
            return self.sess._sess.run_with_ort_values(
                inputs, self.output_names, self.run_options
            )
        except RuntimeError:
            return self.sess._sess.run_with_ort_values(
                {k: v._get_c_value() for k, v in inputs.items()},
                self.output_names,
                self.run_options,
            )

    @staticmethod
    def process_profiling(js):
        """
        Flattens json returned by onnxruntime profiling.

        :param js: json
        :return: list of dictionaries
        """
        rows = []
        for row in js:
            if "args" in row and isinstance(row["args"], dict):
                for k, v in row["args"].items():
                    row[f"args_{k}"] = v
                del row["args"]
            rows.append(row)
        return rows

    def get_profiling(self):
        """Returns the profiling information."""
        prof = self.sess.end_profiling()
        with open(prof, "r") as f:
            content = f.read()
        js = json.loads(content)
        return OnnxWholeSession.process_profiling(js)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_and_tokenizer() -> tuple[GPT2LMHeadModel, AutoTokenizer]:
    """Download GPT-2 small and its tokenizer from the HF Hub."""
    model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def verify_vanilla_model(model: GPT2LMHeadModel, tokenizer: AutoTokenizer) -> None:
    """Run a quick forward pass to verify the model is working correctly."""
    inputs = tokenizer(SAMPLE_PROMPT, return_tensors="pt")
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


def export_to_onnx(model: GPT2LMHeadModel, tokenizer: AutoTokenizer) -> None:
    """Export the GPT-2 model to ONNX format at ONNX_MODEL_PATH."""
    input_ids: BatchEncoding = tokenizer(
        SAMPLE_PROMPT, add_special_tokens=True,
        return_attention_mask=False, return_tensors="pt",
    )
    for k, v in input_ids.items():
        input_ids[k] = v.type(dtype=torch.int32)
    input_tensor = input_ids["input_ids"]

    torch.onnx.export(
        model,
        f=ONNX_MODEL_PATH,
        args=(input_tensor,),
        input_names=["input_ids"],
        output_names=["logits"],
        quantization=False,
        var_output_seq=True,
        do_constant_folding=True,
        opset_version=OPSET_VERSION,
    )
    model.eval()


def get_example_inputs(prompt_text, tokenizer, num_layer: int, device: str = "cpu"):
    """Prepare tokenized inputs (with empty KV-cache placeholders) for ONNX inference."""
    num_attention_heads = tokenizer.model_max_length  # set externally via closure
    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)

    input_ids = torch.tensor(encodings_dict["input_ids"], dtype=torch.int32)
    attention_mask = torch.tensor(encodings_dict["attention_mask"], dtype=torch.int32)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(position_ids < 0, 0)
    position_ids = position_ids.to(torch.int32)

    empty_past = []
    batch_size = input_ids.size(0)
    return input_ids, attention_mask, position_ids, empty_past


def _get_example_inputs_with_config(prompt_text, tokenizer, model_config, device: str = "cpu"):
    """Prepare tokenized inputs using explicit model config values."""
    num_attention_heads = model_config.n_head
    hidden_size = model_config.n_embd
    num_layer = model_config.n_layer

    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)
    input_ids = torch.tensor(encodings_dict["input_ids"], dtype=torch.int32)
    attention_mask = torch.tensor(encodings_dict["attention_mask"], dtype=torch.int32)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(position_ids < 0, 0)
    position_ids = position_ids.to(torch.int32)

    empty_past = []
    batch_size = input_ids.size(0)
    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
    for _ in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))

    return input_ids, attention_mask, position_ids, empty_past


def run_onnx_profiling(onnx_path: str, input_ids) -> str:
    """Run ONNX inference with profiling enabled; return the profiling JSON file path."""
    so = onnxruntime.SessionOptions()
    so.enable_profiling = True
    session = onnxruntime.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    ort_inputs = {"input_ids": np.ascontiguousarray(input_ids.cpu().numpy())}
    session.run(None, ort_inputs)
    return session.end_profiling()


def optimize_onnx_model(num_attention_heads: int, hidden_size: int) -> None:
    """Optimize the exported ONNX model and save it to ONNX_OPTIM_MODEL_PATH."""
    # Set logging to INFO so applied optimizations are visible
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    optimized_model = optimizer.optimize_model(
        ONNX_MODEL_PATH,
        model_type="gpt2",
        num_heads=num_attention_heads,
        hidden_size=hidden_size,
        use_gpu=False,
        opt_level=OPT_LEVEL,
        verbose=True,
    )
    optimized_model.convert_float_to_float16()
    optimized_model.save_model_to_file(ONNX_OPTIM_MODEL_PATH)


def clean_up_profiling_data(prof_path: str) -> pd.DataFrame:
    """Load raw ONNX profiling JSON and return a clean DataFrame."""
    with open(prof_path, "r") as f:
        js = json.load(f)
    return pd.DataFrame(OnnxWholeSession.process_profiling(js))


def transform_profiling_data_for_visualization(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate profiling data by operator type for visualization.

    Returns:
        gr_dur: total duration per op type (sorted ascending)
        gr_n: occurrence count per op type (same order as gr_dur)
        gr_dur_perc: fraction of total inference time per op type
    """
    gr_dur = df[["dur", "args_op_name"]].groupby("args_op_name").sum().sort_values("dur")
    gr_n = df[["dur", "args_op_name"]].groupby("args_op_name").count().sort_values("dur")
    gr_n = gr_n.loc[gr_dur.index, :]
    gr_dur_perc = gr_dur / gr_dur["dur"].sum()
    return gr_dur, gr_n, gr_dur_perc


def visualize_profiling(gr_dur: pd.DataFrame, gr_n: pd.DataFrame, gr_dur_perc: pd.DataFrame, title_prefix: str = "") -> None:
    """Render three Plotly bar charts for duration, occurrences, and proportion."""
    fig = px.bar(
        gr_dur, x="dur",
        labels={"dur": "Duration (ms)", "args_op_name": "Operation type"},
        title=f"{title_prefix}Duration",
    )
    fig.show()

    fig = px.bar(
        gr_n, x="dur",
        labels={"dur": "Op count", "args_op_name": "Operation type"},
        title=f"{title_prefix}Occurrences",
    )
    fig.show()

    fig = px.bar(
        gr_dur_perc, x="dur",
        labels={"dur": "Duration (%)", "args_op_name": "Operation type"},
        title=f"{title_prefix}Proportion",
    )
    fig.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate ONNX export, optimization, profiling, and visualization."""
    model, tokenizer = load_model_and_tokenizer()
    verify_vanilla_model(model, tokenizer)
    export_to_onnx(model, tokenizer)

    # Collect model config values needed for input preparation
    num_layer = model.config.n_layer
    num_attention_heads = model.config.n_head
    hidden_size = model.config.n_embd

    # Profile the base ONNX model
    tokenizer.pad_token = tokenizer.eos_token
    input_ids, _, _, _ = _get_example_inputs_with_config(
        [SAMPLE_PROMPT], tokenizer, model.config,
    )
    prof = run_onnx_profiling(ONNX_MODEL_PATH, input_ids)

    # Optimize and profile the optimized ONNX model
    optimize_onnx_model(num_attention_heads, hidden_size)
    _ = onnx.load(ONNX_OPTIM_MODEL_PATH)  # validate the optimized model loads correctly
    input_ids, _, _, _ = _get_example_inputs_with_config(
        [SAMPLE_PROMPT], tokenizer, model.config,
    )
    prof_optimized = run_onnx_profiling(ONNX_OPTIM_MODEL_PATH, input_ids)

    # Visualize profiling results for both models
    gr_dur, gr_n, gr_dur_perc = transform_profiling_data_for_visualization(
        clean_up_profiling_data(prof)
    )
    visualize_profiling(gr_dur, gr_n, gr_dur_perc, title_prefix="Base ONNX – ")

    gr_dur, gr_n, gr_dur_perc = transform_profiling_data_for_visualization(
        clean_up_profiling_data(prof_optimized)
    )
    visualize_profiling(gr_dur, gr_n, gr_dur_perc, title_prefix="Optimized ONNX – ")


if __name__ == "__main__":
    main()
