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
import sys
from pathlib import Path

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnx
import onnxruntime
import pandas as pd
import plotly.express as px
import torch
from onnxruntime.transformers import optimizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, GPT2LMHeadModel

from common.ui import (
    STYLE_NUMBER,
    STYLE_PRIMARY,
    STYLE_SECONDARY,
    STYLE_SUCCESS,
    STYLE_TEXT,
    STYLE_WARNING,
    console,
    create_table,
    pause,
    render_banner,
    render_card,
    render_step,
    render_takeaways,
    status_spinner,
)

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
    """

    def __init__(self, onnx_data, runtime, runtime_options=None, device=None):
        if runtime not in ("onnxruntime1", "onnxruntime1-cuda"):
            raise NotImplementedError(f"runtime '{runtime}' is not implemented.")

        from onnxruntime import (
            GraphOptimizationLevel,
            InferenceSession,
            RunOptions,
            SessionOptions,
        )
        from onnxruntime.capi._pybind_state import (
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
            session_options = None if runtime_options is None else runtime_options.get("session_options", None)
            self.runtime = runtime
            sess_options = session_options or SessionOptions()
        self.run_options = RunOptions()
        self.run_options.log_severity_level = 3
        self.run_options.log_verbosity_level = 1

        if session_options is None:
            if runtime_options is not None:
                if runtime_options.get("disable_optimisation", False):
                    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
                if runtime_options.get("enable_profiling", True):
                    sess_options.enable_profiling = True
                if runtime_options.get("log_severity_level", 2) != 2:
                    v = runtime_options.get("log_severity_level", 2)
                    sess_options.log_severity_level = v
                    self.run_options.log_severity_level = v
        elif runtime_options is not None and "enable_profiling" in runtime_options:
            raise RuntimeError("session_options and enable_profiling cannot be defined at the same time.")
        elif runtime_options is not None and "disable_optimisation" in runtime_options:
            raise RuntimeError("session_options and disable_optimisation cannot be defined at the same time.")
        elif runtime_options is not None and "log_severity_level" in runtime_options:
            raise RuntimeError("session_options and log_severity_level cannot be defined at the same time.")

        providers = ["CPUExecutionProvider"]
        if runtime == "onnxruntime1-cuda":
            providers = ["CUDAExecutionProvider", *providers]
        try:
            self.sess = InferenceSession(onnx_data, sess_options=sess_options, device=device, providers=providers)
        except (
            OrtFail,
            OrtNotImplemented,
            OrtInvalidGraph,
            OrtInvalidArgument,
            OrtRuntimeException,
            RuntimeError,
        ) as e:
            raise RuntimeError(f"Unable to create InferenceSession due to '{e}'\n{onnx_data0}.") from e
        self.output_names = [_.name for _ in self.sess.get_outputs()]

    def run(self, inputs):
        v = next(iter(inputs.values()))
        if isinstance(v, (np.ndarray, dict)):
            try:
                return self.sess._sess.run(self.output_names, inputs, self.run_options)
            except ValueError as e:
                raise ValueError(
                    f"Issue running inference inputs={sorted(inputs)!r}, expected inputs={[i.name for i in self.sess.get_inputs()]!r}."
                ) from e
        try:
            return self.sess._sess.run_with_ort_values(inputs, self.output_names, self.run_options)
        except RuntimeError:
            return self.sess._sess.run_with_ort_values(
                {k: v._get_c_value() for k, v in inputs.items()},
                self.output_names,
                self.run_options,
            )

    @staticmethod
    def process_profiling(js):
        rows = []
        for row in js:
            if "args" in row and isinstance(row["args"], dict):
                for k, v in row["args"].items():
                    row[f"args_{k}"] = v
                del row["args"]
            rows.append(row)
        return rows

    def get_profiling(self):
        prof = self.sess.end_profiling()
        with open(prof) as f:
            content = f.read()
        js = json.loads(content)
        return OnnxWholeSession.process_profiling(js)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_model_and_tokenizer() -> tuple[GPT2LMHeadModel, AutoTokenizer]:
    """Download GPT-2 small and its tokenizer from the HF Hub."""
    with console.status(f"[bold green]Loading {MODEL_NAME}..."):
        model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model.config.pad_token_id = tokenizer.eos_token_id
    console.print("[bold green]✔[/bold green] GPT-2 model and tokenizer loaded.")
    return model, tokenizer


def verify_vanilla_model(model: GPT2LMHeadModel, tokenizer: AutoTokenizer) -> None:
    """Run a quick forward pass to verify the model is working correctly."""
    inputs = tokenizer(SAMPLE_PROMPT, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    columns = [("Property", STYLE_PRIMARY, "left"), ("Dimension / Value", STYLE_SUCCESS, "left")]
    rows = [
        ("Input Tensor Shape", str(list(inputs["input_ids"].size()))),
        ("Output Logits Shape", str(list(logits.shape))),
    ]
    console.print(create_table("PyTorch Model Forward Verification", columns, rows))
    pause()


def export_to_onnx(model: GPT2LMHeadModel, tokenizer: AutoTokenizer) -> None:
    """Export the GPT-2 model to ONNX format at ONNX_MODEL_PATH."""
    input_ids: BatchEncoding = tokenizer(
        SAMPLE_PROMPT,
        add_special_tokens=True,
        return_attention_mask=False,
        return_tensors="pt",
    )
    for k, v in input_ids.items():
        input_ids[k] = v.type(dtype=torch.int32)
    input_tensor = input_ids["input_ids"]

    with console.status(f"[bold green]Exporting PyTorch model to ONNX ({ONNX_MODEL_PATH})..."):
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
    console.print(f"[bold green]✔[/bold green] ONNX model exported to [yellow]{ONNX_MODEL_PATH}[/yellow]")
    model.eval()


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
    with console.status(f"[bold green]Executing profiling session on {onnx_path}..."):
        so = onnxruntime.SessionOptions()
        so.enable_profiling = True
        session = onnxruntime.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        ort_inputs = {"input_ids": np.ascontiguousarray(input_ids.cpu().numpy())}
        session.run(None, ort_inputs)
        prof_file = session.end_profiling()
    console.print(f"[bold green]✔[/bold green] Profile trace recorded: [yellow]{prof_file}[/yellow]")
    return prof_file


def optimize_onnx_model(num_attention_heads: int, hidden_size: int) -> None:
    """Optimize the exported ONNX model and save it to ONNX_OPTIM_MODEL_PATH."""
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    with console.status("[bold green]Optimizing ONNX model graph (fusion & float16 conversion)..."):
        optimized_model = optimizer.optimize_model(
            ONNX_MODEL_PATH,
            model_type="gpt2",
            num_heads=num_attention_heads,
            hidden_size=hidden_size,
            use_gpu=False,
            opt_level=OPT_LEVEL,
            verbose=False,
        )
        optimized_model.convert_float_to_float16()
        optimized_model.save_model_to_file(ONNX_OPTIM_MODEL_PATH)
    console.print(f"[bold green]✔[/bold green] Optimized ONNX model saved to [yellow]{ONNX_OPTIM_MODEL_PATH}[/yellow]")


def clean_up_profiling_data(prof_path: str) -> pd.DataFrame:
    """Load raw ONNX profiling JSON and return a clean DataFrame."""
    with open(prof_path) as f:
        js = json.load(f)
    return pd.DataFrame(OnnxWholeSession.process_profiling(js))


def transform_profiling_data_for_visualization(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate profiling data by operator type for visualization."""
    gr_dur = df[["dur", "args_op_name"]].groupby("args_op_name").sum().sort_values("dur", ascending=False)
    gr_n = df[["dur", "args_op_name"]].groupby("args_op_name").count().loc[gr_dur.index, :]
    gr_dur_perc = (gr_dur / gr_dur["dur"].sum()) * 100
    return gr_dur, gr_n, gr_dur_perc


def display_profile_table(gr_dur: pd.DataFrame, gr_n: pd.DataFrame, gr_dur_perc: pd.DataFrame, title: str) -> None:
    """Display the top operations in a formatted Rich Table."""
    columns = [
        ("Operation Type", STYLE_PRIMARY, "left"),
        ("Total Duration (μs)", STYLE_WARNING, "right"),
        ("Executions", STYLE_NUMBER, "right"),
        ("Time Percentage (%)", STYLE_SUCCESS, "right"),
    ]
    rows = []
    for op in gr_dur.index[:10]:
        dur_val = gr_dur.loc[op, "dur"]
        count_val = gr_n.loc[op, "dur"]
        perc_val = gr_dur_perc.loc[op, "dur"]
        rows.append((str(op), f"{dur_val:,.1f}", f"{count_val:,}", f"{perc_val:.2f}%"))

    console.print(create_table(title, columns, rows))
    pause()


def visualize_profiling(
    gr_dur: pd.DataFrame, gr_n: pd.DataFrame, gr_dur_perc: pd.DataFrame, title_prefix: str = ""
) -> None:
    """Render Plotly bar charts for duration, occurrences, and proportion."""
    fig = px.bar(
        gr_dur.sort_values("dur"),
        x="dur",
        labels={"dur": "Duration (μs)", "args_op_name": "Operation type"},
        title=f"{title_prefix}Duration",
    )
    fig.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Orchestrate ONNX export, optimization, profiling, and visualization."""
    render_banner(
        title="Profiling & Optimizing ONNX Transformer Models",
        subtitle="Chapter 10: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_NAME,
            "Opset Version": str(OPSET_VERSION),
            "Optimization Level": str(OPT_LEVEL),
        },
        icon="🚀",
    )

    # Step 1: Loading and Verifying PyTorch Model
    render_step(1, "Loading and Verifying PyTorch Model", icon="📋")
    model, tokenizer = load_model_and_tokenizer()
    verify_vanilla_model(model, tokenizer)

    # Step 2: Exporting PyTorch Model to ONNX Graph
    render_step(2, "Exporting PyTorch Model to ONNX Graph", icon="⚙️")
    export_to_onnx(model, tokenizer)

    num_attention_heads = model.config.n_head
    hidden_size = model.config.n_embd

    # Step 3: Profiling Base ONNX Model
    render_step(3, "Profiling Base ONNX Model Operators", icon="⏱️")
    tokenizer.pad_token = tokenizer.eos_token
    input_ids, _, _, _ = _get_example_inputs_with_config([SAMPLE_PROMPT], tokenizer, model.config)
    prof = run_onnx_profiling(ONNX_MODEL_PATH, input_ids)

    # Step 4: Optimizing ONNX Graph & Profiling
    render_step(4, "Optimizing ONNX Graph (Kernel Fusion) & Re-Profiling", icon="⚡")
    optimize_onnx_model(num_attention_heads, hidden_size)
    prof_optimized = run_onnx_profiling(ONNX_OPTIM_MODEL_PATH, input_ids)

    # Step 5: Profiling Results Breakdown
    render_step(5, "Analyzing Operator Latency Breakdown & Hotspots", icon="📊")
    gr_dur, gr_n, gr_dur_perc = transform_profiling_data_for_visualization(clean_up_profiling_data(prof))
    display_profile_table(gr_dur, gr_n, gr_dur_perc, "Base ONNX Profile (Top Operators by Execution Time)")

    gr_dur_opt, gr_n_opt, gr_dur_perc_opt = transform_profiling_data_for_visualization(
        clean_up_profiling_data(prof_optimized)
    )
    display_profile_table(
        gr_dur_opt, gr_n_opt, gr_dur_perc_opt, "Optimized ONNX Profile (Top Operators by Execution Time)"
    )

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "ONNX Runtime Session Profiling",
                "Setting enable_profiling = True generates microsecond-accurate JSON execution traces of every individual operator node.",
            ),
            (
                "Operator Fusion Impact",
                "Optimization transforms dozens of standalone MatMul, Add, and Softmax calls into a single fused Attention or FastGELU kernel, slashing operator invocation counts.",
            ),
            (
                "Bottleneck Identification",
                "Profiling tables quickly highlight whether latency is dominated by matrix multiplication or memory-bound elementwise operations.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
