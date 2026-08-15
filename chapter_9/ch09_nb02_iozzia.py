"""
Using SmoothQuant on OPT large models.

Companion script for chapter 9 of "Domain Specific LLMs in Action"
by Guglielmo Iozzia (Manning Publications, 2024).

Shows that for LLMs with 6B+ parameters, systematic outliers in a model's
activations degrade accuracy after naive quantization, and that applying
SmoothQuant (https://github.com/mit-han-lab/smoothquant) mitigates that risk.
Targets Meta AI's OPT 6.7B model; the same approach applies to other models.
Requires hardware acceleration (GPU).

Setup (run once before executing this script):
    # pip install --force-reinstall datasets
    # pip install git+https://github.com/mit-han-lab/smoothquant.git
    # mkdir ./act_scales && wget -P ./act_scales \
    #   https://huggingface.co/mit-han-lab/smoothquant-scales/resolve/main/opt-6.7b.pt
"""

import torch
from datasets import load_dataset
from smoothquant.fake_quant import W8A8Linear
from smoothquant.smooth import smooth_lm
from transformers import GPT2Tokenizer
from transformers.models.opt.modeling_opt import (
    OPTAttention,
    OPTDecoderLayer,
    OPTForCausalLM,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "facebook/opt-6.7b"
DATASET_NAME = "cimec/lambada"
DATASET_SPLIT = "validation[:1000]"
ACT_SCALES_PATH = "./act_scales/opt-6.7b.pt"
OFFLOAD_FOLDER = "."
SMOOTHQUANT_ALPHA = 0.5


# ---------------------------------------------------------------------------
# Model quantization
# ---------------------------------------------------------------------------

def quantize_model(
    model: OPTForCausalLM,
    weight_quant: str = "per_tensor",
    act_quant: str = "per_tensor",
    quantize_bmm_input: bool = True,
) -> OPTForCausalLM:
    """Quantize a model's weights and activations to INT8 precision."""
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.out_proj = W8A8Linear.from_float(
                m.out_proj, weight_quant=weight_quant, act_quant=act_quant,
            )
    return model


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Evaluate an LLM on a tokenized dataset using last-token accuracy."""

    def __init__(self, dataset, tokenizer, device: str) -> None:
        self.device = device
        self.tokenizer = tokenizer

        def tokenize_function(examples):
            return self.tokenizer(examples["text"])

        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format(type="torch", columns=["input_ids"])
        self.dataset = dataset

    @torch.no_grad()
    def evaluate(self, model) -> float:
        """Return accuracy (fraction of correct last-token predictions)."""
        model.eval()
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch["input_ids"].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        return hit / total


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def build_evaluator() -> Evaluator:
    """Download the LAMBADA subset and the OPT tokenizer; return an Evaluator."""
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    evaluator = Evaluator(dataset, tokenizer, "cuda")
    print("Dataset loaded and Evaluator initialized successfully.")
    return evaluator


def evaluate_fp16(evaluator: Evaluator) -> float:
    """Download OPT 6.7B in FP16, evaluate, and return accuracy."""
    model_fp16 = OPTForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder=OFFLOAD_FOLDER,
    )
    model_fp16.eval()
    acc = evaluator.evaluate(model_fp16)
    print(f"Original model (fp16) accuracy: {acc}")
    return acc


def evaluate_naive_w8a8(evaluator: Evaluator) -> float:
    """Reload OPT 6.7B in FP16, apply naive W8A8 quantization, and evaluate."""
    model_fp16 = OPTForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder=OFFLOAD_FOLDER,
    )
    model_w8a8 = quantize_model(model_fp16)
    print(model_w8a8)
    acc = evaluator.evaluate(model_w8a8)
    print(f"Naive W8A8 quantized model accuracy: {acc}")
    return acc


def evaluate_smoothquant_w8a8(evaluator: Evaluator) -> float:
    """Apply SmoothQuant then W8A8 quantization to OPT 6.7B and evaluate.

    Requires act_scales downloaded to ACT_SCALES_PATH beforehand.
    The accuracy of this version should be comparable to the FP16 baseline,
    while the naive W8A8 model may drop up to 40%.
    """
    model_fp16 = OPTForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder=OFFLOAD_FOLDER,
    )
    act_scales = torch.load(ACT_SCALES_PATH)
    smooth_lm(model_fp16, act_scales, SMOOTHQUANT_ALPHA)
    model_sq = quantize_model(model_fp16)
    print(model_sq)
    model_sq.eval()
    acc = evaluator.evaluate(model_sq)
    print(f"SmoothQuant W8A8 quantized model accuracy: {acc}")
    return acc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate FP16, naive W8A8, and SmoothQuant W8A8 accuracy comparison."""
    evaluator = build_evaluator()
    evaluate_fp16(evaluator)
    evaluate_naive_w8a8(evaluator)
    evaluate_smoothquant_w8a8(evaluator)


if __name__ == "__main__":
    main()
