"""Small Language Model Offline Serving with vLLM.

Companion script for chapter 11 of "Domain Specific LLMs in Action"
by Guglielmo Iozzia (Manning Publications, 2024).

Demonstrates serving GPT-2 small and Microsoft's Phi-3 mini 4k Instruct
through the vLLM API, including text generation, chat, continuous batch inference,
custom chat templates, and GGUF format support.
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import gc
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Third-party
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# Common functional & UI utilities
from common.ui import (
    STYLE_INDEX,
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
# Immutable Domain Records & Constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChatMessage:
    """Immutable chat dialogue turn."""

    role: str
    content: str


GPT2_MODEL_ID = "openai-community/gpt2"
PHI3_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
PHI3_GGUF_REPO_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"
PHI3_GGUF_FILENAME = "Phi-3-mini-4k-instruct-q4.gguf"
PHI3_TOKENIZER_ID = "microsoft/Phi-3-mini-4k-instruct"
PHI3_CHAT_TEMPLATE_PATH = "phi-3.jinja"

TEXT_GEN_PROMPT = "Once upon a time in a land far away"
CHAT_TEMPERATURE = 0.5
GGUF_TEMPERATURE = 0.0
GGUF_MAX_TOKENS = 128

EXAMPLE_CONVERSATION: tuple[ChatMessage, ...] = (
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="Hi"),
    ChatMessage(role="assistant", content="Hi! How can I assist you today?"),
    ChatMessage(role="user", content="Write an essay about the Monte Carlo Tree Search algorithm."),
)

GGUF_PROMPTS: tuple[str, ...] = (
    "How to explain Internet for a medieval knight?",
    "What's the future of AI?",
)


# ---------------------------------------------------------------------------
# Pure Helpers
# ---------------------------------------------------------------------------
def conversation_to_dicts(convo: Sequence[ChatMessage]) -> list[dict[str, str]]:
    """Pure transformation of immutable messages into list-of-dicts."""
    return [{"role": m.role, "content": m.content} for m in convo]


def free_gpu_memory(model: Any) -> None:
    """Release GPU memory cleanly."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute vLLM high-throughput offline inference pipeline."""
    render_banner(
        title="Offline Small Language Model Serving with vLLM",
        subtitle="Chapter 11: Domain-Specific Small Language Models",
        metadata={
            "Engines": f"{GPT2_MODEL_ID}, {PHI3_MODEL_ID}",
            "PagedAttention": "Enabled",
            "Quantization Formats": "BF16, FP16, GGUF Q4",
        },
        icon="🚀",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1: GPT-2 Text Generation with vLLM
    render_step(1, "Continuous PagedAttention Serving for GPT-2", icon="📋")
    try:
        from vllm import LLM, SamplingParams

        with status_spinner(f"Initializing vLLM engine for '{GPT2_MODEL_ID}' on {device}..."):
            vllm_gpt2 = LLM(GPT2_MODEL_ID, device=device, enforce_eager=True)

        with status_spinner(f"Generating completion for prompt: '{TEXT_GEN_PROMPT}'..."):
            output = vllm_gpt2.generate(TEXT_GEN_PROMPT)
            gen_text = output[0].outputs[0].text.strip()

        render_card(
            title="GPT-2 Output",
            content=f"[text.muted]Prompt:[/text.muted] {TEXT_GEN_PROMPT}\n\n[status.success]Completion:[/status.success]\n{gen_text}",
            icon="📄",
        )
        free_gpu_memory(vllm_gpt2)

        # Step 2: Phi-3 Mini Chat & Batch Inference
        render_step(2, "Phi-3 Mini Multi-Turn Chat & Parallel Batching", icon="💬")
        with status_spinner(f"Loading '{PHI3_MODEL_ID}' in FP16 with vLLM..."):
            llm_phi3 = LLM(model=PHI3_MODEL_ID, device=device, dtype=torch.half)
            sampling_params = SamplingParams(temperature=CHAT_TEMPERATURE)

        convo_dicts = conversation_to_dicts(EXAMPLE_CONVERSATION)
        with status_spinner("Running single chat turn..."):
            single_out = llm_phi3.chat(convo_dicts, sampling_params=sampling_params, use_tqdm=False)
            response_text = single_out[0].outputs[0].text.strip()

        render_card("Phi-3 Mini Single Chat", response_text, icon="✨")

        # Step 3: Batch Chat Throughput
        render_step(3, "Evaluating Parallel Continuous Batching Throughput", icon="⚡")
        batch_convos = [convo_dicts for _ in range(10)]
        with status_spinner("Executing 10 parallel batched conversations..."):
            batch_outputs = llm_phi3.chat(messages=batch_convos, sampling_params=sampling_params, use_tqdm=False)

        render_card(
            "Batch Chat Status",
            f"Processed [text.highlight]{len(batch_outputs)}[/text.highlight] parallel conversations concurrently.",
            icon="✔",
        )
        free_gpu_memory(llm_phi3)

        # Step 4: GGUF Format Serving
        render_step(4, "Serving GGUF Quantized Binary with vLLM", icon="📦")
        with status_spinner(f"Downloading GGUF artifact from '{PHI3_GGUF_REPO_ID}'..."):
            model_path = hf_hub_download(PHI3_GGUF_REPO_ID, filename=PHI3_GGUF_FILENAME)

        with status_spinner("Serving GGUF Q4 binary with vLLM PagedAttention..."):
            gguf_params = SamplingParams(temperature=GGUF_TEMPERATURE, max_tokens=GGUF_MAX_TOKENS)
            gguf_prompts = [[{"role": "user", "content": p}] for p in GGUF_PROMPTS]
            llm_gguf = LLM(model=model_path, tokenizer=PHI3_TOKENIZER_ID)
            gguf_outputs = llm_gguf.chat(gguf_prompts, gguf_params)

        for i, out in enumerate(gguf_outputs, 1):
            render_card(
                f"GGUF Query #{i}",
                f"[text.muted]Prompt:[/text.muted] {GGUF_PROMPTS[i - 1]}\n\n[status.success]Response:[/status.success]\n{out.outputs[0].text.strip()}",
                icon="🔬",
            )
        free_gpu_memory(llm_gguf)
    except (ImportError, Exception):
        render_card(
            "Environment Note", "vLLM with CUDA backend required for native PagedAttention execution.", icon="ℹ️"
        )

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "PagedAttention Memory Management",
                "vLLM treats the Key-Value (KV) cache as virtual memory pages, reducing KV memory waste from ~60-80% down to under 4%.",
            ),
            (
                "Continuous Batching",
                "Requests entering at different times are dynamically interleaved into forward passes without waiting for prior sequences to finish.",
            ),
            (
                "Multi-Format Engine",
                "Supports unquantized FP16/BF16 checkpoints, AWQ, GPTQ, and GGUF quantized binaries under a unified OpenAI-compatible API.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
