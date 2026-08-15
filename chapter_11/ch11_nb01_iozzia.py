"""
Small Language Model Offline Serving with vLLM.

Companion script for chapter 11 of "Domain Specific LLMs in Action"
by Guglielmo Iozzia (Manning Publications, 2024).

Demonstrates serving GPT-2 small and Microsoft's Phi-3 mini 4k Instruct
through the vLLM API, including text generation, chat, batch inference,
custom chat templates, and GGUF format support.
The same code applies to any Open Source LLM. Hardware acceleration required.

Setup (run once before executing this script):
    # pip install vllm
    # wget https://raw.githubusercontent.com/chujiezheng/chat_templates/main/chat_templates/phi-3.jinja
    # pip uninstall -y torchao  (interferes with vLLM GGUF inference)
"""

import gc

import torch
from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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

EXAMPLE_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hi! How can I assist you today?"},
    {"role": "user", "content": "Write an essay about the Monte Carlo Tree Search algorithm."},
]

GGUF_PROMPTS = [
    "How to explain Internet for a medieval knight?",
    "What's the future of AI?",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def print_outputs(outputs) -> None:
    """Print vLLM generation outputs in a readable format."""
    print("=" * 80)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)


def free_gpu_memory(model) -> None:
    """Delete a model and release GPU memory."""
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Text generation with GPT-2 via vLLM
# ---------------------------------------------------------------------------

def run_gpt2_text_generation() -> None:
    """Load GPT-2 small via vLLM and generate text from a sample prompt."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # vLLM downloads config, then checkpoints and tokenizer, profiles memory,
    # initialises the KV cache, and warms up the model before returning.
    vllm_model = LLM(GPT2_MODEL_ID, device=device, enforce_eager=True)

    output = vllm_model.generate(TEXT_GEN_PROMPT)
    for item in output:
        generated_text = item.outputs[0].text
        print(generated_text)

    free_gpu_memory(vllm_model)


# ---------------------------------------------------------------------------
# Chat with Phi-3 mini via vLLM
# ---------------------------------------------------------------------------

def run_phi3_chat() -> LLM:
    """Load Phi-3 mini 4k Instruct, run single and batch chat, return the engine."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm = LLM(model=PHI3_MODEL_ID, device=device, dtype=torch.half)
    sampling_params = SamplingParams(temperature=CHAT_TEMPERATURE)

    # Single-turn chat
    outputs = llm.chat(EXAMPLE_CONVERSATION, sampling_params=sampling_params, use_tqdm=False)
    print_outputs(outputs)

    # Batch chat (same conversation repeated 10 times)
    conversations = [EXAMPLE_CONVERSATION for _ in range(10)]
    outputs = llm.chat(messages=conversations, sampling_params=sampling_params, use_tqdm=True)
    print_outputs(outputs)

    return llm


def run_phi3_chat_with_template(llm: LLM) -> None:
    """Re-run Phi-3 chat using a custom Jinja chat template."""
    sampling_params = SamplingParams(temperature=CHAT_TEMPERATURE)

    with open(PHI3_CHAT_TEMPLATE_PATH, "r") as f:
        chat_template = f.read()

    conversations = [EXAMPLE_CONVERSATION for _ in range(10)]
    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        use_tqdm=False,
        chat_template=chat_template,
    )
    print_outputs(outputs)


# ---------------------------------------------------------------------------
# GGUF inference with Phi-3 mini via vLLM
# ---------------------------------------------------------------------------

def run_gguf_chat() -> None:
    """Download the Phi-3 GGUF model and run chat inference with vLLM."""
    model_path = hf_hub_download(PHI3_GGUF_REPO_ID, filename=PHI3_GGUF_FILENAME)
    sampling_params = SamplingParams(temperature=GGUF_TEMPERATURE, max_tokens=GGUF_MAX_TOKENS)

    # Wrap each prompt string in the user-turn message format
    prompts = [[{"role": "user", "content": p}] for p in GGUF_PROMPTS]

    llm = LLM(model=model_path, tokenizer=PHI3_TOKENIZER_ID)
    outputs = llm.chat(prompts, sampling_params)
    print_outputs(outputs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate GPT-2 text generation, Phi-3 chat, template chat, and GGUF chat."""
    run_gpt2_text_generation()

    llm = run_phi3_chat()
    run_phi3_chat_with_template(llm)
    free_gpu_memory(llm)

    run_gguf_chat()


if __name__ == "__main__":
    main()
