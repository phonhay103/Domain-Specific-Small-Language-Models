"""Optimizing SLM Inference with Hugging Face.

Companion script for Chapter 4 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Demonstrates device placement, single and batched generation, prompt design,
and the performance impact of KV-caching with GPT-2.
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Third-party
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Common functional & UI utilities
from common.functional import calculate_speedup
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
    render_code_block,
    render_device_info,
    render_step,
    render_takeaways,
    status_spinner,
)


# ---------------------------------------------------------------------------
# Immutable Domain Records & Constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StreamingComparisonResult:
    """Immutable throughput measurements for streaming generation."""

    with_cache_tokens_per_sec: float
    without_cache_tokens_per_sec: float
    with_cache_time_ms: float
    without_cache_time_ms: float
    speedup_ratio: float


MODEL_ID = "openai-community/gpt2"
BASE_PROMPT = "The best small language models are"
SENTIMENT_PROMPT = "Text: The movie was great and the food was delicious. Sentiment:"
CODE_PROMPT = 'def fibonacci(n):\n    """Return the nth Fibonacci number."""\n'

BATCH_PROMPTS: tuple[str, ...] = (
    "The secret to building great software is",
    "Once upon a time in a digital world,",
    "Artificial intelligence in medicine will",
)


BENCHMARK_PROMPT = (
    "Deep learning is a subset of machine learning, which is essentially a neural network with three or more layers. "
    "These neural networks attempt to simulate the behavior of the human brain—albeit far from matching its ability—allowing "
    "it to 'learn' from large amounts of data. While a neural network with a single layer can still make approximate "
    "predictions, additional hidden layers can help to optimize and refine for accuracy. Deep learning drives many "
    "artificial intelligence (AI) applications and services that improve automation, performing analytical and physical "
    "tasks without human intervention. Deep learning technology lies behind everyday products and services (such as digital "
    "assistants, voice-enabled TV remotes, and credit card fraud detection) as well as emerging technologies (such as self-driving cars). "
    "Neural networks, or artificial neural networks (ANNs), are comprised of node layers, containing an input layer, "
    "one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. "
    "If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. "
    "Otherwise, no data is passed along to the next layer. Deep learning models can be trained to recognize patterns "
    "in data, such as images, text, and sound, to make accurate predictions. Large language models (LLMs) are a type of deep learning model "
    "that can process and generate natural language. They are trained on massive text datasets, enabling them to understand "
    "grammar, semantics, and context, and to produce coherent text. Today, we will explore how small language models (SLMs) "
    "compare to their larger counterparts, looking at trade-offs in computational efficiency, memory footprint, "
    "fine-tuning speed, and generalizability across niche domain tasks. Specifically, we will look at:"
) * 2

BENCHMARK_MAX_NEW_TOKENS = 80

WARMUP_RUNS = 2
BENCHMARK_RUNS = 5
MAX_NEW_TOKENS = 64


# ---------------------------------------------------------------------------
# Pure Generation & Benchmarking Logic
# ---------------------------------------------------------------------------
def generate_text_pure(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    do_sample: bool = False,
    temperature: float = 1.0,
    use_cache: bool = True,
) -> str:
    """Pure generative inference wrapper: returns completion substring."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        use_cache=use_cache,
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.inference_mode():
        output_ids = model.generate(**inputs, generation_config=config)

    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return str(tokenizer.decode(new_tokens, skip_special_tokens=True))


def generate_batch_pure(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[str, ...]:
    """Pure batched inference using left-padding."""
    inputs = tokenizer(list(prompts), return_tensors="pt", padding=True).to(model.device)
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.inference_mode():
        output_ids = model.generate(**inputs, generation_config=config)

    return tuple(
        tokenizer.decode(output_ids[i, -max_new_tokens:], skip_special_tokens=True) for i in range(len(prompts))
    )


def run_streaming_comparison(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 80,
) -> StreamingComparisonResult:
    """Run real-time streaming token generation with and without KV-cache and measure throughput."""
    device = next(model.parameters()).device

    # -------------------------------------------------------------------------
    # 1. WITH KV-CACHE
    # -------------------------------------------------------------------------
    console.print("\n[bold green]▶ Streaming Generation WITH KV-Cache:[/bold green]")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    past_key_values = None
    input_ids = inputs["input_ids"]
    generated_tokens_with = []

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True)
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    generated_tokens_with.append(next_token.item())

    token_text = tokenizer.decode(next_token, skip_special_tokens=True)
    print(token_text, end="", flush=True)

    input_ids = next_token.unsqueeze(0)
    past_key_values = outputs.past_key_values

    start_time = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_tokens_with.append(next_token.item())

        token_text = tokenizer.decode(next_token, skip_special_tokens=True)
        print(token_text, end="", flush=True)

        input_ids = next_token.unsqueeze(0)
        past_key_values = outputs.past_key_values

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_with_cache = (time.perf_counter() - start_time) * 1000.0
    tokens_with_sec = len(generated_tokens_with) / (time_with_cache / 1000.0)

    console.print(f"\n[dim]↳ Completed WITH cache in {time_with_cache:.2f} ms ({tokens_with_sec:.2f} tok/s)[/dim]\n")

    # -------------------------------------------------------------------------
    # 2. WITHOUT KV-CACHE
    # -------------------------------------------------------------------------
    console.print("[bold red]▶ Streaming Generation WITHOUT KV-Cache (re-computing entire sequence):[/bold red]")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    generated_tokens_without = []

    start_time = time.perf_counter()
    for _ in range(max_new_tokens):
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, use_cache=False)

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_tokens_without.append(next_token.item())

        token_text = tokenizer.decode(next_token, skip_special_tokens=True)
        print(token_text, end="", flush=True)

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_without_cache = (time.perf_counter() - start_time) * 1000.0
    tokens_without_sec = len(generated_tokens_without) / (time_without_cache / 1000.0)

    console.print(
        f"\n[dim]↳ Completed WITHOUT cache in {time_without_cache:.2f} ms ({tokens_without_sec:.2f} tok/s)[/dim]\n"
    )

    return StreamingComparisonResult(
        with_cache_tokens_per_sec=tokens_with_sec,
        without_cache_tokens_per_sec=tokens_without_sec,
        with_cache_time_ms=time_with_cache,
        without_cache_time_ms=time_without_cache,
        speedup_ratio=calculate_speedup(time_without_cache, time_with_cache),
    )


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------


def render_batch_results_table(prompts: Sequence[str], completions: Sequence[str]) -> None:
    """Render batch generation results."""
    columns = [
        ("#", STYLE_NUMBER, "center"),
        ("Input Prompt", STYLE_PRIMARY, "left"),
        ("Generated Continuation", STYLE_TEXT, "left"),
    ]
    rows = [(i, prompt, comp.strip()) for i, (prompt, comp) in enumerate(zip(prompts, completions), start=1)]
    console.print(create_table("Batched Generation Results (Left-Padded)", columns, rows))
    pause()


def render_kv_cache_table(res: StreamingComparisonResult) -> None:
    """Render KV cache throughput comparison table."""
    columns = [
        ("Configuration", STYLE_PRIMARY, "left"),
        ("Decoding Throughput", STYLE_WARNING, "right"),
        ("Latency (ms)", STYLE_TEXT, "right"),
        ("Speedup Factor", STYLE_SUCCESS, "right"),
    ]
    rows = [
        (
            "With KV-Cache (use_cache=True)",
            f"{res.with_cache_tokens_per_sec:.1f} tok/s",
            f"{res.with_cache_time_ms:.1f} ms",
            f"{res.speedup_ratio:.2f}x",
        ),
        (
            "Without KV-Cache (use_cache=False)",
            f"{res.without_cache_tokens_per_sec:.1f} tok/s",
            f"{res.without_cache_time_ms:.1f} ms",
            "1.00x (baseline)",
        ),
    ]
    console.print(create_table("KV-Cache Real-Time Decoding Performance", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute SLM inference optimization pipeline."""
    render_banner(
        title="Optimizing Small Language Model Inference (Hugging Face)",
        subtitle="Chapter 4: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_ID,
            "Max New Tokens": str(MAX_NEW_TOKENS),
            "Benchmark Runs": str(BENCHMARK_RUNS),
        },
        icon="🚀",
    )

    # Step 1: Loading Model & Device Mapping
    render_step(1, "Loading Model & Device Placement", icon="📋")
    with status_spinner(f"Loading '{MODEL_ID}' with device_map='auto'..."):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        model.eval()

    render_device_info(model.device, model=model)

    # Step 2: Single-Prompt Text Generation
    render_step(2, "Executing Greedy Continuation on Base Prompt", icon="💬")
    completion = generate_text_pure(model, tokenizer, BASE_PROMPT)
    render_card(
        title="Greedy Autoregressive Output",
        content=f"[text.muted]Prompt:[/text.muted] {BASE_PROMPT}\n\n[status.success]Completion:[/status.success] [text.highlight]{completion.strip()}[/text.highlight]",
        icon="✨",
    )

    # Step 3: Structured Prompting (Few-Shot & Code)
    render_step(3, "Structured Prompting: Sentiment & Code Generation", icon="⚡")
    sentiment_out = generate_text_pure(model, tokenizer, SENTIMENT_PROMPT, max_new_tokens=8)
    render_card(
        title="Sentiment Extraction",
        content=f"[text.muted]Prompt:[/text.muted] {SENTIMENT_PROMPT}\n[status.warning]Extracted Sentiment:[/status.warning] [text.highlight]{sentiment_out.strip()}[/text.highlight]",
        icon="🎯",
    )

    code_out = generate_text_pure(model, tokenizer, CODE_PROMPT, max_new_tokens=48)
    render_code_block(CODE_PROMPT + code_out, language="python", title="Python Code Generation Output")

    # Step 4: Batched Inference
    render_step(4, "Batched Left-Padded Inference", icon="📦")
    batch_completions = generate_batch_pure(model, tokenizer, BATCH_PROMPTS)
    render_batch_results_table(BATCH_PROMPTS, batch_completions)

    # Step 5: Streaming Token Comparison
    render_step(5, "Real-Time Streaming Autoregressive Decoding & KV-Cache Comparison", icon="📊")
    kv_res = run_streaming_comparison(model, tokenizer, BENCHMARK_PROMPT, max_new_tokens=BENCHMARK_MAX_NEW_TOKENS)
    render_kv_cache_table(kv_res)

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Why KV-Cache is Essential",
                "In autoregressive generation, computing key/value projections for all previous tokens repeatedly scales quadratically O(N^2). Storing past KV tensors reduces computational complexity per new token to O(N).",
            ),
            (
                "Left-Padding Importance",
                "Decoder-only architectures must pad on the left because generation happens sequentially on the right. Right-padding would cause attention masks to misalign new token predictions.",
            ),
            (
                "Device Map Auto",
                "Automatically detects available CUDA devices and places tensor layers optimally across VRAM without manual .to(device) calls.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
