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
class KVCacheBenchmarkResult:
    """Immutable latency measurements for KV cache comparison."""

    latency_with_cache_ms: float
    latency_without_cache_ms: float
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


def measure_kv_cache_benchmark(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> KVCacheBenchmarkResult:
    """Benchmark autoregressive token generation with and without past KV tensors."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    def _timed_run(use_cache: bool) -> float:
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.pad_token_id,
        )
        for _ in range(WARMUP_RUNS):
            with torch.inference_mode():
                model.generate(**inputs, generation_config=config)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(BENCHMARK_RUNS):
            with torch.inference_mode():
                model.generate(**inputs, generation_config=config)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return (time.perf_counter() - start) / BENCHMARK_RUNS

    time_with_cache = _timed_run(use_cache=True) * 1000.0
    time_without_cache = _timed_run(use_cache=False) * 1000.0

    return KVCacheBenchmarkResult(
        latency_with_cache_ms=time_with_cache,
        latency_without_cache_ms=time_without_cache,
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
    rows = [
        (i, prompt, comp.strip()[:100] + "...") for i, (prompt, comp) in enumerate(zip(prompts, completions), start=1)
    ]
    console.print(create_table("Batched Generation Results (Left-Padded)", columns, rows))
    pause()


def render_kv_cache_table(res: KVCacheBenchmarkResult) -> None:
    """Render KV cache comparison table."""
    columns = [
        ("Configuration", STYLE_PRIMARY, "left"),
        ("Latency (ms)", STYLE_WARNING, "right"),
        ("Speedup Factor", STYLE_SUCCESS, "right"),
    ]
    rows = [
        ("With KV-Cache (use_cache=True)", f"{res.latency_with_cache_ms:.2f} ms", f"{res.speedup_ratio:.2f}x"),
        ("Without KV-Cache (use_cache=False)", f"{res.latency_without_cache_ms:.2f} ms", "1.00x (baseline)"),
    ]
    console.print(create_table("KV-Cache Impact Benchmark", columns, rows))
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

    # Step 5: KV-Cache Benchmark
    render_step(5, "Benchmarking Autoregressive KV-Cache Acceleration", icon="📊")
    with status_spinner(f"Running {BENCHMARK_RUNS} benchmark iterations with/without KV cache..."):
        kv_res = measure_kv_cache_benchmark(model, tokenizer, BASE_PROMPT)
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
