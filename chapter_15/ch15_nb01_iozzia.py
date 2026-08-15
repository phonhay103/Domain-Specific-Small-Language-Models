"""
AutoThink Example with OptiLLM and Qwen 2.5 0.5B Instruct.

Companion script for chapter 15 of "Domain-Specific Small Language Models"
by Guglielmo Iozzia, Manning Publications, 2025.

Demonstrates the AutoThink technique in OptiLLM with the Qwen 2.5 0.5B
Instruct model, and benchmarks it on the OptiLLMBench / GSM8k dataset.
GPU is recommended.

=============================================================================
EDUCATIONAL CONCEPTS DEMONSTRATED:
1. Test-Time Compute Scaling:
   - Enhances reasoning quality of small models without altering model weights by allocating more compute at generation time (lookahead, verification, multi-path search).
2. AutoThink & ThinkDeeper:
   - AutoThink dynamically analyzes query complexity and selectively applies multi-step reasoning only when needed, avoiding unnecessary latency on simple queries.
3. CoT (Chain-of-Thought) vs Entropy Decoding:
   - CoT forces explicit intermediate reasoning tokens.
   - Entropy decoding measures output token probability distributions; when uncertainty/entropy spikes, it initiates deeper verification branches.
=============================================================================

# Install before running:
# pip install optillm datasets
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import datasets
import pandas as pd
import torch
from datasets import load_dataset
from optillm.autothink import autothink_decode
from optillm.cot_decoding import cot_decode
from optillm.entropy_decoding import entropy_decode
from optillm.thinkdeeper import thinkdeeper_decode
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OPTILLM_DATASET = "codelion/optillmbench"
DATASET_CATEGORY = "gsm8k"
MAX_SAMPLES = 28
OUTPUT_DIR = "results"

DECODE_PARAMS = {"do_sample": True, "temperature": 0.1, "max_new_tokens": 1024}

MATH_PROMPT = (
    "In a dance class of 20 students, 20% enrolled in contemporary dance, "
    "25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. "
    "What percentage of the entire students enrolled in hip-hop dance?"
)

GSM8K_PROMPTS = [
    "There are 4,000 jelly beans in a jar. If three fourths of the jelly beans are red, "
    "and one quarter of the red jelly beans are coconut flavored, how many jelly beans are coconut flavored?",
    'There have been 15 "Where\'s Waldo?" books published. Each book has 30 puzzles to find Waldo. '
    "The average person takes 3 minutes to find Waldo in a puzzle. How long would it take to find every Waldo?",
    "Bart makes a mixtape. The first side has 6 songs. The second side has 4 songs. "
    "Each song is 4 minutes. How long is the total tape?",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def download_model_from_hf(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Download model checkpoints and tokenizer from the HF Hub."""
    with console.status(f"[bold green]Loading {model_name}..."):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    console.print(f"[bold green]✔[/bold green] Loaded [bold]{model_name}[/bold]")
    return model, tokenizer


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# ---------------------------------------------------------------------------
# Decoding comparisons
# ---------------------------------------------------------------------------


def run_decoding_comparison(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
) -> None:
    """Run ThinkDeeper, AutoThink, CoT, Entropy, and greedy decoding on the same prompt."""
    with status_spinner("Running ThinkDeeper decoding..."):
        res_thinkdeeper = thinkdeeper_decode(model, tokenizer, messages, DECODE_PARAMS)
    render_card("ThinkDeeper Decoding", str(res_thinkdeeper), icon="🧠")

    with status_spinner("Running AutoThink decoding..."):
        res_autothink = autothink_decode(model, tokenizer, messages, DECODE_PARAMS)
    render_card("AutoThink Decoding", str(res_autothink), icon="⚡")

    with status_spinner("Running Chain-of-Thought (CoT) decoding..."):
        res_cot, confidence = cot_decode(
            model,
            tokenizer,
            messages,
            aggregate_paths=True,
            temperature=0.1,
            max_new_tokens=1024,
        )
    render_card("CoT Decoding", f"{res_cot}\n\n[text.muted]Confidence Score: {confidence}[/text.muted]", icon="✨")

    with status_spinner("Running Entropy decoding..."):
        res_entropy = entropy_decode(model, tokenizer, messages, temperature=0.1, max_new_tokens=1024)
    render_card("Entropy Decoding", str(res_entropy), icon="🔬")

    # Greedy decoding
    with status_spinner("Running Greedy decoding baseline..."):
        device = get_device()
        model = model.to(device)
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        attention_mask = torch.ones_like(input_ids)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        input_length = input_ids.shape[1]

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            do_sample=False,  # greedy
            num_beams=1,
            pad_token_id=pad_token_id,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
        )
        output_ids = output_ids.cpu()
        response = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
    render_card("Greedy Decoding Baseline", response, icon="📄")


def run_vanilla_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
) -> str:
    """Run vanilla (non-OptiLLM) inference on a single GSM8k prompt."""
    question = (
        "Solve this math problem step by step. After solving, provide the final "
        "numerical answer after '### ' (three hash symbols and a space).\n\n"
        f"Question: {prompt}\n\n"
        "Show your work, then give the final answer after '### '."
    )
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant focused on providing precise answers in the requested format.",
        },
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        do_sample=True,
        temperature=0.1,
        max_new_tokens=1024,
    )
    generated_ids = [out[len(inp) :] for inp, out in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_optillm_bench() -> datasets.Dataset:
    """Load and filter the OptiLLMBench dataset to GSM8k samples only."""
    with console.status(f"[bold green]Loading {OPTILLM_DATASET} ({DATASET_CATEGORY})..."):
        dataset = load_dataset(OPTILLM_DATASET)
        filtered = dataset["test"].filter(lambda ex: ex["category"] == DATASET_CATEGORY)
    console.print(f"[bold green]✔[/bold green] Loaded [bold]{len(filtered)}[/bold] benchmark samples.")
    return filtered


def get_prompt_for_category(question: str, category: str) -> str:
    """Generate the appropriate prompt template for each OptiLLMBench category."""
    if category == "gsm8k":
        return (
            f"Solve this math problem step by step. After solving, provide the final "
            f"numerical answer after '### ' (three hash symbols and a space).\n\n"
            f"Question: {question}\n\n"
            f"Show your work, then give the final answer after '### '."
        )
    elif category == "mmlu_math":
        return f"Solve this math problem. Provide only the answer with no explanation.\n\nQuestion: {question}"
    elif category == "boolq":
        return f"Answer this yes/no question with only 'yes' or 'no'.\n\nQuestion: {question}"
    elif category == "aqua_rat":
        return f"Choose the correct answer. Provide only the letter choice with no explanation.\n\nQuestion: {question}"
    else:
        return f"Question: {question}"


# ---------------------------------------------------------------------------
# Answer processing
# ---------------------------------------------------------------------------


def remove_thinking_blocks(text: str) -> str:
    if not text:
        return text
    if "</think>" in text:
        parts = text.split("</think>")
        return parts[-1].strip()
    return text


def extract_gsm8k_answer(text: str) -> float:
    match = re.search(r"###\s*(-?\d*\.?\d+)", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def extract_choice_index_from_question(question: str, answer: str) -> int:
    answer_clean = answer.strip().lower()
    if "choices:" in question.lower():
        choices_section = question.lower().split("choices:")[1].strip()
        if "\n" not in choices_section:
            all_choices = re.findall(r"(\d+)\s*\.\s*([^0-9.]+?)(?=\s*\d+\s*\.|$)", choices_section)
            for idx, choice_text in all_choices:
                if choice_text.strip().lower() == answer_clean:
                    return int(idx)

        for choice in choices_section.split("\n"):
            choice = choice.strip()
            if not choice:
                continue
            match = re.match(r"\s*(\d+)\s*\.\s*(.*)", choice)
            if match:
                idx = int(match.group(1))
                choice_text = match.group(2).strip()
                if choice_text.lower() == answer_clean:
                    return idx

        pattern = r"(\d+)\s*\.\s*" + re.escape(answer_clean)
        match = re.search(pattern, choices_section)
        if match:
            return int(match.group(1))
    return -1


def is_numeric_only_response(response: str) -> tuple[bool, int]:
    clean_response = re.sub(r"\s", "", response)
    if clean_response.isdigit():
        return True, int(clean_response)
    return False, -1


def evaluate_response(response: str, ground_truth: str, category: str, question: str = None) -> bool:
    if not response or not ground_truth:
        return False

    response = remove_thinking_blocks(response)

    if category == "gsm8k":
        response_num = extract_gsm8k_answer(response)
        ground_truth_num = extract_gsm8k_answer(ground_truth)
        if response_num is None or ground_truth_num is None:
            return False
        return abs(response_num - ground_truth_num) < 1e-6
    elif category == "mmlu_math":
        response_clean = response.strip().lower()
        ground_truth_clean = ground_truth.strip().lower()

        if response_clean == ground_truth_clean:
            return True

        if question:
            correct_index = extract_choice_index_from_question(question, ground_truth)
            if correct_index >= 0:
                is_numeric, value = is_numeric_only_response(response)
                if is_numeric and value == correct_index:
                    return True
                if re.search(rf"{correct_index}\s*\.\s*{re.escape(ground_truth_clean)}", response_clean):
                    return True
                if str(correct_index) in response_clean and ground_truth_clean in response_clean:
                    return True
        return False
    else:
        return response.strip().lower() == ground_truth.strip().lower()


# ---------------------------------------------------------------------------
# Benchmark evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: datasets.Dataset,
    max_samples: int = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate a model on the dataset using AutoThink decoding."""
    metrics = {"total_correct": 0, "total_time": 0.0, "samples": 0}
    category_metrics: dict[str, dict] = {}
    detailed_results: list[dict[str, Any]] = []

    examples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

    for example in tqdm(examples, desc="Evaluating with AutoThink"):
        try:
            prompt = get_prompt_for_category(example["question"], example["category"])
            start_time = time.time()

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant focused on providing precise answers in the requested format.",
                },
                {"role": "user", "content": prompt},
            ]
            response = autothink_decode(model, tokenizer, messages, DECODE_PARAMS)

            time_taken = time.time() - start_time
            raw_response = response
            processed_response = remove_thinking_blocks(raw_response)

            is_correct = evaluate_response(
                processed_response,
                example["answer"],
                example["category"],
                example["question"],
            )

            metrics["total_correct"] += int(is_correct)
            metrics["total_time"] += time_taken
            metrics["samples"] += 1

            cat = example["category"]
            if cat not in category_metrics:
                category_metrics[cat] = {"correct": 0, "total": 0, "time": 0.0}
            category_metrics[cat]["correct"] += int(is_correct)
            category_metrics[cat]["total"] += 1
            category_metrics[cat]["time"] += time_taken

            has_thinking = "</think>" in raw_response
            detailed_results.append(
                {
                    "id": example["id"],
                    "category": cat,
                    "correct": is_correct,
                    "time_taken": time_taken,
                    "raw_response": raw_response,
                    "processed_response": processed_response if has_thinking else None,
                    "has_thinking": has_thinking,
                    "ground_truth": example["answer"],
                }
            )

        except Exception as e:
            logger.error(f"Error processing example {example['id']}: {e}")
            continue

    n = metrics["samples"]
    final_metrics: dict[str, float] = {
        "accuracy": metrics["total_correct"] / n if n > 0 else 0.0,
        "average_time": metrics["total_time"] / n if n > 0 else 0.0,
        "total_time": metrics["total_time"],
        "total_samples": n,
    }
    for cat, cm in category_metrics.items():
        final_metrics[f"{cat}_accuracy"] = cm["correct"] / cm["total"]
        final_metrics[f"{cat}_average_time"] = cm["time"] / cm["total"]

    return final_metrics, detailed_results


def save_results(
    metrics: dict[str, float],
    detailed_results: list[dict[str, Any]],
    model_name: str,
    output_dir: str,
) -> None:
    """Save evaluation metrics and detailed results to JSON/CSV files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_dir, model_name.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)
    base_filename = os.path.join(model_dir, f"_{timestamp}")

    with open(f"{base_filename}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(f"{base_filename}_detailed.json", "w") as f:
        json.dump(detailed_results, f, indent=2)

    df = pd.DataFrame(
        [
            {k: v for k, v in result.items() if k not in ("raw_response", "processed_response")}
            for result in detailed_results
        ]
    )
    df.to_csv(f"{base_filename}_summary.csv", index=False)
    console.print(f"[bold green]✔[/bold green] Results saved to [yellow]{base_filename}_*[/yellow]")


def generate_report(all_metrics: dict[str, float], output_dir: str) -> None:
    """Generate a markdown report from evaluation metrics."""
    report = [
        "# OptiLLM Bench Evaluation Report",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Overall Results",
    ]

    rows = [
        [
            f"{all_metrics['accuracy'] * 100:.2f}%",
            f"{all_metrics['average_time']:.2f}",
            f"{all_metrics['total_time']:.2f}",
        ]
    ]
    df = pd.DataFrame(rows, columns=["Accuracy", "Avg Time (s)", "Total Time (s)"])
    report.append(df.to_markdown())

    report_path = f"{output_dir}/evaluation_report.md"
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n\n".join(report))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Download model, run decoding comparison, then benchmark on OptiLLMBench/GSM8k."""
    render_banner(
        title="AutoThink & Test-Time Reasoning with OptiLLM & Qwen 2.5",
        subtitle="Chapter 15: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_NAME,
            "Benchmark": OPTILLM_DATASET,
            "Subset": DATASET_CATEGORY,
        },
        icon="🚀",
    )

    # Step 1: Decoding Strategies Comparison (Math Prompt)
    render_step(1, "Comparing Test-Time Search & Decoding Strategies", icon="📋")
    model, tokenizer = download_model_from_hf(MODEL_NAME)
    render_card("Math Evaluation Prompt", MATH_PROMPT, icon="❓")
    messages = [{"role": "user", "content": MATH_PROMPT}]
    run_decoding_comparison(model, tokenizer, messages)

    # Step 2: Vanilla GSM8k Single-Sample Inference
    render_step(2, "Evaluating Greedy Baseline on GSM8k", icon="🧠")
    with status_spinner("Running vanilla GSM8k baseline inference..."):
        response = run_vanilla_inference(model, tokenizer, GSM8K_PROMPTS[2])
    render_card("Vanilla GSM8k Output", response, icon="📄")

    # Step 3: Benchmarking AutoThink on OptiLLMBench
    render_step(3, "Evaluating Adaptive AutoThink on OptiLLMBench", icon="📊")
    dataset = load_optillm_bench()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        metrics, detailed_results = evaluate_model(model, tokenizer, dataset, MAX_SAMPLES)
        save_results(metrics, detailed_results, MODEL_NAME, OUTPUT_DIR)

        columns = [("Evaluation Metric", STYLE_PRIMARY, "left"), ("Measurement", STYLE_SUCCESS, "right")]
        rows = [
            ("Total Samples Evaluated", str(metrics["total_samples"])),
            ("Accuracy", f"{metrics['accuracy'] * 100:.2f}%"),
            ("Average Time per Sample", f"{metrics['average_time']:.2f} s"),
            ("Total Evaluation Time", f"{metrics['total_time']:.2f} s"),
        ]
        console.print(create_table("OptiLLM AutoThink GSM8k Benchmark Summary", columns, rows))
        pause()

        generate_report(metrics, os.path.join(OUTPUT_DIR, MODEL_NAME.replace("/", "_")))
    except Exception as e:
        render_card("Benchmark Notice", f"Benchmark requires full dependencies: {e}", icon="ℹ️")

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Test-Time Search vs Model Size",
                "Small language models (0.5B-3B) with smart inference-time decoding (AutoThink, Entropy decoding) can rival models 5-10x their size on complex multi-step math and reasoning tasks.",
            ),
            (
                "Adaptive Compute Allocation",
                "Rather than paying a fixed high computational cost on every prompt, AutoThink only triggers deep chain-of-thought exploration when token probability entropy indicates ambiguity or complexity.",
            ),
            (
                "Removing Thinking Blocks",
                "Structured reasoning models produce <think>...</think> tags during internal exploration. Parsing and stripping thinking blocks before final answer evaluation ensures clean automated validation.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
