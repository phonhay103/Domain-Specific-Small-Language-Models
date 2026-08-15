"""
AutoThink Example with OptiLLM and Qwen 2.5 0.5B Instruct.

Companion script for chapter 15 of "Domain-Specific Small Language Models"
by Guglielmo Iozzia, Manning Publications, 2025.

Demonstrates the AutoThink technique in OptiLLM with the Qwen 2.5 0.5B
Instruct model, and benchmarks it on the OptiLLMBench / GSM8k dataset.
GPU is recommended.

# Install before running:
# pip install optillm datasets
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

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
    "There have been 15 \"Where's Waldo?\" books published. Each book has 30 puzzles to find Waldo. "
    "The average person takes 3 minutes to find Waldo in a puzzle. How long would it take to find every Waldo?",
    "Bart makes a mixtape. The first side has 6 songs. The second side has 4 songs. "
    "Each song is 4 minutes. How long is the total tape?",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def download_model_from_hf(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Download model checkpoints and tokenizer from the HF Hub."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    messages: List[Dict[str, str]],
) -> None:
    """Run ThinkDeeper, AutoThink, CoT, Entropy, and greedy decoding on the same prompt."""
    result = thinkdeeper_decode(model, tokenizer, messages, DECODE_PARAMS)
    print(f"ThinkDeeper Decoding:\n {result}")

    result = autothink_decode(model, tokenizer, messages, DECODE_PARAMS)
    print(f"AutoThink Decoding:\n {result}")

    # CoT decoding also returns a confidence score
    result, confidence = cot_decode(
        model, tokenizer, messages,
        aggregate_paths=True, temperature=0.1, max_new_tokens=1024,
    )
    print(f"CoT Decoding:\n {result}")

    result = entropy_decode(model, tokenizer, messages, temperature=0.1, max_new_tokens=1024)
    print(f"\nEntropy Decoding:\n {result}")

    # Greedy decoding — for comparison with OptiLLM techniques
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
        do_sample=False,   # greedy
        num_beams=1,
        pad_token_id=pad_token_id,
        temperature=1.0,   # required to be 1.0 for greedy
        top_p=1.0,
        use_cache=True,
    )
    output_ids = output_ids.cpu()
    response = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
    print(f"Greedy Decoding:\n {response}")


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
        {"role": "system", "content": "You are a helpful AI assistant focused on providing precise answers in the requested format."},
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
    generated_ids = [
        out[len(inp):]
        for inp, out in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_optillm_bench() -> datasets.Dataset:
    """Load and filter the OptiLLMBench dataset to GSM8k samples only."""
    try:
        dataset = load_dataset(OPTILLM_DATASET)
        return dataset["test"].filter(lambda ex: ex["category"] == DATASET_CATEGORY)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


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
    """Remove <think>...</think> blocks from a response.

    If a </think> tag is present, only the content after it is retained.
    """
    if not text:
        return text
    if '</think>' in text:
        parts = text.split('</think>')
        return parts[-1].strip()
    return text


def extract_gsm8k_answer(text: str) -> float:
    """Extract the numerical answer after '### ' in GSM8k responses."""
    match = re.search(r'###\s*(-?\d*\.?\d+)', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def extract_choice_index_from_question(question: str, answer: str) -> int:
    """Extract the integer index of the correct answer in a multiple-choice question.

    Args:
        question: Question text that includes the choices.
        answer: The correct answer text (no index prefix).

    Returns:
        The integer index of the correct answer, or -1 if not found.
    """
    answer_clean = answer.strip().lower()
    logger.debug(f"Looking for answer: '{answer_clean}' in question")

    if "choices:" in question.lower():
        choices_section = question.lower().split("choices:")[1].strip()
        logger.debug(f"Choices section: '{choices_section}'")

        if '\n' not in choices_section:
            all_choices = re.findall(
                r'(\d+)\s*\.\s*([^0-9.]+?)(?=\s*\d+\s*\.|$)', choices_section
            )
            logger.debug(f"Single line choices found: {all_choices}")
            for idx, choice_text in all_choices:
                if choice_text.strip().lower() == answer_clean:
                    logger.debug(f"Found match at index {idx}: '{choice_text.strip()}'")
                    return int(idx)

        for choice in choices_section.split("\n"):
            choice = choice.strip()
            if not choice:
                continue
            logger.debug(f"Checking choice: '{choice}'")
            match = re.match(r'\s*(\d+)\s*\.\s*(.*)', choice)
            if match:
                idx = int(match.group(1))
                choice_text = match.group(2).strip()
                logger.debug(f"Parsed choice: index={idx}, text='{choice_text}'")
                if choice_text.lower() == answer_clean:
                    logger.debug(f"Found exact match at index {idx}")
                    return idx

        # Fallback: look for any occurrence of the number followed by the answer
        pattern = r'(\d+)\s*\.\s*' + re.escape(answer_clean)
        match = re.search(pattern, choices_section)
        if match:
            logger.debug(f"Fallback match found at index {match.group(1)}")
            return int(match.group(1))

    logger.debug("No match found for answer in choices")
    return -1


def is_numeric_only_response(response: str) -> Tuple[bool, int]:
    """Check whether a response is purely numeric.

    Args:
        response: The response text.

    Returns:
        (is_numeric, value) tuple.
    """
    clean_response = re.sub(r'\s', '', response)
    if clean_response.isdigit():
        return True, int(clean_response)
    return False, -1


def evaluate_response(
    response: str, ground_truth: str, category: str, question: str = None
) -> bool:
    """Evaluate whether a model response matches the ground truth.

    Args:
        response: Model's response.
        ground_truth: Correct answer.
        category: Problem category (gsm8k, mmlu_math, boolq, aqua_rat).
        question: Original question text — required for MMLU evaluation.

    Returns:
        True if the response is correct.
    """
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
            logger.debug("Exact text match")
            return True

        if question:
            correct_index = extract_choice_index_from_question(question, ground_truth)
            if correct_index >= 0:
                is_numeric, value = is_numeric_only_response(response)
                if is_numeric and value == correct_index:
                    logger.debug(f"Numeric match: {value} == {correct_index}")
                    return True
                if re.search(fr"{correct_index}\s*\.\s*{re.escape(ground_truth_clean)}", response_clean):
                    logger.debug("Pattern match for 'index. answer'")
                    return True
                if str(correct_index) in response_clean and ground_truth_clean in response_clean:
                    logger.debug("Contains both index and answer")
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
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Evaluate a model on the dataset using AutoThink decoding.

    Returns:
        (final_metrics, detailed_results)
    """
    metrics = {"total_correct": 0, "total_time": 0.0, "samples": 0}
    category_metrics: Dict[str, Dict] = {}
    detailed_results: List[Dict[str, Any]] = []

    examples = dataset if max_samples is None else dataset.select(range(max_samples))

    for example in tqdm(examples, desc="Evaluating"):
        try:
            prompt = get_prompt_for_category(example['question'], example['category'])
            start_time = time.time()

            messages = [
                {"role": "system", "content": "You are a helpful AI assistant focused on providing precise answers in the requested format."},
                {"role": "user", "content": prompt},
            ]
            response = autothink_decode(model, tokenizer, messages, DECODE_PARAMS)

            time_taken = time.time() - start_time
            raw_response = response
            processed_response = remove_thinking_blocks(raw_response)

            is_correct = evaluate_response(
                processed_response,
                example['answer'],
                example['category'],
                example['question'],
            )

            metrics["total_correct"] += int(is_correct)
            metrics["total_time"] += time_taken
            metrics["samples"] += 1

            cat = example['category']
            if cat not in category_metrics:
                category_metrics[cat] = {"correct": 0, "total": 0, "time": 0.0}
            category_metrics[cat]["correct"] += int(is_correct)
            category_metrics[cat]["total"] += 1
            category_metrics[cat]["time"] += time_taken

            has_thinking = '</think>' in raw_response
            detailed_results.append({
                "id": example['id'],
                "category": cat,
                "correct": is_correct,
                "time_taken": time_taken,
                "raw_response": raw_response,
                "processed_response": processed_response if has_thinking else None,
                "has_thinking": has_thinking,
                "ground_truth": example['answer'],
            })

        except Exception as e:
            logger.error(f"Error processing example {example['id']}: {e}")
            continue

    n = metrics["samples"]
    final_metrics: Dict[str, float] = {
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
    metrics: Dict[str, float],
    detailed_results: List[Dict[str, Any]],
    model_name: str,
    output_dir: str,
) -> None:
    """Save evaluation metrics and detailed results to JSON/CSV files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_dir, model_name.replace('/', '_'))
    os.makedirs(model_dir, exist_ok=True)
    base_filename = os.path.join(model_dir, f"_{timestamp}")

    with open(f"{base_filename}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(f"{base_filename}_detailed.json", "w") as f:
        json.dump(detailed_results, f, indent=2)

    df = pd.DataFrame([
        {k: v for k, v in result.items()
         if k not in ('raw_response', 'processed_response')}
        for result in detailed_results
    ])
    df.to_csv(f"{base_filename}_summary.csv", index=False)

    logger.info(f"Results saved to {base_filename}_*")


def generate_report(all_metrics: Dict[str, float], output_dir: str) -> None:
    """Generate a markdown report from evaluation metrics."""
    report = [
        "# OptiLLM Bench Evaluation Report",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Overall Results",
    ]

    rows = [[
        f"{all_metrics['accuracy'] * 100:.2f}%",
        f"{all_metrics['average_time']:.2f}",
        f"{all_metrics['total_time']:.2f}",
    ]]
    df = pd.DataFrame(rows, columns=["Accuracy", "Avg Time (s)", "Total Time (s)"])
    report.append(df.to_markdown())

    report.append("\n## Results by Category")
    for category in ["gsm8k", "mmlu_math", "boolq", "aqua_rat"]:
        report.append(f"\n### {category.upper()}")
        rows = []
        if f"{category}_accuracy" in all_metrics:
            rows.append([
                f"{all_metrics[f'{category}_accuracy'] * 100:.2f}%",
                f"{all_metrics[f'{category}_average_time']:.2f}",
            ])
        df = pd.DataFrame(rows, columns=["Accuracy", "Avg Time (s)"])
        report.append(df.to_markdown())

    report_path = f"{output_dir}/evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("\n\n".join(report))

    logger.info(f"Report saved to {report_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Download model, run decoding comparison, then benchmark on OptiLLMBench/GSM8k."""
    # --- Decoding comparison on a single math prompt ---
    model, tokenizer = download_model_from_hf(MODEL_NAME)
    messages = [{"role": "user", "content": MATH_PROMPT}]
    run_decoding_comparison(model, tokenizer, messages)

    # --- Vanilla inference on GSM8k sample (for comparison) ---
    response = run_vanilla_inference(model, tokenizer, GSM8K_PROMPTS[2])
    print(response)

    # --- Benchmark on OptiLLMBench/GSM8k ---
    dataset = load_optillm_bench()

    # Re-download model to ensure clean state before benchmarking
    model, tokenizer = download_model_from_hf(MODEL_NAME)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        metrics, detailed_results = evaluate_model(model, tokenizer, dataset, MAX_SAMPLES)
        save_results(metrics, detailed_results, MODEL_NAME, OUTPUT_DIR)

        logger.info("Completed evaluation.")
        logger.info(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
        logger.info(f"Average time per sample: {metrics['average_time']:.2f}s")
    except Exception as e:
        logger.error(f"Error evaluating: {e}")

    print(metrics)
    generate_report(metrics, os.path.join(OUTPUT_DIR, MODEL_NAME.replace('/', '_')))


if __name__ == "__main__":
    main()
