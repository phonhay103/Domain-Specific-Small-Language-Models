# ==========================================
# Extracted from CH15_NB01_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # AutoThink Example with OptiLLM and Qwen 2.5 0.5B Instruct
# This notebook is a companion of chapter 15 of the "Domain-Specific Small Language Models" [book](https://www.manning.com/books/domain-specific-small-language-models), author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2025.  
# The code in this notebook is an example of usage of the [AutoThink](https://dx.doi.org/10.2139/ssrn.5253327) technique in [OptiLLM](https://github.com/codelion/optillm/) with the [Qwen 2.5 0.5B instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) model. Hardware acceleration (GPU) is recommended.   
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install OptiLLM. A session restart is needed at the end of the installation process.
# ------------------------------------------------------------

# !pip install optillm

# ------------------------------------------------------------
# Define a custom function to download the model checkpoints and associated tokenizer from the HF's Hub.
# ------------------------------------------------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def download_model_from_hf(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

# ------------------------------------------------------------
# Download the Qwen 2.5 0.5B Instuct model and companion tokenizer from the HF's Hub.
# ------------------------------------------------------------

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model, tokenizer = download_model_from_hf(model_name)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ------------------------------------------------------------
# Provide a prompt (a mathematical task).
# ------------------------------------------------------------

messages = [
    {"role": "user", "content": "In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?"}
]

# ------------------------------------------------------------
# Test the model response using different OptiLLM built-in decoding techniques (Thinkdeeper, AutoThink, CoT Decoding and Entropy Decoding).
# ------------------------------------------------------------

from optillm.thinkdeeper import thinkdeeper_decode

result = thinkdeeper_decode(model, tokenizer, messages, {"do_sample": True, "temperature": 0.1, "max_new_tokens": 1024})
print(f"ThinkDeeper Decoding:\n {result}")

from optillm.autothink import autothink_decode

result = autothink_decode(model, tokenizer, messages, {"do_sample": True, "temperature": 0.1, "max_new_tokens": 1024})
print(f"AutoThink Decoding:\n {result}")

from optillm.cot_decoding import cot_decode

# Generate the response using CoT decoding
result, confidence = cot_decode(model, tokenizer, messages, aggregate_paths=True, temperature=0.1, max_new_tokens=1024)
print(f"CoT Decoding:\n {result}")
# print(f"Confidence: {confidence}")

from optillm.entropy_decoding import entropy_decode

# Generate the response using Entropy decoding
result = entropy_decode(model, tokenizer, messages, temperature=0.1, max_new_tokens=1024)
print(f"\nEntropy Decoding:\n {result}")

# ------------------------------------------------------------
# Do greedy decoding with the same model on the same prompt to compare results.
# ------------------------------------------------------------

def get_device():
  if torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")

device = get_device()
model = model.to(device)

# Prepare input with proper attention mask
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
attention_mask = torch.ones_like(input_ids)  # Create attention mask
input_ids = input_ids.to(device)
input_length = input_ids.shape[1]
attention_mask = attention_mask.to(device)

# Get pad and eos token ids
pad_token_id = tokenizer.pad_token_id
if pad_token_id is None:
    pad_token_id = tokenizer.eos_token_id

# Configure generation parameters properly for greedy decoding
output_ids = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=1024,
    do_sample=False,     # Greedy decoding
    num_beams=1,        # Single beam for greedy
    pad_token_id=pad_token_id,
    temperature=1.0,    # Remove or set to 1.0 for greedy
    top_p=1.0,         # Remove or set to 1.0 for greedy
    use_cache=True,    # Enable KV caching for faster generation
)

output_ids = output_ids.cpu()
# Decode only the newly generated tokens
response = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
print(f"Greedy Decoding:\n {response}")

# ------------------------------------------------------------
# ### Vanilla Model Inference on GSM8k Samples.
# The next three code cells are just to show how to do inference on some GSM8k dataset samples using the vanilla Qwen 2.5 0.5 Instruct model (without the OptiLLM proxy). You can skip this section if you are interested only in evaluating OptiLLM.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Repeating below the code to download the model and companion tokenizer from the HF's Hub, just in case you would start executing this notebook from here.
# ------------------------------------------------------------

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model, tokenizer = download_model_from_hf(model_name)

# ------------------------------------------------------------
# Provide some sample from the GSM8k dataset and tokenize them.
# ------------------------------------------------------------

prompts = [
    'There are 4,000 jelly beans in a jar. If three fourths of the jelly beans are red, and one quarter of the red jelly beans are coconut flavored, how many jelly beans are coconut flavored?',
    'There have been 15 "Where\'s Waldo?" books published. Each book has 30 puzzles to find Waldo. The average person takes 3 minutes to find Waldo in a puzzle. How long would it take to find every Waldo?',
    'Bart makes a mixtape. The first side has 6 songs. The second side has 4 songs. Each song is 4 minutes. How long is the total tape?'
]
prompt = prompts[2]
question = f"""Solve this math problem step by step. After solving, provide the final numerical answer after '### ' (three hash symbols and a space).\n\n
            Question: {prompt}\n\n
            Show your work, then give the final answer after '### '."""

messages = [
    {"role": "system", "content": "You are a helpful AI assistant focused on providing precise answers in the requested format."},
    {"role": "user", "content": question}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# ------------------------------------------------------------
# Evaluate the model on the provided list of GSM8k prompts.
# ------------------------------------------------------------

generated_ids = model.generate(
    **model_inputs,
    do_sample=True,
    temperature=0.1,
    max_new_tokens=1024
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

response

# ------------------------------------------------------------
# ### Benchmark (AutoThink in OptiLLM on GSM8k)
# ------------------------------------------------------------

# !pip install -U datasets

# ------------------------------------------------------------
# Setup the logging level to INFO, to minimize the number of output messages during the benchmark.
# ------------------------------------------------------------

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Define a custom function to load the benchmarking dataset, [OptiLLMBench](https://huggingface.co/datasets/codelion/optillmbench). It contains 500 selected challenging problems across multiple datasets (competition_math, HumanEval, GSM8K, MMLU, BBH).
# ------------------------------------------------------------

import datasets
from datasets import load_dataset

def load_optillm_bench() -> datasets.Dataset:
    """Load the OptiLLM Bench dataset."""
    try:
        dataset = load_dataset("codelion/optillmbench")
        gsm8k_dataset = dataset["test"].filter(lambda example: example["category"] == "gsm8k")
        return gsm8k_dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

# ------------------------------------------------------------
# Load the dataset.
# ------------------------------------------------------------

dataset = load_optillm_bench()

# ------------------------------------------------------------
# Define a custom function to set the approriate prompt for each of the categories included in the OptiLLMBench dataset.
# ------------------------------------------------------------

def get_prompt_for_category(question: str, category: str) -> str:
    """
    Generate appropriate prompt based on category.
    """
    if category == "gsm8k":
        return (
            f"Solve this math problem step by step. After solving, provide the final "
            f"numerical answer after '### ' (three hash symbols and a space).\n\n"
            f"Question: {question}\n\n"
            f"Show your work, then give the final answer after '### '."
        )
    elif category == "mmlu_math":
        return (
            f"Solve this math problem. Provide only the answer with no explanation.\n\n"
            f"Question: {question}"
        )
    elif category == "boolq":
        return (
            f"Answer this yes/no question with only 'yes' or 'no'.\n\n"
            f"Question: {question}"
        )
    elif category == "aqua_rat":
        return (
            f"Choose the correct answer. Provide only the letter choice with no explanation.\n\n"
            f"Question: {question}"
        )
    else:
        return f"Question: {question}"

# ------------------------------------------------------------
# Define a custom function to remove the thinking blocks from the model responses.
# ------------------------------------------------------------

def remove_thinking_blocks(text: str) -> str:
    """
    Remove <think>...</think> blocks from the response.
    If there's a </think> tag, only keep the content after it.
    """
    if not text:
        return text

    # Check if there's a thinking block
    if '</think>' in text:
        # Get everything after the last </think> tag
        parts = text.split('</think>')
        return parts[-1].strip()

    return text

# ------------------------------------------------------------
# Define a custom function to extract numerical answer from responses to GSM8K questions.
# ------------------------------------------------------------

def extract_gsm8k_answer(text: str) -> float:
    """Extract numerical answer after ### from GSM8K responses."""
    match = re.search(r'###\s*(-?\d*\.?\d+)', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

# ------------------------------------------------------------
# Define a function to extract the correct answer from a multiple-choice question.
# ------------------------------------------------------------

import re

def extract_choice_index_from_question(question: str, answer: str) -> int:
    """
    Extract the index of the correct answer from a multiple-choice question.

    Args:
        question: The question text containing choices
        answer: The correct answer (just the text, no index)

    Returns:
        int: The index of the correct answer, or -1 if not found
    """
    # Look for a pattern like "N. answer" in the question
    answer_clean = answer.strip().lower()

    # Debug logging for critical examples
    logger.debug(f"Looking for answer: '{answer_clean}' in question")

    # Check for "Choices:" marker in the question
    if "choices:" in question.lower():
        # Split the question by lines after "Choices:"
        choices_section = question.lower().split("choices:")[1].strip()

        # Log the choices section
        logger.debug(f"Choices section: '{choices_section}'")

        # If it's all on one line, use a more comprehensive regex
        if '\n' not in choices_section:
            # This pattern matches "N. text" where N is a digit and text is any text up to the next number or end
            all_choices = re.findall(r'(\d+)\s*\.\s*([^0-9.]+?)(?=\s*\d+\s*\.|$)', choices_section)

            logger.debug(f"Single line choices found: {all_choices}")

            for idx, choice_text in all_choices:
                choice_text_clean = choice_text.strip()
                if choice_text_clean.lower() == answer_clean:
                    logger.debug(f"Found match at index {idx}: '{choice_text_clean}'")
                    return int(idx)

        # Try splitting by newlines
        choices = choices_section.split("\n")

        for i, choice in enumerate(choices):
            choice = choice.strip()
            if not choice:
                continue

            logger.debug(f"Checking choice {i}: '{choice}'")

            # Try to extract the index and choice text
            match = re.match(r'\s*(\d+)\s*\.\s*(.*)', choice)
            if match:
                idx = int(match.group(1))
                choice_text = match.group(2).strip()

                logger.debug(f"Parsed choice: index={idx}, text='{choice_text}'")

                if choice_text.lower() == answer_clean:
                    logger.debug(f"Found exact match at index {idx}")
                    return idx

        # Fallback: just look for any occurrence of the number followed by the answer
        pattern = r'(\d+)\s*\.\s*' + re.escape(answer_clean)
        match = re.search(pattern, choices_section)
        if match:
            logger.debug(f"Fallback match found at index {match.group(1)}")
            return int(match.group(1))

    logger.debug("No match found for answer in choices")
    return -1

# ------------------------------------------------------------
# Define a function to check if a response from the model is purely numerical.
# ------------------------------------------------------------

from typing import Tuple

def is_numeric_only_response(response: str) -> Tuple[bool, int]:
    """
    Check if the response is just a numeric value, possibly with whitespace and newlines.

    Args:
        response: The response text to check

    Returns:
        Tuple of (is_numeric, value)
    """
    # Strip all whitespace, including newlines
    clean_response = re.sub(r'\s', '', response)

    # Check if it's just a number
    if clean_response.isdigit():
        return True, int(clean_response)

    return False, -1

# ------------------------------------------------------------
# Define a function that, using the other custom functions implemented above, to evaluate the responsed from the model.
# ------------------------------------------------------------

def evaluate_response(response: str, ground_truth: str, category: str, question: str = None) -> bool:
    """
    Evaluate if the response matches the ground truth based on category.

    Args:
        response: Model's response
        ground_truth: Correct answer
        category: Problem category (gsm8k, mmlu_math, boolq, aqua_rat)
        question: Original question text, needed for MMLU evaluation

    Returns:
        bool: Whether the response is correct
    """
    if not response or not ground_truth:
        return False

    # First, remove any thinking blocks
    response = remove_thinking_blocks(response)

    if category == "gsm8k":
        # Extract numerical answers after ### and compare
        response_num = extract_gsm8k_answer(response)
        ground_truth_num = extract_gsm8k_answer(ground_truth)

        if response_num is None or ground_truth_num is None:
            return False

        # Compare with small tolerance for floating point
        return abs(response_num - ground_truth_num) < 1e-6
    elif category == "mmlu_math":
        # Special handling for MMLU-math multiple choice questions
        response_clean = response.strip().lower()
        ground_truth_clean = ground_truth.strip().lower()

        # Case 1: Exact match of answer text
        if response_clean == ground_truth_clean:
            logger.debug("Exact text match")
            return True

        # For other cases, we need to find what index corresponds to the ground truth
        if question:
            correct_index = extract_choice_index_from_question(question, ground_truth)

            if correct_index >= 0:
                # Case 2: Check if response is just the digit (most common LLM response for indices)
                is_numeric, value = is_numeric_only_response(response)
                if is_numeric and value == correct_index:
                    logger.debug(f"Numeric match: response '{response}' -> {value} matches index {correct_index}")
                    return True

                # Case 3: Check if response is "index. answer"
                if re.search(fr"{correct_index}\s*\.\s*{re.escape(ground_truth_clean)}", response_clean):
                    logger.debug("Pattern match for 'index. answer'")
                    return True

                # Case 4: Check if response contains both the index and the answer text
                if str(correct_index) in response_clean and ground_truth_clean in response_clean:
                    logger.debug("Contains both index and answer")
                    return True

        return False
    else:
        # Clean up both strings for comparison
        response_clean = response.strip().lower()
        ground_truth_clean = ground_truth.strip().lower()
        return response_clean == ground_truth_clean

# ------------------------------------------------------------
# Define a custom function that iterates through the samples in the datasets, run the models using the AutoThink technique on them, evaluate the responses and then returns the results and calculate evaluation metrics.
# ------------------------------------------------------------

import time
from typing import Dict, List, Any
from optillm.autothink import autothink_decode
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: datasets.Dataset,
    max_samples: int = None
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Evaluate a model on the dataset using a specific approach.
    Returns metrics and detailed results.
    """
    metrics = {
        "total_correct": 0,
        "total_time": 0,
        "samples": 0,
    }

    # Initialize category-specific metrics
    category_metrics = {}

    # Detailed results for each example
    detailed_results = []

    # Prepare the dataset
    examples = dataset if max_samples is None else dataset.select(range(max_samples))

    for example in tqdm(examples, desc=f"Evaluating"):
        try:
            # Get appropriate prompt for the category
            prompt = get_prompt_for_category(example['question'], example['category'])

            # Record start time
            start_time = time.time()

            # Do inference
            messages=[
                    {"role": "system", "content": "You are a helpful AI assistant focused on providing precise answers in the requested format."},
                    {"role": "user", "content": prompt}
                ]
            response = autothink_decode(model, tokenizer, messages, {"do_sample": True, "temperature": 0.1, "max_new_tokens": 1024})
            #print(response)

            # Calculate time taken
            time_taken = time.time() - start_time

            # Get the response text
            response_text = response

            # Also store the raw response for reference
            raw_response = response_text

            # Process the response to remove thinking blocks
            processed_response = remove_thinking_blocks(response_text)

            # Evaluate the processed response
            is_correct = evaluate_response(
                processed_response,
                example['answer'],
                example['category'],
                example['question']  # Pass the question for MMLU evaluation
            )

            # Update metrics
            metrics["total_correct"] += int(is_correct)
            metrics["total_time"] += time_taken
            metrics["samples"] += 1

            # Update category metrics
            if example['category'] not in category_metrics:
                category_metrics[example['category']] = {
                    "correct": 0,
                    "total": 0,
                    "time": 0
                }
            category_metrics[example['category']]["correct"] += int(is_correct)
            category_metrics[example['category']]["total"] += 1
            category_metrics[example['category']]["time"] += time_taken

            # Check if thinking blocks were removed
            has_thinking = '</think>' in raw_response

            # Record detailed result
            detailed_results.append({
                "id": example['id'],
                "category": example['category'],
                "correct": is_correct,
                "time_taken": time_taken,
                "raw_response": raw_response,
                "processed_response": processed_response if has_thinking else None,
                "has_thinking": has_thinking,
                "ground_truth": example['answer']
            })

        except Exception as e:
            logger.error(f"Error processing example {example['id']}: {e}")
            continue

    # Calculate final metrics
    final_metrics = {
        "accuracy": metrics["total_correct"] / metrics["samples"] if metrics["samples"] > 0 else 0,
        "average_time": metrics["total_time"] / metrics["samples"] if metrics["samples"] > 0 else 0,
        "total_time": metrics["total_time"],
        "total_samples": metrics["samples"],
    }

    # Add category-specific metrics
    for category, cat_metrics in category_metrics.items():
        final_metrics[f"{category}_accuracy"] = cat_metrics["correct"] / cat_metrics["total"]
        final_metrics[f"{category}_average_time"] = cat_metrics["time"] / cat_metrics["total"]

    return final_metrics, detailed_results

# ------------------------------------------------------------
# Define a custom function to save the evaluation results to file.
# ------------------------------------------------------------

import json
from datetime import datetime

def save_results(metrics: Dict[str, float], detailed_results: List[Dict[str, Any]],
                model: str, output_dir: str):
    """Save evaluation results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create model-specific directory
    model_dir = os.path.join(output_dir, model.replace('/', '_'))
    os.makedirs(model_dir, exist_ok=True)

    base_filename = os.path.join(model_dir, f"_{timestamp}")

    # Save metrics
    with open(f"{base_filename}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save detailed results
    with open(f"{base_filename}_detailed.json", "w") as f:
        json.dump(detailed_results, f, indent=2)

    # Create a summary DataFrame for easier analysis
    df = pd.DataFrame([
        {k: v for k, v in result.items() if k != 'raw_response' and k != 'processed_response'}
        for result in detailed_results
    ])
    df.to_csv(f"{base_filename}_summary.csv", index=False)

    logger.info(f"Results saved to {base_filename}_*")

# ------------------------------------------------------------
# Define a custom function to generate a report starting from the evaluation metrics.
# ------------------------------------------------------------

import pandas as pd
from datetime import datetime

def generate_report(all_metrics: Dict[str, Dict[str, float]], output_dir: str):
    """Generate a comprehensive report comparing all approaches."""
    report = []

    # Header
    report.append("# OptiLLM Bench Evaluation Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Overall Results Table
    report.append("## Overall Results")
    headers = ["Accuracy", "Avg Time (s)", "Total Time (s)"]
    rows = []

    '''for metrics in all_metrics.items():
        rows.append([
            f"{metrics['accuracy']*100:.2f}%",
            f"{metrics['average_time']:.2f}",
            f"{metrics['total_time']:.2f}"
        ])'''
    rows.append([
        f"{all_metrics['accuracy']*100:.2f}%",
        f"{all_metrics['average_time']:.2f}",
        f"{all_metrics['total_time']:.2f}"
    ])

    # Convert to DataFrame for nice formatting
    df = pd.DataFrame(rows, columns=headers)
    report.append(df.to_markdown())

    # Category-wise Results
    report.append("\n## Results by Category")
    categories = ["gsm8k", "mmlu_math", "boolq", "aqua_rat"]

    for category in categories:
        report.append(f"\n### {category.upper()}")
        headers = ["Accuracy", "Avg Time (s)"]
        rows = []
        if f"{category}_accuracy" in all_metrics:
            rows.append([
                f"{all_metrics[f'{category}_accuracy']*100:.2f}%",
                f"{all_metrics[f'{category}_average_time']:.2f}"
            ])

        df = pd.DataFrame(rows, columns=headers)
        report.append(df.to_markdown())

    # Save report
    report_path = f"{output_dir}/evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("\n\n".join(report))

    logger.info(f"Report saved to {report_path}")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model, tokenizer = download_model_from_hf(model_name)

# ------------------------------------------------------------
# Do model evaluation on the downloaded dataset.
# ------------------------------------------------------------

import os

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
try:
    metrics, detailed_results = evaluate_model(
        model,
        tokenizer,
        dataset,
        28
    )

    save_results(metrics, detailed_results, model_id,
                output_dir)

    logger.info(f"Completed evaluation.")
    logger.info(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"Average time per sample: {metrics['average_time']:.2f}s")

except Exception as e:
    logger.error(f"Error evaluating: {e}")

# ------------------------------------------------------------
# Display the evaluation metrics.
# ------------------------------------------------------------

metrics

# ------------------------------------------------------------
# Generate the final report.
# ------------------------------------------------------------

generate_report(metrics, os.path.join(output_dir, model_id.replace('/', '_')))

