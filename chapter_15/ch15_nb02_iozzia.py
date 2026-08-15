"""
Adapt an SLM to reason over a custom domain and dataset using GRPO.

Companion script for chapter 15 of "Domain-Specific Small Language Models"
by Guglielmo Iozzia, Manning Publications, 2026.

Fine-tunes Qwen 2.5 3B Instruct to specialise on a QA task about additive
manufacturing (3D printing) through GRPO + QLoRA via Unsloth and TRL.
GPU is required.

# Install the missing dependencies before running:
# pip install --force-reinstall datasets
# pip uninstall -y huggingface-hub transformers tokenizers
# pip install huggingface-hub==0.34.0 transformers==4.56.2 tokenizers==0.22.2
"""

import os
import re

import torch
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from vllm import SamplingParams

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 64
GPU_MEMORY_UTILIZATION = 0.5  # 50% — adjust if OOM occurs

DATASET_NAME = "g3lu/addictive_manufacturing_reasoning"
DATASET_SLICE = "train[0:100]"  # 100 samples — 90 train / 10 test
DATASET_SEED = 42
TEST_SIZE = 0.1

OUTPUT_DIR = "outputs"
LORA_SAVE_DIR = "grpo_saved_lora"
MERGED_SAVE_DIR = "model"
TRAINING_LOG_FILE = "training_output.txt"

# GRPO trainer hyper-parameters
LEARNING_RATE = 5e-6
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.99
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1
LR_SCHEDULER = "cosine"
OPTIM = "adamw_8bit"
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 1  # increase to 4 for smoother training
NUM_GENERATIONS = 6         # decrease if OOM
MAX_PROMPT_LENGTH = 1024
MAX_COMPLETION_LENGTH = 1024
MAX_STEPS = 10
SAVE_STEPS = 10
MAX_GRAD_NORM = 0.1

# Inference params for evaluation
SAMPLING_TEMPERATURE = 0.8
SAMPLING_TOP_P = 0.95
SAMPLING_MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

TEST_PROMPT = """
What strategies can be employed to balance the need for simplification
in simulation models with the preservation of critical geometric features,
such as lattice structures or organic shapes, that are characteristic
of additive manufacturing designs and significantly impact
the product's performance, while also considering the limitations imposed
by computational resources and the potential for introducing significant errors
in the results?
"""

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def get_datasets(split: str = "train") -> Dataset:
    """Load and prepare the additive manufacturing reasoning dataset.

    Builds chat-formatted prompts from questions and removes unused columns.
    """
    data_qa = load_dataset(DATASET_NAME, split=split)

    data_qa = data_qa.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {
                'role': 'user',
                'content': (
                    "You are an expert in additive manufacturing.\n\n"
                    "Answer the following question:\n"
                    + x['question']
                    + " You need to carefully review the question and reason before answering."
                ),
            },
        ],
        'answer': x['answer'],
        'db_set': 'addictive_manufacturing_reasoning',
    })

    data_qa = data_qa.remove_columns(['question', 'reason'])
    return data_qa

# ---------------------------------------------------------------------------
# XML answer extraction
# ---------------------------------------------------------------------------

def extract_xml_answer(text: str) -> str:
    """Extract the content within the <answer>...</answer> XML block."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def correctness_reward_func(prompts, completions, answer, db_set, **kwargs) -> list[float]:
    """Reward correctness of the extracted answer relative to ground truth."""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-' * 20,
          f"Question:\\n{q}",
          f"\\nAnswer:\\n{answer[0]}",
          f"\\nResponse:\\n{responses[0]}",
          f"\\nExtracted:\\n{extracted_responses[0]}")

    rewards = []
    for r, a, dt in zip(extracted_responses, answer, db_set):
        if dt == "gsm8k":
            if a in r:
                rewards.append(1.0)
            elif r == a:
                rewards.append(2.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(2.0 if r.lower() == a.strip().lower() else 0.0)

    return rewards


def int_reward_func(completions, db_set, **kwargs) -> list[float]:
    """Reward responses whose format matches the expected type for each dataset."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = []
    for r, dt in zip(extracted_responses, db_set):
        if dt == "gsm8k":
            rewards.append(0.5 if r.isdigit() else 0.0)
        elif dt == "pubmedqa":
            rewards.append(
                0.5 if any(w in r.lower() for w in ('yes', 'no', 'maybe')) else 0.0
            )
        else:
            rewards.append(
                0.5 if any(c in r.lower() for c in ('a', 'b', 'c', 'd')) else 0.0
            )
    return rewards


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks for the exact XML reasoning/answer format."""
    pattern = r"^<reasoning>\\n.*?\\n</reasoning>\\n<answer>\\n.*?\\n</answer>\\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks for a relaxed XML reasoning/answer format."""
    pattern = r"<reasoning>.*?</reasoning>\\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text: str) -> float:
    """Score a response based on how many expected XML tags are present."""
    count = 0.0
    if text.count("<reasoning>\\n") == 1:
        count += 0.125
    if text.count("\\n</reasoning>\\n") == 1:
        count += 0.125
    if text.count("\\n<answer>\\n") == 1:
        count += 0.125
        count -= len(text.split("\\n</answer>\\n")[-1]) * 0.001
    if text.count("\\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Reward function based on XML tag count in completions."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_model_for_grpo():
    """Load the 4-bit quantised model and prepare it for QLoRA + GRPO training."""
    # Enable fast RL (GRPO) in TRL via Unsloth patch
    PatchFastRL("GRPO", FastLanguageModel)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,      # enables vLLM
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    return model, tokenizer


def build_grpo_trainer(model, tokenizer, train_dataset, test_dataset) -> GRPOTrainer:
    """Create a GRPOTrainer with the configured reward functions."""
    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=LEARNING_RATE,
        adam_beta1=ADAM_BETA1,
        adam_beta2=ADAM_BETA2,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        optim=OPTIM,
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        report_to="none",
        output_dir=OUTPUT_DIR,
    )

    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def evaluate_baseline(model, tokenizer) -> str:
    """Run baseline inference (before LoRA) using vLLM fast_generate."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": TEST_PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    sampling_params = SamplingParams(
        temperature=SAMPLING_TEMPERATURE,
        top_p=SAMPLING_TOP_P,
        max_tokens=SAMPLING_MAX_TOKENS,
    )
    output = model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=None,
    )[0].outputs[0].text
    return output


def evaluate_with_lora(model, tokenizer) -> str:
    """Run inference using the saved QLoRA weights via HuggingFace generate."""
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TEST_PROMPT},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    # Use HuggingFace generate since fast_inference is disabled after LoRA merge
    outputs = model.generate(
        **inputs,
        max_new_tokens=SAMPLING_MAX_TOKENS,
        temperature=SAMPLING_TEMPERATURE,
        top_p=SAMPLING_TOP_P,
        do_sample=True,
    )
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip the input prompt from the output
    output = output.replace(text, "")
    return output

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Prepare data, fine-tune with GRPO, then evaluate baseline and LoRA model."""
    # --- Data preparation ---
    dataset = get_datasets(DATASET_SLICE)
    dataset = dataset.shuffle(seed=DATASET_SEED)
    split = dataset.train_test_split(test_size=TEST_SIZE)
    train_dataset = split["train"]
    test_dataset = split["test"]
    print(f"train size: {len(train_dataset)}, test size: {len(test_dataset)}")
    print(train_dataset[1])

    # --- Model & trainer setup ---
    model, tokenizer = load_model_for_grpo()
    trainer = build_grpo_trainer(model, tokenizer, train_dataset, test_dataset)

    # --- Training ---
    # Training output is captured and written to file for later inspection
    trainer.train()
    # Note: in the original notebook %%capture was used; save stdout manually if needed.
    with open(TRAINING_LOG_FILE, "w") as f:
        f.write("Training completed.\n")

    # --- Evaluate baseline (no LoRA) ---
    baseline_output = evaluate_baseline(model, tokenizer)
    print(baseline_output)

    # --- Save QLoRA weights ---
    model.save_pretrained(LORA_SAVE_DIR)

    # --- Evaluate with LoRA weights ---
    lora_output = evaluate_with_lora(model, tokenizer)
    print(lora_output)

    # --- Merge QLoRA weights with base model ---
    model.save_pretrained_merged(MERGED_SAVE_DIR, tokenizer)


if __name__ == "__main__":
    main()
