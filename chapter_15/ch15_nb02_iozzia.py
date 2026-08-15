r"""
Adapt an SLM to reason over a custom domain and dataset using GRPO.

Companion script for chapter 15 of "Domain-Specific Small Language Models"
by Guglielmo Iozzia, Manning Publications, 2026.

Fine-tunes Qwen 2.5 3B Instruct to specialise on a QA task about additive
manufacturing (3D printing) through GRPO + QLoRA via Unsloth and TRL.
GPU is required.

=============================================================================
EDUCATIONAL CONCEPTS DEMONSTRATED:
1. GRPO (Group Relative Policy Optimization):
   - Eliminates the Critic/Value model required by PPO (saving ~50% VRAM).
   - Generates a group of $G$ completions per prompt, evaluates them with reward
     functions, and computes normalized advantage: $A_i = \frac{r_i - \mu_r}{\sigma_r}$.
2. Multi-Objective Reward Functions:
   - Format Rewards: Encourage the model to structure thinking within `<reasoning>` and final outputs within `<answer>`.
   - Correctness Rewards: Score domain accuracy relative to ground truth without requiring human annotators in the loop.
3. Unsloth + QLoRA Efficiency:
   - 4-bit base weights + LoRA rank $r=64$ allows fine-tuning 3B reasoning models on single consumer GPUs (e.g. 16GB VRAM).
=============================================================================

# Install the missing dependencies before running:
# pip install --force-reinstall datasets
# pip uninstall -y huggingface-hub transformers tokenizers
# pip install huggingface-hub==0.34.0 transformers==4.56.2 tokenizers==0.22.2
"""

import os
import re
import sys
from pathlib import Path

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from vllm import SamplingParams

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
    render_device_info,
    render_step,
    render_takeaways,
    status_spinner,
)

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
NUM_GENERATIONS = 6  # decrease if OOM
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
    """Load and prepare the additive manufacturing reasoning dataset."""
    with console.status(f"[bold green]Loading {DATASET_NAME} ({split})..."):
        data_qa = load_dataset(DATASET_NAME, split=split)

        data_qa = data_qa.map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "You are an expert in additive manufacturing.\n\n"
                            "Answer the following question:\n"
                            + x["question"]
                            + " You need to carefully review the question and reason before answering."
                        ),
                    },
                ],
                "answer": x["answer"],
                "db_set": "addictive_manufacturing_reasoning",
            }
        )

        data_qa = data_qa.remove_columns(["question", "reason"])
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
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

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
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = []
    for r, dt in zip(extracted_responses, db_set):
        if dt == "gsm8k":
            rewards.append(0.5 if r.isdigit() else 0.0)
        elif dt == "pubmedqa":
            rewards.append(0.5 if any(w in r.lower() for w in ("yes", "no", "maybe")) else 0.0)
        else:
            rewards.append(0.5 if any(c in r.lower() for c in ("a", "b", "c", "d")) else 0.0)
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
    with console.status(f"[bold green]Loading {MODEL_ID} with Unsloth fast inference..."):
        PatchFastRL("GRPO", FastLanguageModel)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            fast_inference=True,
            max_lora_rank=LORA_RANK,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_RANK,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

    console.print("[bold green]✔[/bold green] Unsloth QLoRA model configured for GRPO.")
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
    output = (
        model.fast_generate(
            [text],
            sampling_params=sampling_params,
            lora_request=None,
        )[0]
        .outputs[0]
        .text
    )
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

    outputs = model.generate(
        **inputs,
        max_new_tokens=SAMPLING_MAX_TOKENS,
        temperature=SAMPLING_TEMPERATURE,
        top_p=SAMPLING_TOP_P,
        do_sample=True,
    )
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output = output.replace(text, "")
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Prepare data, fine-tune with GRPO, then evaluate baseline and LoRA model."""
    render_banner(
        title="GRPO Reinforcement Learning for Domain Reasoning (3D Printing)",
        subtitle="Chapter 15: Domain-Specific Small Language Models",
        metadata={
            "Base Model": MODEL_ID,
            "LoRA Rank": str(LORA_RANK),
            "Trainer": "TRL GRPOTrainer + Unsloth",
        },
        icon="🚀",
    )

    # Step 1: Dataset Ingestion & Split
    render_step(1, "Additive Manufacturing Reasoning Dataset Partitioning", icon="📋")
    dataset = get_datasets(DATASET_SLICE)
    dataset = dataset.shuffle(seed=DATASET_SEED)
    split = dataset.train_test_split(test_size=TEST_SIZE)
    train_dataset = split["train"]
    test_dataset = split["test"]

    columns = [("Dataset Partition", STYLE_PRIMARY, "left"), ("Sample Count", STYLE_SUCCESS, "right")]
    rows = [
        ("Training Partition", str(len(train_dataset))),
        ("Hold-Out Evaluation Partition", str(len(test_dataset))),
    ]
    console.print(create_table("Dataset Partitioning Overview", columns, rows))
    pause()

    # Step 2: Model & GRPO Trainer Setup
    render_step(2, "Initializing 4-bit Unsloth QLoRA Base Model & GRPO Trainer", icon="🧠")
    model, tokenizer = load_model_for_grpo()
    render_device_info("cuda" if torch.cuda.is_available() else "cpu", model=model)
    trainer = build_grpo_trainer(model, tokenizer, train_dataset, test_dataset)

    # Step 3: Baseline Inference (Before GRPO)
    render_step(3, "Evaluating Baseline Pre-GRPO Inference", icon="⚡")
    with status_spinner("Generating baseline unaligned response..."):
        baseline_output = evaluate_baseline(model, tokenizer)
    render_card("Baseline Output (Pre-GRPO)", baseline_output, icon="📄")

    # Step 4: Executing GRPO Reinforcement Learning
    render_step(4, f"Executing GRPO Policy Gradient Optimization ({MAX_STEPS} steps)", icon="🏋️")
    console.print("[bold green]Running GRPO training with multi-objective format & accuracy rewards...[/bold green]")
    trainer.train()

    with open(TRAINING_LOG_FILE, "w") as f:
        f.write("Training completed.\n")
    render_card(
        "Training Status",
        f"GRPO Training completed. Log saved to [text.highlight]{TRAINING_LOG_FILE}[/text.highlight]",
        icon="✔",
    )

    # Step 5: Post-GRPO Reasoning Inference
    render_step(5, "Evaluating Post-GRPO Multi-Step Structured Reasoning", icon="✨")
    model.save_pretrained(LORA_SAVE_DIR)
    with status_spinner("Generating reasoned output with <reasoning> + <answer> structure..."):
        lora_output = evaluate_with_lora(model, tokenizer)

    render_card("Post-GRPO Reasoned Output", lora_output, icon="🎯")

    # Step 6: Saving Merged Model Weights
    render_step(6, "Merging LoRA Adapters into Standalone Base Model", icon="💾")
    with status_spinner(f"Saving merged 16-bit weights to '{MERGED_SAVE_DIR}'..."):
        model.save_pretrained_merged(MERGED_SAVE_DIR, tokenizer)
    render_card(
        "Merged Checkpoint",
        f"Standalone model weights saved to [text.highlight]{MERGED_SAVE_DIR}[/text.highlight]",
        icon="✔",
    )

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Why GRPO replaces PPO for SLMs",
                "Standard PPO requires training both an Actor and a Critic model simultaneously. GRPO eliminates the Critic by computing baseline rewards across a group of sampled responses (G=6), slashing VRAM by ~50%.",
            ),
            (
                "Emergence of Self-Correction",
                "By rewarding format structure (<reasoning> tags) and factual accuracy, the model learns to 'think before answering' without needing expensive supervised Chain-of-Thought human datasets.",
            ),
            (
                "QLoRA + Unsloth Synergy",
                "Combining 4-bit quantization with LoRA and Triton kernel fusions enables full Reinforcement Learning fine-tuning on consumer-grade single GPU hardware.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
