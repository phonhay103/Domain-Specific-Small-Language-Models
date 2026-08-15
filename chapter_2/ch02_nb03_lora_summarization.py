"""Dialogue Summarization with FLAN-T5-base and PEFT LoRA.

Companion script for Chapter 2 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Demonstrates parameter-efficient fine-tuning (PEFT) using LoRA (Low-Rank Adaptation)
on the SAMSum dataset for dialogue summarization, evaluated with ROUGE metrics.
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import time as current_time
from typing import Any

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Third-party
import evaluate
import numpy as np
import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from peft import AutoPeftModelForSeq2SeqLM, LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Common functional & UI utilities
from common.functional import calculate_speedup, format_percentage
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
    render_device_info,
    render_step,
    render_takeaways,
    render_training_metrics_table,
    silence_hf_logs,
    silence_trainer,
    status_spinner,
)


# ---------------------------------------------------------------------------
# Immutable Domain Records & Constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoRAParamStats:
    """Immutable representation of model parameter counts under LoRA."""

    trainable_params: int
    total_params: int
    trainable_ratio: float


@dataclass(frozen=True)
class LengthPercentiles:
    """Immutable representation of sequence length quantiles."""

    percentile: int
    dialogue_length: int
    summary_length: int


MODEL_ID = "google/flan-t5-base"
DATASET_ID = "knkarthick/samsum"
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 50
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
OUTPUT_DIR = "experiments"
EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


# ---------------------------------------------------------------------------
# Pure Functions & Data Transformations
# ---------------------------------------------------------------------------
def compute_length_percentiles(
    dataset: DatasetDict,
    tokenizer: Any,
    percentiles: Sequence[int] = (80, 85, 90, 95, 100),
) -> tuple[LengthPercentiles, ...]:
    """Pure analysis: calculate token length distributions across train/test splits."""
    combined = concatenate_datasets([dataset["train"], dataset["test"]])
    num_cores = os.cpu_count() or 1

    tokenized_inputs = combined.map(
        lambda x: tokenizer(x["dialogue"], truncation=True),
        batched=True,
        remove_columns=["dialogue", "summary"],
        num_proc=num_cores,
    )
    input_lengths = [len(x) for x in tokenized_inputs["input_ids"]]

    tokenized_targets = combined.map(
        lambda x: tokenizer(x["summary"], truncation=True),
        batched=True,
        remove_columns=["dialogue", "summary"],
        num_proc=num_cores,
    )
    target_lengths = [len(x) for x in tokenized_targets["input_ids"]]

    return tuple(
        LengthPercentiles(
            percentile=p,
            dialogue_length=int(np.percentile(input_lengths, p)),
            summary_length=int(np.percentile(target_lengths, p)),
        )
        for p in percentiles
    )


def format_dialogue_prompt(dialogue: str) -> str:
    """Pure string formatter for FLAN-T5 instruction."""
    return f"Summarize the following conversation.\n\n{dialogue}"


def preprocess_batch(sample: Mapping[str, Any], tokenizer: Any, padding: str = "max_length") -> dict[str, Any]:
    """Pure batch preprocessor: prepends instruction prompt and encodes targets."""
    inputs = [format_dialogue_prompt(item) for item in sample["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=MAX_SOURCE_LENGTH, padding=padding, truncation=True)
    labels = tokenizer(
        text_target=sample["summary"],
        max_length=MAX_TARGET_LENGTH,
        padding=padding,
        truncation=True,
    )

    if padding == "max_length":
        labels["input_ids"] = [
            [(label_id if label_id != tokenizer.pad_token_id else -100) for label_id in label_seq]
            for label_seq in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def create_lora_config() -> LoraConfig:
    """Pure factory for LoRA PEFT configuration."""
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q", "v"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )


def create_training_args(output_dir: str) -> Seq2SeqTrainingArguments:
    """Pure factory for seq2seq training arguments optimized for RTX 5060 Ti 16GB."""
    has_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_strategy="steps",
        logging_steps=10,
        fp16=torch.cuda.is_available() and not has_bf16,
        bf16=has_bf16,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        dataloader_pin_memory=torch.cuda.is_available(),
        save_strategy="no",
        disable_tqdm=False,
        remove_unused_columns=False,
    )


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_dataset_stats(dataset: DatasetDict) -> None:
    """Render dataset partition sizes."""
    columns = [
        ("Dataset Split", STYLE_PRIMARY, "left"),
        ("Dialogue Count", STYLE_NUMBER, "right"),
    ]
    rows = [(split_name.capitalize(), f"{len(split_data):,}") for split_name, split_data in dataset.items()]
    console.print(create_table("SAMSum Dataset Partition Statistics", columns, rows))
    pause()


def render_percentiles_table(stats: Sequence[LengthPercentiles]) -> None:
    """Render length percentiles in an eye-friendly table."""
    columns = [
        ("Percentile", STYLE_PRIMARY, "center"),
        ("Dialogue Token Length", STYLE_SECONDARY, "right"),
        ("Summary Token Length", STYLE_WARNING, "right"),
    ]
    rows = [(f"{s.percentile}%", str(s.dialogue_length), str(s.summary_length)) for s in stats]
    console.print(create_table("Token Length Distribution (Percentiles)", columns, rows))
    pause()


def render_param_stats_table(stats: LoRAParamStats) -> None:
    """Render parameter efficiency table."""
    columns = [
        ("Parameter Metric", STYLE_PRIMARY, "left"),
        ("Value", STYLE_SUCCESS, "right"),
    ]
    rows = [
        ("Trainable LoRA Parameters", f"{stats.trainable_params:,}"),
        ("Total Model Parameters", f"{stats.total_params:,}"),
        ("Trainable Proportion", f"{stats.trainable_ratio:.2f}%"),
        ("Frozen Base Proportion", f"{100.0 - stats.trainable_ratio:.2f}%"),
    ]
    console.print(create_table("LoRA Parameter Efficiency", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute functional dialogue summarization with FLAN-T5 and LoRA."""
    silence_hf_logs()

    # Parse CLI arguments for checkpoint loading
    import glob

    checkpoint_path = None
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--checkpoint="):
            checkpoint_path = arg.split("=", 1)[1]
        elif arg == "--checkpoint":
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("-"):
                checkpoint_path = sys.argv[i + 1]
            else:
                checkpoints = sorted(glob.glob(f"{OUTPUT_DIR}/peft_model_*"))
                if checkpoints:
                    checkpoint_path = checkpoints[-1]
                else:
                    console.print(f"[bold red]Error: No checkpoints found in {OUTPUT_DIR}[/bold red]")
                    sys.exit(1)

    render_banner(
        title="Dialogue Summarization with FLAN-T5 & PEFT LoRA",
        subtitle="Chapter 2: Domain-Specific Small Language Models",
        metadata={
            "Model": MODEL_ID,
            "Dataset": DATASET_ID,
            "LoRA Rank": str(LORA_R),
            "LoRA Alpha": str(LORA_ALPHA),
            "Mode": f"Evaluation ({checkpoint_path})" if checkpoint_path else "Full Training Pipeline",
        },
        icon="🚀",
    )

    # Step 1: Loading SAMSum Dataset
    render_step(1, "Loading SAMSum Dataset & Tokenizer", icon="📋")
    with status_spinner(f"Loading '{DATASET_ID}' dataset..."):
        dataset = load_dataset(DATASET_ID)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path if checkpoint_path else MODEL_ID)

    render_dataset_stats(dataset)

    with status_spinner("Computing sequence length percentiles across splits..."):
        percentile_stats = compute_length_percentiles(dataset, tokenizer)
    render_percentiles_table(percentile_stats)

    if checkpoint_path:
        # Step 3: Loading Saved PEFT LoRA Model
        render_step(3, "Loading Saved PEFT LoRA Model", icon="🧠")
        with status_spinner(f"Loading LoRA adapters from '{checkpoint_path}'..."):
            peft_model = AutoPeftModelForSeq2SeqLM.from_pretrained(
                checkpoint_path,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if torch.cuda.is_available() and hasattr(torch, "compile"):
                peft_model = torch.compile(peft_model)
        render_device_info(peft_model.device, model=peft_model)

        trainable_p, total_p = peft_model.get_nb_trainable_parameters()
        stats = LoRAParamStats(
            trainable_params=trainable_p,
            total_params=total_p,
            trainable_ratio=format_percentage(trainable_p, total_p),
        )
        render_param_stats_table(stats)
    else:
        # Step 2: Tokenizing & Data Collation
        render_step(2, "Preprocessing & Collating Seq2Seq Batches", icon="⚙️")
        with status_spinner("Encoding dialogues and target summaries..."):
            tokenized_dataset = dataset.map(
                lambda s: preprocess_batch(s, tokenizer),
                batched=True,
                remove_columns=["dialogue", "summary", "id"],
                num_proc=os.cpu_count() or 1,
            )
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

        # Step 3: Initializing PEFT LoRA Model
        render_step(3, "Configuring LoRA Low-Rank Decomposition Adapters", icon="🧠")
        with status_spinner(f"Injecting LoRA adapters into '{MODEL_ID}' attention layers..."):
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_ID,
                device_map="auto" if torch.cuda.is_available() else None,
                attn_implementation="sdpa" if torch.cuda.is_available() else None,
            )
            lora_config = create_lora_config()
            peft_model = get_peft_model(base_model, lora_config)
            if torch.cuda.is_available() and hasattr(torch, "compile"):
                peft_model = torch.compile(peft_model)
        render_device_info(peft_model.device, model=peft_model)

        trainable_p, total_p = peft_model.get_nb_trainable_parameters()
        stats = LoRAParamStats(
            trainable_params=trainable_p,
            total_params=total_p,
            trainable_ratio=format_percentage(trainable_p, total_p),
        )
        render_param_stats_table(stats)

        # Step 4: Training with Seq2SeqTrainer
        render_step(4, "Executing PEFT LoRA Training Loop", icon="🏋️")
        training_args = create_training_args(OUTPUT_DIR)
        trainer = Seq2SeqTrainer(
            model=peft_model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
        )
        console.print(f"[bold green]Training LoRA adapter weights for {EPOCHS} epoch(s)...[/bold green]")
        train_output = trainer.train()

        render_training_metrics_table(trainer.state.log_history, title="PEFT LoRA Training Progression")

        save_dir = f"{OUTPUT_DIR}/peft_model_{int(current_time())}"

        # Unwrap torch.compile wrapper if exists
        uncompiled_model = peft_model._orig_mod if hasattr(peft_model, "_orig_mod") else peft_model
        uncompiled_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        render_card(
            "Training Status & Checkpoint Saved",
            (
                f"[status.success]PEFT LoRA fine-tuning completed successfully.[/status.success]\n"
                f"[text.muted]Total Runtime:[/text.muted] [brand.secondary]{train_output.metrics.get('train_runtime', 0):.2f}s[/brand.secondary]  •  "
                f"[text.muted]Final Train Loss:[/text.muted] [status.warning]{train_output.metrics.get('train_loss', 0):.4f}[/status.warning]\n"
                f"[text.muted]Checkpoint Saved To:[/text.muted] [text.highlight]{save_dir}[/text.highlight]"
            ),
            icon="💾",
        )

    # Step 5: Sample Inference Preview
    render_step(5, "Generating Sample Summarization", icon="💬")
    sample_dialogue = dataset["test"][0]["dialogue"]
    reference_summary = dataset["test"][0]["summary"]

    prompt = format_dialogue_prompt(sample_dialogue)
    inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)
    outputs = peft_model.generate(**inputs, max_new_tokens=MAX_TARGET_LENGTH)
    generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    render_card("Input Dialogue", sample_dialogue, icon="📥")
    render_card(
        title="Summarization Comparison",
        content=(
            f"[status.success]Model Summary:[/status.success] [text.highlight]{generated_summary}[/text.highlight]\n\n"
            f"[status.warning]Reference Ground Truth:[/status.warning] [text.main]{reference_summary}[/text.main]"
        ),
        icon="✨",
    )

    # Step 6: ROUGE Evaluation
    render_step(6, "Evaluating ROUGE Linguistic Overlap on Test Split", icon="📊")
    with status_spinner("Computing ROUGE scores on test set..."):
        rouge_metric = evaluate.load("rouge")
        # Run inference on subset for concise evaluation
        test_subset = dataset["test"].select(range(min(10, len(dataset["test"]))))
        predictions = []
        references = []
        for item in test_subset:
            p = format_dialogue_prompt(item["dialogue"])
            inp = tokenizer(p, return_tensors="pt").to(peft_model.device)
            out = peft_model.generate(**inp, max_new_tokens=MAX_TARGET_LENGTH)
            predictions.append(tokenizer.decode(out[0], skip_special_tokens=True))
            references.append(item["summary"])

        rouge_scores = rouge_metric.compute(predictions=predictions, references=references, use_stemmer=True)

    if rouge_scores:
        columns = [("ROUGE Metric", STYLE_PRIMARY, "left"), ("F1 Score", STYLE_SUCCESS, "right")]
        rows = [(k.upper(), f"{v * 100:.2f}%") for k, v in rouge_scores.items()]
        console.print(create_table("ROUGE Overlap Metrics", columns, rows))
        pause()

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Why LoRA works",
                "Over-parameterized models have weight matrices with low 'intrinsic rank'. Adapting only small low-rank matrices A and B captures task-specific changes without catastrophic forgetting.",
            ),
            (
                "Storage & VRAM Savings",
                "Instead of storing ~1 GB of full model checkpoints per fine-tuned task, a LoRA checkpoint is typically only ~10-20 MB.",
            ),
            (
                "ROUGE Score Metrics",
                "ROUGE-1/2 scores evaluate vocabulary precision, while ROUGE-L evaluates the Longest Common Subsequence, confirming whether sequential grammatical structures are preserved.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
