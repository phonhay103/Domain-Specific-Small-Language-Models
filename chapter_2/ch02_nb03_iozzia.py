"""Fine-tune FLAN-T5 Small for text summarisation using LoRA (PEFT).

Companion script for Chapter 2 of "Domain Specific LLMs in Action"
by Guglielmo Iozzia, Manning Publications, 2025.

Introduces LoRA (Low-Rank Adaptation) via the HF PEFT library applied to
FLAN-T5 Small, trained on the SAMSum dialogue-summarisation dataset.
Evaluation uses ROUGE metrics.  A GPU is required.

# Install missing requirements first (Colab / fresh env):
# !pip install datasets peft accelerate bitsandbytes evaluate rouge_score py7zr
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
from __future__ import annotations
import locale
from random import randrange

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import evaluate
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_NAME = "knkarthick/samsum"
MODEL_ID = "google/flan-t5-small"
OUTPUT_DIR = "lora-flan-t5-small"
LORA_MODEL_ID = "flan_t5_lora"
TRAIN_DATA_PATH = "data/train"
EVAL_DATA_PATH = "data/eval"

# Percentile thresholds for computing max sequence lengths
INPUT_LENGTH_PERCENTILE = 85
TARGET_LENGTH_PERCENTILE = 90

# LoRA configuration
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q", "v"]
LORA_DROPOUT = 0.05

# Training hyperparameters
LEARNING_RATE = 1e-3
NUM_TRAIN_EPOCHS = 3
LOGGING_STEPS = 500

# Inference
MAX_NEW_TOKENS_INFERENCE = 10
TOP_P = 0.9
MAX_EVAL_TARGET_LENGTH = 50

LABEL_PAD_TOKEN_ID = -100


# ---------------------------------------------------------------------------
# Locale fix (needed in some Colab/container environments)
# ---------------------------------------------------------------------------

def _patch_locale() -> None:
    """Wrap locale.getpreferredencoding to avoid encoding issues in some VMs."""
    original = locale.getpreferredencoding
    locale.getpreferredencoding = lambda do_raise=True: original()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_raw_dataset():
    """Load the SAMSum dataset from the HF Hub and print split sizes."""
    dataset = load_dataset(DATASET_NAME, trust_remote_code=True)
    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")
    return dataset


def compute_max_lengths(dataset, tokenizer) -> tuple[int, int]:
    """Compute max source and target lengths using percentile thresholds.

    For the input, we take the 85th percentile of the max length for better utilization.
    For the target, we take the 90th percentile of the max length for better utilization.
    """
    combined = concatenate_datasets([dataset["train"], dataset["test"]])

    tokenized_inputs = [
        tokenizer(text=ex["dialogue"], truncation=True)["input_ids"]
        for ex in combined
        if ex["dialogue"] is not None
    ]
    max_source_length = int(np.percentile([len(x) for x in tokenized_inputs], INPUT_LENGTH_PERCENTILE))
    print(f"Max source length: {max_source_length}")

    tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x["summary"], truncation=True),
        batched=True,
        remove_columns=["dialogue", "summary"],
    )
    max_target_length = int(np.percentile([len(x) for x in tokenized_targets["input_ids"]], TARGET_LENGTH_PERCENTILE))
    print(f"Max target length: {max_target_length}")

    return max_source_length, max_target_length


def build_preprocess_fn(tokenizer, max_source_length: int, max_target_length: int):
    """Return a batched preprocessing function closed over tokenizer and lengths."""

    def preprocess_function(sample: dict, padding: str = "max_length") -> dict:
        """Tokenise dialogue/summary pairs; replace pad tokens in labels with -100."""
        # Filter out examples where dialogue is None and keep corresponding summaries
        processed = [
            (d, s)
            for d, s in zip(sample["dialogue"], sample["summary"])
            if d is not None
        ]
        inputs = [f"summarize: {d}" for d, _ in processed]
        labels_text = [s for _, s in processed]

        model_inputs = tokenizer(
            inputs, max_length=max_source_length, padding=padding, truncation=True
        )
        labels = tokenizer(
            text_target=labels_text,
            max_length=max_target_length,
            padding=padding,
            truncation=True,
        )

        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else LABEL_PAD_TOKEN_ID) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function


def prepare_tokenized_dataset(dataset, tokenizer, max_source_length: int, max_target_length: int):
    """Tokenise, save to disk, and return the tokenized dataset."""
    preprocess_fn = build_preprocess_fn(tokenizer, max_source_length, max_target_length)
    tokenized_dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=["dialogue", "summary", "id"],
    )
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

    tokenized_dataset["train"].save_to_disk(TRAIN_DATA_PATH)
    tokenized_dataset["test"].save_to_disk(EVAL_DATA_PATH)
    return tokenized_dataset


# ---------------------------------------------------------------------------
# LoRA fine-tuning
# ---------------------------------------------------------------------------

def build_lora_model(model_id: str):
    """Load the base model in 8-bit, attach LoRA adapters, and return it."""
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, load_in_8bit=True, device_map="auto"
    )

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def fine_tune(model, tokenizer, tokenized_dataset) -> Seq2SeqTrainer:
    """Set up Seq2SeqTrainer and run fine-tuning; return the trainer."""
    # At the end of the execution above, trainable params should be <1% of total.
    # Training process is the same as regular LLM training;
    # the key difference is the LoRA-adapted model.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=LABEL_PAD_TOKEN_ID,
        pad_to_multiple_of=8,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        auto_find_batch_size=True,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        save_strategy="no",
        report_to="tensorboard",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
    )
    model.config.use_cache = False
    trainer.train()
    return trainer


def save_model(trainer: Seq2SeqTrainer, tokenizer) -> None:
    """Persist the fine-tuned LoRA model and tokenizer to disk."""
    trainer.model.save_pretrained(LORA_MODEL_ID)
    tokenizer.save_pretrained(LORA_MODEL_ID)


# ---------------------------------------------------------------------------
# Inference & evaluation
# ---------------------------------------------------------------------------

def load_inference_model(lora_model_id: str):
    """Reload the base model and merge LoRA weights for inference."""
    config = PeftConfig.from_pretrained(lora_model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path, load_in_8bit=True, device_map={"": 0}
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, lora_model_id, device_map={"": 0})
    model.eval()
    return model, tokenizer


def run_sample_inference(model, tokenizer, dataset) -> None:
    """Summarise a random test dialogue and print the result."""
    sample = dataset["test"][randrange(len(dataset["test"]))]
    input_ids = tokenizer(
        sample["dialogue"], return_tensors="pt", truncation=True
    ).input_ids.cuda()
    outputs = model.generate(
        input_ids=input_ids, max_new_tokens=MAX_NEW_TOKENS_INFERENCE,
        do_sample=True, top_p=TOP_P,
    )
    print(f"input sentence: {sample['dialogue']}\n{'---' * 20}")
    print(f"summary:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")


def evaluate_peft_model(model, tokenizer, sample: dict, max_target_length: int = MAX_EVAL_TARGET_LENGTH) -> tuple[str, str]:
    """Generate a summary for one tokenized sample and decode both prediction and label."""
    outputs = model.generate(
        input_ids=sample["input_ids"].unsqueeze(0).cuda(),
        do_sample=True,
        top_p=TOP_P,
        max_new_tokens=max_target_length,
    )
    prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
    labels = np.where(sample["labels"] != LABEL_PAD_TOKEN_ID, sample["labels"], tokenizer.pad_token_id)
    labels = tokenizer.decode(labels, skip_special_tokens=True)
    return prediction, labels


def evaluate_rouge(model, tokenizer) -> None:
    """Evaluate the fine-tuned model on the saved eval set using ROUGE."""
    metric = evaluate.load("rouge")
    test_dataset = load_from_disk(EVAL_DATA_PATH).with_format("torch")

    predictions, references = [], []
    for sample in tqdm(test_dataset):
        p, l = evaluate_peft_model(model, tokenizer, sample)
        predictions.append(p)
        references.append(l)

    rouge = metric.compute(predictions=predictions, references=references, use_stemmer=True)
    print(f"Rogue1: {rouge['rouge1'] * 100:2f}%")
    print(f"rouge2: {rouge['rouge2'] * 100:2f}%")
    print(f"rougeL: {rouge['rougeL'] * 100:2f}%")
    print(f"rougeLsum: {rouge['rougeLsum'] * 100:2f}%")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    """End-to-end pipeline: data prep → LoRA fine-tuning → inference → ROUGE eval."""
    _patch_locale()

    dataset = load_raw_dataset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    max_source_length, max_target_length = compute_max_lengths(dataset, tokenizer)
    tokenized_dataset = prepare_tokenized_dataset(dataset, tokenizer, max_source_length, max_target_length)

    model = build_lora_model(MODEL_ID)
    trainer = fine_tune(model, tokenizer, tokenized_dataset)
    save_model(trainer, tokenizer)

    inf_model, inf_tokenizer = load_inference_model(LORA_MODEL_ID)
    run_sample_inference(inf_model, inf_tokenizer, dataset)
    evaluate_rouge(inf_model, inf_tokenizer)


if __name__ == "__main__":
    main()
