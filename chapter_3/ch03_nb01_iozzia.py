"""Fine-tune GPT-2 Small to generate Manim Python animation code.

Companion script for Chapter 3 of "Domain Specific LLMs in Action"
by Guglielmo Iozzia, Manning Publications, 2024.

Uses Optuna-backed hyperparameter search via HF Trainer, followed by a
full training run with the best found hyperparameters. Inference is
evaluated by writing generated Manim snippets to a CSV file.
A GPU is required.

# Install missing requirements first (Colab / fresh env):
# !pip install optuna
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
from __future__ import annotations
import csv

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import torch
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_NAME = "Edoh/manim_python"
MODEL_NAME = "openai-community/gpt2"
OUTPUT_DIR = "./gpt2-manim-python-finetuned"
OUTPUT_CSV = "gpt2_manim_python_test_outputs.csv"

# Training
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LOGGING_STEPS = 100
SAVE_TOTAL_LIMIT = 2
EARLY_STOPPING_PATIENCE = 2
VALIDATION_SPLIT = 0.1
MAX_TOKENIZED_LENGTH = 512

# Hyperparameter search (Optuna)
HP_TRIALS = 3
HP_LR_MIN = 1e-5
HP_LR_MAX = 5e-4
HP_BATCH_SIZES = [2, 4, 8]
HP_WEIGHT_DECAY_MAX = 0.3
HP_EPOCHS_MIN = 3
HP_EPOCHS_MAX = 6
HP_WARMUP_MAX = 500
HP_GRAD_ACCUM_CHOICES = [1, 2, 4]

# Inference / generation
GEN_MAX_LENGTH = 150
GEN_NUM_BEAMS = 5
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9
GEN_REPETITION_PENALTY = 1.2
GEN_NO_REPEAT_NGRAM_SIZE = 2


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_and_tokenise_dataset(tokenizer: GPT2Tokenizer):
    """Load the Manim dataset, tokenise, and split off a validation set."""
    dataset = load_dataset(DATASET_NAME)
    print(dataset["train"][0])

    tokenized_datasets = dataset.map(
        lambda examples: preprocess_data(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Reserve 10% of training data for validation
    train_val_split = tokenized_datasets["train"].train_test_split(test_size=VALIDATION_SPLIT)
    tokenized_datasets["train"] = train_val_split["train"]
    tokenized_datasets["validation"] = train_val_split["test"]

    return dataset, tokenized_datasets


def preprocess_data(examples: dict, tokenizer: GPT2Tokenizer) -> dict:
    """Concatenate instruction/output pairs, tokenise, and set labels == input_ids."""
    inputs = [
        f"Instruction: {instr}\nOutput: {out}"
        for instr, out in zip(examples["instruction"], examples["output"])
    ]
    tokenized = tokenizer(inputs, truncation=True, max_length=MAX_TOKENIZED_LENGTH, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

def hp_space(trial) -> dict:
    """Define the Optuna hyperparameter search space."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", HP_LR_MIN, HP_LR_MAX, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", HP_BATCH_SIZES),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, HP_WEIGHT_DECAY_MAX),
        "num_train_epochs": trial.suggest_int("num_train_epochs", HP_EPOCHS_MIN, HP_EPOCHS_MAX),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, HP_WARMUP_MAX),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", HP_GRAD_ACCUM_CHOICES),
    }


def run_hyperparameter_search(training_args: TrainingArguments, tokenized_datasets, tokenizer, data_collator):
    """Run Optuna hyperparameter search; return the best run."""

    def model_init():
        # A fresh model is created for each trial; subsequent trials use the cached weights.
        return GPT2LMHeadModel.from_pretrained(MODEL_NAME, device_map="auto")

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    best_run = trainer.hyperparameter_search(
        direction="minimize",
        backend="optuna",
        n_trials=HP_TRIALS,
        hp_space=hp_space,
        compute_objective=lambda metrics: metrics["eval_loss"],
    )
    print("Best hyperparameters found:", best_run.hyperparameters)
    return best_run


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_with_best_hyperparams(best_run, training_args: TrainingArguments, tokenized_datasets, tokenizer, data_collator) -> Trainer:
    """Apply the best hyperparameters to training_args and train the final model."""
    for key, value in best_run.hyperparameters.items():
        setattr(training_args, key, value)

    def model_init():
        return GPT2LMHeadModel.from_pretrained(MODEL_NAME, device_map="auto")

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    return trainer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_finetuned_model(model_dir: str) -> tuple[GPT2LMHeadModel, GPT2Tokenizer, torch.device]:
    """Load the fine-tuned model and tokenizer from disk to GPU memory."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_output(
    instruction: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    max_length: int = GEN_MAX_LENGTH,
    num_beams: int = GEN_NUM_BEAMS,
    temperature: float = GEN_TEMPERATURE,
    top_p: float = GEN_TOP_P,
    repetition_penalty: float = GEN_REPETITION_PENALTY,
) -> str:
    """Generate output text given an instruction using beam search and nucleus sampling.

    Args:
        instruction: The input instruction prompt.
        model: Fine-tuned GPT-2 model.
        tokenizer: Companion tokenizer.
        device: Torch device to run inference on.
        max_length: Maximum length of generated sequence (including prompt).
        num_beams: Number of beams for beam search.
        temperature: Sampling temperature; lower is less random.
        top_p: Nucleus sampling probability threshold.
        repetition_penalty: Penalty for repeated tokens (>1.0 discourages repetition).

    Returns:
        Generated output text (the portion after "Output:").
    """
    prompt = f"Instruction: {instruction}\nOutput:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=num_beams,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True,
        no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM_SIZE,
    )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    output_start = generated_text.find("Output:")
    if output_start != -1:
        return generated_text[output_start + len("Output:"):].strip()
    return generated_text.strip()


def run_inference_on_test_set(dataset, model, tokenizer, device) -> None:
    """Run the fine-tuned model on every test sample and write results to CSV."""
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["instruction", "reference_output", "generated_output"],
        )
        writer.writeheader()

        for example in dataset["test"]:
            instruction = example["instruction"]
            reference_output = example["output"]
            generated = generate_output(instruction, model, tokenizer, device)
            writer.writerow({
                "instruction": instruction,
                "reference_output": reference_output,
                "generated_output": generated,
            })

    print(f"Inference complete. Results saved to {OUTPUT_CSV}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    """End-to-end pipeline: data prep → HP search → training → inference."""
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # Required: GPT-2 has no dedicated pad token

    raw_dataset, tokenized_datasets = load_and_tokenise_dataset(tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy=EVAL_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="none",
    )

    best_run = run_hyperparameter_search(training_args, tokenized_datasets, tokenizer, data_collator)
    train_with_best_hyperparams(best_run, training_args, tokenized_datasets, tokenizer, data_collator)

    model, inf_tokenizer, device = load_finetuned_model(OUTPUT_DIR)
    run_inference_on_test_set(raw_dataset, model, inf_tokenizer, device)


if __name__ == "__main__":
    main()
