"""Synthetic Dataset Generation and Hyperparameter Optimization with Optuna.

Companion script for Chapter 3 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Demonstrates generating synthetic Manim code, fine-tuning Qwen 2.5 0.5B,
and searching for optimal hyperparameters using Optuna Bayesian optimization.
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Third-party
import optuna
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# Common functional & UI utilities
from common.functional import map_tuple
from common.ui import (
    STYLE_INDEX,
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
    render_step,
    render_takeaways,
    status_spinner,
)


# ---------------------------------------------------------------------------
# Immutable Domain Records & Constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ManimExample:
    """Immutable prompt-code pair for animation generation."""

    prompt: str
    code: str


@dataclass(frozen=True)
class OptimalHyperparameters:
    """Immutable container for Optuna best trial results."""

    best_loss: float
    learning_rate: float
    lora_r: int
    num_epochs: int


MODEL_ID = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "./output_dir"
N_OPTUNA_TRIALS = 3
EVAL_PROMPT = "Create a red circle that moves to the right"

SAMPLE_MANIM_DATA: tuple[ManimExample, ...] = (
    ManimExample(
        prompt="Create a red circle that scales up and fades out.",
        code="class ScalingCircle(Scene):\n    def construct(self):\n        c = Circle(color=RED)\n        self.play(Create(c))\n        self.play(c.animate.scale(2))\n        self.play(FadeOut(c))",
    ),
    ManimExample(
        prompt="Draw a blue square and rotate it 90 degrees.",
        code="class RotatingSquare(Scene):\n    def construct(self):\n        s = Square(color=BLUE)\n        self.play(Create(s))\n        self.play(Rotate(s, PI/2))",
    ),
    ManimExample(
        prompt="Show a green triangle and transform it into a circle.",
        code="class TriangleToCircle(Scene):\n    def construct(self):\n        t = Triangle(color=GREEN)\n        c = Circle(color=GREEN)\n        self.play(Create(t))\n        self.play(Transform(t, c))",
    ),
    ManimExample(
        prompt="Display the text 'Hello World' with a write animation.",
        code="class WriteText(Scene):\n    def construct(self):\n        text = Text('Hello World')\n        self.play(Write(text))",
    ),
)


# ---------------------------------------------------------------------------
# Pure Functions & Data Preparation
# ---------------------------------------------------------------------------
def format_manim_instruction(example: ManimExample | Mapping[str, str]) -> str:
    """Pure string constructor for prompt-to-code instruction tuning."""
    prompt = example.prompt if isinstance(example, ManimExample) else example["prompt"]
    code = example.code if isinstance(example, ManimExample) else example["code"]
    return f"Instruction: Generate Manim Python code for the following animation.\nPrompt: {prompt}\nCode:\n{code}"


def create_hf_dataset(examples: Sequence[ManimExample]) -> Dataset:
    """Pure transformation from immutable examples into a Hugging Face Dataset."""
    records = [{"prompt": ex.prompt, "code": ex.code} for ex in examples]
    return Dataset.from_pandas(pd.DataFrame(records))


def tokenize_prompt_code_pair(sample: Mapping[str, Any], tokenizer: Any, max_length: int = 512) -> dict[str, Any]:
    """Pure tokenization mapping function."""
    text = format_manim_instruction(sample)
    tokens = tokenizer(text, truncation=True, max_length=max_length, padding=False)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def run_optuna_objective(
    trial: optuna.Trial,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: Any,
) -> float:
    """Objective function for Bayesian hyperparameter search."""
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
    num_epochs = trial.suggest_int("num_train_epochs", 1, 3)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/trial_{trial.number}",
        learning_rate=lr,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="epoch",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    trainer.train()
    eval_results = trainer.evaluate()
    return float(eval_results["eval_loss"])


def generate_code_inference(model: Any, tokenizer: Any, prompt: str) -> str:
    """Pure generative inference wrapper."""
    input_text = f"Instruction: Generate Manim Python code for the following animation.\nPrompt: {prompt}\nCode:\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.2)
    return str(tokenizer.decode(outputs[0], skip_special_tokens=True))


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_dataset_preview(examples: Sequence[ManimExample]) -> None:
    """Render preview table of synthetic dataset."""
    columns = [
        ("Animation Prompt", STYLE_PRIMARY, "left"),
        ("Manim Code Snippet", STYLE_TEXT, "left"),
    ]
    rows = [(ex.prompt, ex.code[:80] + "...") for ex in examples[:2]]
    console.print(create_table("Synthetic Manim Dataset Preview", columns, rows))
    pause()


def render_hyperparameters_table(best: OptimalHyperparameters) -> None:
    """Render optimal hyperparameters table."""
    columns = [
        ("Hyperparameter", STYLE_PRIMARY, "left"),
        ("Optimal Value", STYLE_SUCCESS, "right"),
    ]
    rows = [
        ("Best Validation Loss", f"{best.best_loss:.4f}"),
        ("Learning Rate", f"{best.learning_rate:.2e}"),
        ("LoRA Rank (r)", str(best.lora_r)),
        ("Training Epochs", str(best.num_epochs)),
    ]
    console.print(create_table("Optuna Hyperparameter Search Results", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute synthetic data and Optuna optimization pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    render_banner(
        title="Synthetic Data Generation & Optuna Hyperparameter Tuning",
        subtitle="Chapter 3: Domain-Specific Small Language Models",
        metadata={
            "Base Model": MODEL_ID,
            "Optuna Trials": str(N_OPTUNA_TRIALS),
            "Domain": "Manim Python Animation Code",
        },
        icon="🚀",
    )

    # Step 1: Creating Synthetic Dataset
    render_step(1, "Creating Synthetic Manim Code Dataset", icon="📋")
    raw_dataset = create_hf_dataset(SAMPLE_MANIM_DATA)
    with status_spinner(f"Loading tokenizer for '{MODEL_ID}'..."):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    render_dataset_preview(SAMPLE_MANIM_DATA)

    # Step 2: Tokenizing Dataset Splits
    render_step(2, "Tokenizing & Splitting Train/Validation Sets", icon="⚙️")
    tokenized_ds = raw_dataset.map(lambda s: tokenize_prompt_code_pair(s, tokenizer), batched=False)
    split = tokenized_ds.train_test_split(test_size=0.25, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    render_card(
        title="Dataset Split Counts",
        content=(
            f"[text.muted]Training Samples:[/text.muted] [text.highlight]{len(train_ds)}[/text.highlight]\n"
            f"[text.muted]Validation Samples:[/text.muted] [text.highlight]{len(eval_ds)}[/text.highlight]"
        ),
        icon="✔",
    )

    # Step 3: Optuna Optimization
    render_step(3, "Executing Bayesian Hyperparameter Search", icon="🔍")
    study = optuna.create_study(direction="minimize")
    with status_spinner(f"Running {N_OPTUNA_TRIALS} Bayesian optimization trials..."):
        study.optimize(
            lambda t: run_optuna_objective(t, train_ds, eval_ds, tokenizer),
            n_trials=N_OPTUNA_TRIALS,
        )

    best_config = OptimalHyperparameters(
        best_loss=study.best_value,
        learning_rate=study.best_params["learning_rate"],
        lora_r=study.best_params["lora_r"],
        num_epochs=study.best_params["num_train_epochs"],
    )
    render_hyperparameters_table(best_config)

    # Step 4: Fine-Tuning with Optimal Parameters
    render_step(4, "Fine-Tuning Model with Optimal Hyperparameters", icon="🏋️")
    with status_spinner("Training final model with best LoRA rank and learning rate..."):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        lora_config = LoraConfig(
            r=best_config.lora_r,
            lora_alpha=best_config.lora_r * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/best_model",
            learning_rate=best_config.learning_rate,
            num_train_epochs=best_config.num_epochs,
            per_device_train_batch_size=2,
            eval_strategy="no",
            save_strategy="epoch",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
        )
        trainer.train()

    render_card(
        "Model Checkpoint",
        f"Optimal model weights saved to:\n[text.highlight]{OUTPUT_DIR}/best_model[/text.highlight]",
        icon="💾",
    )

    # Step 5: Code Generation
    render_step(5, "Generating Manim Animation Python Code", icon="⚡")
    with status_spinner(f"Generating code for '{EVAL_PROMPT}'..."):
        generated_code = generate_code_inference(model, tokenizer, EVAL_PROMPT)

    render_code_block(generated_code, language="python", title=f"Generated Manim Code: '{EVAL_PROMPT}'")

    # Step 6: Exporting Artifacts
    render_step(6, "Exporting Dataset Evaluation CSV", icon="💾")
    test_csv_path = f"{OUTPUT_DIR}/test_set.csv"
    pd.DataFrame([{"prompt": ex.prompt, "code": ex.code} for ex in SAMPLE_MANIM_DATA]).to_csv(
        test_csv_path, index=False
    )
    render_card("Export Status", f"Dataset exported to [text.highlight]{test_csv_path}[/text.highlight]", icon="✔")

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Synthetic Data for Domain Specialization",
                "Pairing domain-specific prompts with structured Python output teaches generalist SLMs precise syntax and library constraints (e.g. Manim animations).",
            ),
            (
                "Bayesian Hyperparameter Search",
                "Optuna uses Tree-structured Parzen Estimators (TPE) to find optimal learning rates and LoRA rank values far faster than brute-force grid search.",
            ),
            (
                "Code Generation Temperature",
                "Setting temperature=0.2 reduces token sampling entropy, ensuring deterministic syntax compilation without hallucinated method calls.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
