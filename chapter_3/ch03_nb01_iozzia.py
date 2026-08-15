# ==========================================
# Extracted from CH03_NB01_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # End-to-End Small Language Model Fine Tuning to Generate Manim Code
# This notebook is a companion of chapter 3 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is about fine tuning a small language model, [GPT-2 small](https://huggingface.co/openai-community/gpt2) to generate Python [Manim](https://github.com/ManimCommunity/manim) code. Hardware acceleration (GPU) is needed.  
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the missing dependencies in the Colab VM (only Optuna for hyperparameter search isn't available by default).
# ------------------------------------------------------------

# !pip install optuna

# ------------------------------------------------------------
# ### Data Preparation
# ------------------------------------------------------------

# ------------------------------------------------------------
# Download the [Manim_Python dataset](https://huggingface.co/datasets/Edoh/manim_python) from the Hugging Face's Hub.
# ------------------------------------------------------------

from datasets import load_dataset

dataset = load_dataset("Edoh/manim_python")

# ------------------------------------------------------------
# Display some training sample.
# ------------------------------------------------------------

print(dataset["train"][0])

# ------------------------------------------------------------
# Download the GPT2 Small model's tokenizer.
# ------------------------------------------------------------

from transformers import GPT2Tokenizer

model_name = "openai-community/gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# ------------------------------------------------------------
# Set pad token to eos token for the model's tokenizer.
# ------------------------------------------------------------

tokenizer.pad_token = tokenizer.eos_token

# ------------------------------------------------------------
# Define a custom function to preprocess the raw data. It concatenates the `instruction` and `output` columms of the dataset, tokenizes the concatenated text and set the lables same as the input_ids.
# ------------------------------------------------------------

def preprocess_data(examples):
    inputs = [
        f"Instruction: {instr}\nOutput: {out}"
        for instr, out in zip(examples["instruction"], examples["output"])
    ]
    tokenized = tokenizer(inputs, truncation=True, max_length=512, padding="max_length")

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# ------------------------------------------------------------
# Use the `preprocess_data` function to tokenize the training data. The original columns are also removed, as they aren't needed anymore for training purposes.
# ------------------------------------------------------------

tokenized_datasets = dataset.map(preprocess_data,
                                 batched=True,
                                 remove_columns=dataset["train"].column_names)

# ------------------------------------------------------------
# ### Hyperparameter Tuning
# ------------------------------------------------------------

# ------------------------------------------------------------
# Import the required classes.
# ------------------------------------------------------------

from transformers import (
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# ------------------------------------------------------------
# Define a custom function to init the model during the hyperparameter search. This way we will have a fresh model for each trial. The baseline model weights are pulled from the HF's Hub only during the first trial. Subsequents trials will use the cached weights.
# ------------------------------------------------------------

def model_init():
    return GPT2LMHeadModel.from_pretrained(model_name, device_map='auto')

# ------------------------------------------------------------
# Set the training arguments.
# ------------------------------------------------------------

training_args = TrainingArguments(
    output_dir="./gpt2-manim-python-finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    report_to="none",
)

# ------------------------------------------------------------
# Split the training data, to reserve 10% for validation.
# ------------------------------------------------------------

train_val_split = tokenized_datasets["train"].train_test_split(test_size=0.1)
tokenized_datasets["train"] = train_val_split["train"]
tokenized_datasets["validation"] = train_val_split["test"]

# ------------------------------------------------------------
# Create an instance of data collator for causal language modeling.
# ------------------------------------------------------------

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ------------------------------------------------------------
# Create an instance of Trainer. An early stopping callback is set too.
# ------------------------------------------------------------

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ------------------------------------------------------------
# Define the hyperparameter search space.
# ------------------------------------------------------------

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [2, 4, 8]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 6),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 500),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4]),
    }

# ------------------------------------------------------------
# Run the hyperparameter search. The Optuna backend is used.
# ------------------------------------------------------------

trials_count = 3
best_run = trainer.hyperparameter_search(
    direction="minimize",
    backend="optuna",
    n_trials=trials_count,
    hp_space=hp_space,
    compute_objective=lambda metrics: metrics["eval_loss"],
)

# ------------------------------------------------------------
# Display the best combination of hyperparmaters found.
# ------------------------------------------------------------

print("Best hyperparameters found:", best_run.hyperparameters)

# ------------------------------------------------------------
# ### Training
# ------------------------------------------------------------

# ------------------------------------------------------------
# Configure the Trainer with the best hyperparameters.
# ------------------------------------------------------------

for key, value in best_run.hyperparameters.items():
    setattr(training_args, key, value)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets.get("validation"),
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ------------------------------------------------------------
# Start the model fine-tuning.
# ------------------------------------------------------------

trainer.train()

# ------------------------------------------------------------
# Save the fine-tuned model and the tokenizer to disk.
# ------------------------------------------------------------

trainer.save_model("./gpt2-manim-python-finetuned")
tokenizer.save_pretrained("./gpt2-manim-python-finetuned")

# ------------------------------------------------------------
# ### Test
# ------------------------------------------------------------

# ------------------------------------------------------------
# Load the fine-tuned model from disk to GPU memory. Load also its companion tokenizer.
# ------------------------------------------------------------

import torch

model_dir = "./gpt2-manim-python-finetuned"
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ------------------------------------------------------------
# Define a custom function to do Manim code generation with the fine-tuned model and decode the output.
# ------------------------------------------------------------

def generate_output(instruction, max_length=150, num_beams=5, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
    """
    Generate output text given an instruction using beam search and nucleus sampling.

    Args:
        instruction (str): The input instruction prompt.
        max_length (int): Maximum length of generated sequence (including prompt).
        num_beams (int): Number of beams for beam search.
        temperature (float): Sampling temperature; lower is less random.
        top_p (float): Nucleus sampling probability threshold.
        repetition_penalty (float): Penalty for repeated tokens (>1.0 discourages repetition).

    Returns:
        str: Generated output text.
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
        no_repeat_ngram_size=2,
    )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    output_start = generated_text.find("Output:")
    if output_start != -1:
        output_text = generated_text[output_start + len("Output:"):].strip()
    else:
        output_text = generated_text.strip()

    return output_text

# ------------------------------------------------------------
# Execute the fine-tuned model on all the samples in the test set and save the generated Manim code snippets (along with the corresponding prompt and ground truth) in a CSV file.
# ------------------------------------------------------------

import csv

output_csv = "gpt2_manim_python_test_outputs.csv"
with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["instruction", "reference_output", "generated_output"])
    writer.writeheader()

    for example in dataset['test']:
        instruction = example["instruction"]
        reference_output = example["output"]

        generated_output = generate_output(instruction)

        writer.writerow({
            "instruction": instruction,
            "reference_output": reference_output,
            "generated_output": generated_output,
        })

print(f"Inference complete. Results saved to {output_csv}")

