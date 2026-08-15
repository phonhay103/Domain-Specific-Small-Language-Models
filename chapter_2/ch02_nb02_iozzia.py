"""Fine-tune DistilBERT for extractive question answering on a SQuAD subset.

Companion script for Chapter 2 of "Domain Specific LLMs in Action"
by Guglielmo Iozzia, Manning Publications, 2024.

Demonstrates end-to-end QA fine-tuning on DistilBERT-base-uncased:
data prep → tokenisation → fine-tuning → inference.
Hardware acceleration (GPU) is recommended for the fine-tuning step.

# Install missing requirements first (Colab / fresh env):
# !pip install datasets accelerate
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
from __future__ import annotations

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_NAME = "rajpurkar/squad"
DATASET_SPLIT = "train[:5000]"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "my_awesome_qa_model"
MAX_SEQ_LENGTH = 384
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
TEST_SIZE = 0.2

# Sample question/context used for the inference demo
DEMO_QUESTION = "How many official league titles has Juventus won?"
DEMO_CONTEXT = (
    "Juventus Football Club (from Latin: iuventūs), colloquially known as Juve, "
    "is a professional football club based in Turin, Piedmont, Italy, that competes "
    "in the Serie A, the top tier of the Italian football league system. Founded in "
    "1897 by a group of Torinese students, the club has worn a black and white striped "
    "home kit since 1903 and has played home matches in different grounds around its "
    "city, the latest being the 41,507-capacity Juventus Stadium. Nicknamed la Vecchia "
    "Signora (the Old Lady), the club has won 36 official league titles, 14 Coppa Italia "
    "titles and nine Supercoppa Italiana titles, being the record holder for all these "
    "competitions;"
)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_and_split_dataset(tokenizer: AutoTokenizer):
    """Load a SQuAD subset and split it 80/20 into train and test."""
    squad = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    squad = squad.train_test_split(test_size=TEST_SIZE)
    print(squad["train"][0])
    return squad


def preprocess_function(examples: dict, tokenizer: AutoTokenizer) -> dict:
    """Tokenise QA examples and map answer character spans to token positions.

    Handles long contexts via truncation of the context only, and uses offset
    mapping to locate start/end token positions of each answer.
    """
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_SEQ_LENGTH,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions: list[int] = []
    end_positions: list[int] = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def tokenise_dataset(squad, tokenizer: AutoTokenizer):
    """Apply preprocessing across the full dataset in batches."""
    return squad.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=squad["train"].column_names,
    )


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def fine_tune(tokenized_squad, tokenizer: AutoTokenizer) -> Trainer:
    """Load DistilBERT and fine-tune it for QA; return the trained Trainer."""
    model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL_NAME, device_map="auto"
    )
    data_collator = DefaultDataCollator()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        push_to_hub=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_squad["train"],
        eval_dataset=tokenized_squad["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    return trainer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, tokenizer: AutoTokenizer) -> None:
    """Run the fine-tuned model on a demo question and print the decoded answer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(DEMO_QUESTION, DEMO_CONTEXT, return_tensors="pt")
    inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    # Guard against inverted span predictions
    if answer_end_index < answer_start_index:
        answer_end_index = answer_start_index

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    decoded_answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

    print(f"Decoded answer: {decoded_answer}")

    # Also decode the full input tokens to better understand the context and tokenization
    print(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    """End-to-end pipeline: data prep → fine-tuning → inference."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    squad = load_and_split_dataset(tokenizer)
    tokenized_squad = tokenise_dataset(squad, tokenizer)

    trainer = fine_tune(tokenized_squad, tokenizer)

    run_inference(trainer.model, tokenizer)


if __name__ == "__main__":
    main()
