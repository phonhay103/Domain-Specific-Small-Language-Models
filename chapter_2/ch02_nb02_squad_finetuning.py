"""Fine-tuning a Language Model on SQuAD with Span Mapping.

Companion script for Chapter 2 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Demonstrates preparing the SQuAD dataset for extractive question answering,
token-to-character span alignment using tokenizer offset mappings, and fine-tuning
an encoder model (distilbert-base-uncased).
Refactored using Functional Programming principles and eye-friendly UI components.
"""

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure root workspace is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Third-party
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

# Common functional & UI utilities
from common.ui import (
    STYLE_INDEX,
    STYLE_PRIMARY,
    STYLE_SECONDARY,
    STYLE_TEXT,
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
class QASample:
    """Immutable representation of a Question-Answering instance."""

    question: str
    context: str
    answer_text: str
    start_char: int


@dataclass(frozen=True)
class SpanIndices:
    """Immutable token span boundary representation."""

    start_token: int
    end_token: int


DATASET_ID = "rajpurkar/squad"
MODEL_ID = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 384
STRIDE = 128
TRAIN_SAMPLES = 100
EVAL_SAMPLES = 50
OUTPUT_DIR = "my_squad_model"
SAMPLE_QUESTION = "How many members does the band have?"
SAMPLE_CONTEXT = "The band consists of 4 members: John, Paul, George, and Ringo."


# ---------------------------------------------------------------------------
# Pure Span Alignment Logic
# ---------------------------------------------------------------------------
def compute_token_span(
    offset: Sequence[tuple[int, int]],
    sequence_ids: Sequence[int | None],
    start_char: int,
    end_char: int,
) -> SpanIndices:
    """Pure function: calculate token start/end indices for a character range."""
    # Find context boundaries in sequence_ids
    context_indices = [i for i, sid in enumerate(sequence_ids) if sid == 1]
    if not context_indices:
        return SpanIndices(0, 0)

    context_start, context_end = context_indices[0], context_indices[-1]

    # Check if answer is contained within this context chunk
    if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
        return SpanIndices(0, 0)

    # Locate start token
    start_idx = context_start
    while start_idx <= context_end and offset[start_idx][0] <= start_char:
        start_idx += 1
    token_start = start_idx - 1

    # Locate end token
    end_idx = context_end
    while end_idx >= context_start and offset[end_idx][1] >= end_char:
        end_idx -= 1
    token_end = end_idx + 1

    return SpanIndices(token_start, token_end)


def preprocess_training_examples(examples: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Pure batch mapping function: tokenizes text and aligns ground-truth spans."""
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]

    spans = [
        compute_token_span(
            offset=offset,
            sequence_ids=inputs.sequence_ids(i),
            start_char=answers[sample_map[i]]["answer_start"][0],
            end_char=answers[sample_map[i]]["answer_start"][0] + len(answers[sample_map[i]]["text"][0]),
        )
        for i, offset in enumerate(offset_mapping)
    ]

    inputs["start_positions"] = [span.start_token for span in spans]
    inputs["end_positions"] = [span.end_token for span in spans]
    return inputs


def create_training_arguments(output_dir: str) -> TrainingArguments:
    """Pure factory for training configuration."""
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        fp16=torch.cuda.is_available(),
        weight_decay=0.01,
        push_to_hub=False,
        disable_tqdm=True,
        report_to="none",
        save_strategy="no",
    )


def predict_extractive_answer(
    model: Any,
    tokenizer: Any,
    question: str,
    context: str,
) -> dict[str, Any]:
    """Pure inference function: predicts answer span from context via model logits."""
    inputs = tokenizer(question, context, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping")[0]
    sequence_ids = inputs.sequence_ids(0)

    device = next(model.parameters()).device
    inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs_on_device)

    start_logits = outputs.start_logits[0].cpu()
    end_logits = outputs.end_logits[0].cpu()

    # Context token indices (where sequence_id == 1)
    context_indices = [i for i, sid in enumerate(sequence_ids) if sid == 1]
    if not context_indices:
        return {"answer": "", "score": 0.0, "start": 0, "end": 0}

    best_score = -float("inf")
    best_start, best_end = context_indices[0], context_indices[0]

    start_probs = torch.softmax(start_logits, dim=-1)
    end_probs = torch.softmax(end_logits, dim=-1)

    for s in context_indices:
        for e in context_indices:
            if s <= e and (e - s < 30):
                score = (start_logits[s] + end_logits[e]).item()
                if score > best_score:
                    best_score = score
                    best_start, best_end = s, e

    prob_score = (start_probs[best_start] * end_probs[best_end]).item()
    start_char = int(offset_mapping[best_start][0])
    end_char = int(offset_mapping[best_end][1])
    answer = context[start_char:end_char]

    return {
        "answer": answer,
        "score": float(prob_score),
        "start": start_char,
        "end": end_char,
    }


# ---------------------------------------------------------------------------
# View / Rendering Functions
# ---------------------------------------------------------------------------
def render_sample_table(sample: QASample) -> None:
    """Render sample training record in an eye-friendly table."""
    columns = [
        ("Field", STYLE_PRIMARY, "left"),
        ("Content", STYLE_TEXT, "left"),
    ]
    rows = [
        ("Question", sample.question),
        ("Context Preview", sample.context[:160] + "..."),
        ("Ground Truth Answer", sample.answer_text),
        ("Start Character Position", str(sample.start_char)),
    ]
    console.print(create_table("Sample SQuAD Training Record", columns, rows))
    pause()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute SQuAD extractive question-answering training workflow."""
    silence_hf_logs()

    render_banner(
        title="Fine-Tuning DistilBERT on SQuAD with Span Mapping",
        subtitle="Chapter 2: Domain-Specific Small Language Models",
        metadata={
            "Base Model": MODEL_ID,
            "Train Subset": f"{TRAIN_SAMPLES} samples",
            "Eval Subset": f"{EVAL_SAMPLES} samples",
        },
        icon="🚀",
    )

    # Step 1: Loading Dataset & Tokenizer
    render_step(1, "Loading SQuAD Dataset & Subword Tokenizer", icon="📋")
    with status_spinner("Loading SQuAD dataset partition from Hugging Face..."):
        squad_train = load_dataset(DATASET_ID, split=f"train[:{TRAIN_SAMPLES}]")
        squad_eval = load_dataset(DATASET_ID, split=f"validation[:{EVAL_SAMPLES}]")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    first_item = squad_train[0]
    sample_record = QASample(
        question=first_item["question"],
        context=first_item["context"],
        answer_text=first_item["answers"]["text"][0],
        start_char=first_item["answers"]["answer_start"][0],
    )
    render_sample_table(sample_record)

    # Step 2: Preprocessing & Span Alignment
    render_step(2, "Preprocessing & Mapping Subword Token Spans", icon="⚙️")
    with status_spinner("Mapping character offsets to subword token boundaries..."):
        train_dataset = squad_train.map(
            lambda ex: preprocess_training_examples(ex, tokenizer),
            batched=True,
            remove_columns=squad_train.column_names,
        )
        eval_dataset = squad_eval.map(
            lambda ex: preprocess_training_examples(ex, tokenizer),
            batched=True,
            remove_columns=squad_eval.column_names,
        )

    render_card(
        title="Span Alignment Statistics",
        content=(
            f"[text.muted]Generated Training Chunks:[/text.muted] [text.highlight]{len(train_dataset)}[/text.highlight]\n"
            f"[text.muted]Generated Validation Chunks:[/text.muted] [text.highlight]{len(eval_dataset)}[/text.highlight]\n"
            f"[text.muted]Window Stride:[/text.muted] [brand.secondary]{STRIDE} tokens[/brand.secondary]"
        ),
        icon="✔",
    )

    # Step 3: Model & Trainer Setup
    render_step(3, "Initializing Model & Training Configuration", icon="🧠")
    with status_spinner(f"Loading QA head for '{MODEL_ID}'..."):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID).to(device)
    render_device_info(next(model.parameters()).device, model=model)
    training_args = create_training_arguments(OUTPUT_DIR)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DefaultDataCollator(),
    )
    silence_trainer(trainer)

    # Step 4: Training Execution
    render_step(4, "Executing DistilBERT Fine-Tuning", icon="🏋️")
    with status_spinner("Running 10-epoch fine-tuning & evaluation loop..."):
        train_output = trainer.train()

    render_training_metrics_table(trainer.state.log_history, title="DistilBERT Fine-Tuning Progression")
    render_card(
        "Training Status",
        (
            f"[status.success]DistilBERT fine-tuning completed successfully.[/status.success]\n"
            f"[text.muted]Total Runtime:[/text.muted] [brand.secondary]{train_output.metrics.get('train_runtime', 0):.2f}s[/brand.secondary]  •  "
            f"[text.muted]Throughput:[/text.muted] [text.highlight]{train_output.metrics.get('train_samples_per_second', 0):.1f} samples/s[/text.highlight]  •  "
            f"[text.muted]Final Train Loss:[/text.muted] [status.warning]{train_output.metrics.get('train_loss', 0):.4f}[/status.warning]"
        ),
        icon="✔",
    )

    # Step 5: Extractive QA Pipeline Inference
    render_step(5, "Evaluating Extractive QA Pipeline", icon="🎯")
    with status_spinner("Running extractive span prediction..."):
        prediction = predict_extractive_answer(model, tokenizer, SAMPLE_QUESTION, SAMPLE_CONTEXT)

    render_card(
        title="Extractive QA Inference Result",
        content=(
            f"[text.muted]Context:[/text.muted] [text.main]{SAMPLE_CONTEXT}[/text.main]\n"
            f"[text.muted]Question:[/text.muted] [text.main]{SAMPLE_QUESTION}[/text.main]\n\n"
            f"[status.success]Extracted Answer:[/status.success] [text.highlight]{prediction['answer']}[/text.highlight]\n"
            f"[text.muted]Confidence Score:[/text.muted] [status.warning]{prediction['score']:.4f}[/status.warning]  •  "
            f"[text.muted]Span:[/text.muted] [brand.secondary][{prediction['start']}:{prediction['end']}][/brand.secondary]"
        ),
        icon="✨",
    )

    # Educational Takeaways
    render_takeaways(
        points=(
            (
                "Extractive vs Generative QA",
                "In extractive QA, the model does not generate new text; it classifies which token in the input context is the start of the answer and which is the end.",
            ),
            (
                "Sliding Window (Stride)",
                "When context exceeds MAX_LENGTH=384, the text is split into overlapping chunks with STRIDE=128 to prevent answer boundary cuts.",
            ),
            (
                "Token Offset Mapping",
                "Because tokenizers break words into subwords (e.g., 'playing' -> 'play', '##ing'), offset mappings are critical to align character-level annotations with token indices.",
            ),
        ),
    )


if __name__ == "__main__":
    main()
