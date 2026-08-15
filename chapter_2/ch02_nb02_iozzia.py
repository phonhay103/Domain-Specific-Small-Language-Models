# ==========================================
# Extracted from CH02_NB02_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Fine Tuning BERT for Question Answering
# This notebook is a companion of chapter 2 of the "Domain Specific LLms in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.
# The code in this notebook is to show how to fine tune a [DistilBert](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) model for extractive question answering (extract the answer from a given context). While all the steps refer to DistilBert, the same apply to a larger pool of LLM architectures.   
# While no hardware acceleration is required to execute all the code cells, it is recommended to use a GPU to speed up the fine tuning step.
# ------------------------------------------------------------

# ------------------------------------------------------------
# ## Settings
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the missing requirements in the Colab VM (HF's Datasets and Accelerate).
# ------------------------------------------------------------

# !pip install datasets accelerate

# ------------------------------------------------------------
# ## Data Preparation
# ------------------------------------------------------------

# ------------------------------------------------------------
# Load a subset (5000 samples) of the *SQuAD* dataset using the HF's *Datasets* library.
# ------------------------------------------------------------

from datasets import load_dataset

squad = load_dataset("rajpurkar/squad", split="train[:5000]")

# ------------------------------------------------------------
# Split the dataset's `train` split into a train and test set (80%/20%) with the *train_test_split* method:
# ------------------------------------------------------------

squad = squad.train_test_split(test_size=0.2)

# ------------------------------------------------------------
# Display a sample:
# ------------------------------------------------------------

squad["train"][0]

# ------------------------------------------------------------
# The three important fields are:
# 
# - `answers`: the starting location of the answer token and the answer text.
# - `context`: background information from which the model needs to extract the answer.
# - `question`: the question a model should answer.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Load a DistilBERT tokenizer to process the `question` and `context` fields in the training set.
# ------------------------------------------------------------

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# ------------------------------------------------------------
# Some data preprocessing is needed:
# 
# 1. Some examples in a dataset may have a very long `context` that exceeds the maximum input length of the model. To deal with longer sequences, truncate only the `context` by setting `truncation="only_second"`.
# 2. Map the start and end positions of the answer to the original `context` by setting
#    `return_offset_mapping=True`.
# 3. Use the *sequence_ids* method to
#    find which part of the offset corresponds to the `question` and which corresponds to the `context`.
# 
# Let's define a function that implementes the aforementioned preprocessing steps:
# ------------------------------------------------------------

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

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

# ------------------------------------------------------------
# To apply the preprocessing function over the entire dataset, we can use the HF's Datasets *map* function. By setting `batched=True`, multiple elements of the dataset will be processed at once.
# ------------------------------------------------------------

tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

# ------------------------------------------------------------
# Now create a batch of examples using DefaultDataCollator. Unlike other data collators in the HF's Transformers library, it doesn't apply any additional preprocessing.
# ------------------------------------------------------------

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()

# ------------------------------------------------------------
# ## Fine Tuning
# ------------------------------------------------------------

# ------------------------------------------------------------
# Load DistilBERT with *AutoModelForQuestionAnswering*:
# ------------------------------------------------------------

from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased",
                                                    device_map="auto")

# ------------------------------------------------------------
# Define the training hyperparameters in a *TrainingArguments* instance. The only required parameter is *output_dir* which specifies where to save our model.
# ------------------------------------------------------------

training_args = TrainingArguments(
    output_dir="my_awesome_qa_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    push_to_hub=False,
    report_to="none",
)

# ------------------------------------------------------------
# Pass the training arguments to a *Trainer* instance along with the model, the dataset, the tokenizer, and the data collator.
# ------------------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

# ------------------------------------------------------------
# Start the fine tuning:
# ------------------------------------------------------------

trainer.train()

# ------------------------------------------------------------
# ## Inference
# ------------------------------------------------------------

# ------------------------------------------------------------
# The finetuned model can now be used for inference. Let's provide a question and some context we'd like the model to predict:
# ------------------------------------------------------------

question = "How many official league titles has Juventus won?"
context = "Juventus Football Club (from Latin: iuventūs), colloquially known as Juve, is a professional football club based in Turin, Piedmont, Italy, that competes in the Serie A, the top tier of the Italian football league system. Founded in 1897 by a group of Torinese students, the club has worn a black and white striped home kit since 1903 and has played home matches in different grounds around its city, the latest being the 41,507-capacity Juventus Stadium. Nicknamed la Vecchia Signora (the Old Lady), the club has won 36 official league titles, 14 Coppa Italia titles and nine Supercoppa Italiana titles, being the record holder for all these competitions;"

# ------------------------------------------------------------
# First, tokenize the text and return PyTorch tensors:
# ------------------------------------------------------------

import torch
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

inputs = tokenizer(question, context, return_tensors="pt")
inputs.to(device)

# ------------------------------------------------------------
# Then let's pass our inputs to the model and return the `logits`:
# ------------------------------------------------------------

from transformers import AutoModelForQuestionAnswering

with torch.no_grad():
    outputs = model(**inputs)

# ------------------------------------------------------------
# Get the highest probability from the model output for the start and end positions:
# ------------------------------------------------------------

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

# ------------------------------------------------------------
# Decode the predicted tokens to get the answer in natural language:
# ------------------------------------------------------------

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
if answer_end_index < answer_start_index:
    answer_end_index = answer_start_index

predict_answer_tokens_corrected = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
decoded_answer = tokenizer.decode(predict_answer_tokens_corrected, skip_special_tokens=True)

print(f"Decoded answer: {decoded_answer}")

# Also, let's decode the full input tokens to better understand the context and tokenization
print(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False))

