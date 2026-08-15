# ==========================================
# Extracted from CH02_NB03_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # LoRA on FLAN-T5 Small with the HF's Transformers Library
# This notebook is a companion of chapter 2 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2025.  
# The code in this notebook is to introduce readers to a PEFT (Parameter-Efficient Fine-Tuning) technique called [LoRA](https://arxiv.org/abs/2106.09685) (Low Ranking Adaptation). The pre-trained LLM model used as baseline is [FLAN-T5 small](https://huggingface.co/google/flan-t5-small) loaded through the Hugging Face's [Transformers library](https://github.com/huggingface/transformers). It is going to be tuned for text summarization on a subset of the [samsum](https://huggingface.co/datasets/samsum) dataset. Code execution requires a Colab free VM with hardware acceleration (GPU).  
# More details about this code example can be found in the book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the missing requirements in the Colab VM.
# ------------------------------------------------------------

# !pip install datasets peft accelerate bitsandbytes evaluate rouge_score py7zr

import locale

original_getpreferredencoding = locale.getpreferredencoding

def getpreferredencoding_wrapper(do_raise=True):
  return original_getpreferredencoding()

locale.getpreferredencoding = getpreferredencoding_wrapper

# ------------------------------------------------------------
# ### Data Preparation
# ------------------------------------------------------------

# ------------------------------------------------------------
# Load the **sansum** dataset from the HF Hub.
# ------------------------------------------------------------

from datasets import load_dataset

dataset = load_dataset("knkarthick/samsum", trust_remote_code=True)

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

# ------------------------------------------------------------
# Load the FLAN-T5 small model tokenizer from the HF Hub.
# ------------------------------------------------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id="google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# ------------------------------------------------------------
# Some preprocessing of the training/test data is needed.  
# We need to truncate training and test sequences that are longer than the maximum input sequence after tokenization and pad those that are shorter. This applies to both input and target.  
# For the input, we take the 85 percentile of the max length for better utilization.
# ------------------------------------------------------------

from datasets import concatenate_datasets
import numpy as np

combined_dataset = concatenate_datasets([dataset["train"], dataset["test"]])

tokenized_inputs = []
for example in combined_dataset:
    if example["dialogue"] is not None:
        tokenized_input = tokenizer(text=example["dialogue"], truncation=True)
        tokenized_inputs.append(tokenized_input["input_ids"])

input_lenghts = [len(x) for x in tokenized_inputs]
max_source_length = int(np.percentile(input_lenghts, 85))
print(f"Max source length: {max_source_length}")

# ------------------------------------------------------------
# For the target, we take the 90 percentile of the max length for better utilization.
# ------------------------------------------------------------

tokenized_targets = concatenate_datasets(
    [dataset["train"], dataset["test"]]).map(
        lambda x: tokenizer(x["summary"], truncation=True), batched=True,
        remove_columns=["dialogue", "summary"])
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
max_target_length = int(np.percentile(target_lenghts, 90))
print(f"Max target length: {max_target_length}")

# ------------------------------------------------------------
# We can now define a single function that executes all the preprocessing steps (input tokenization, truncation and padding).
# ------------------------------------------------------------

def preprocess_function(sample, padding="max_length"):
    # Filter out examples where dialogue is None and keep corresponding summaries
    processed_samples = [(dialogue, summary) for dialogue, summary in zip(sample["dialogue"], sample["summary"]) if dialogue is not None]

    inputs = ["summarize: " + dialogue for dialogue, summary in processed_samples]
    labels_text = [summary for dialogue, summary in processed_samples]

    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    labels = tokenizer(text_target=labels_text, max_length=max_target_length, padding=padding, truncation=True)

    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ------------------------------------------------------------
# Apply the function defined at the previous cell to the tokenized dataset.
# ------------------------------------------------------------

tokenized_dataset = dataset.map(preprocess_function, batched=True,
                                remove_columns=["dialogue", "summary", "id"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# ------------------------------------------------------------
# Save the preprocessed datasets to disk to reuse them later.
# ------------------------------------------------------------

tokenized_dataset["train"].save_to_disk("data/train")
tokenized_dataset["test"].save_to_disk("data/eval")

# ------------------------------------------------------------
# ### Fine tuning with LoRA and [bitsandbytes](https://github.com/TimDettmers/bitsandbytes#) int8.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Load the FLAN-T5 small model in 8-bit precision from the HF's Hub.
# ------------------------------------------------------------

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True,
                                              device_map="auto")

# ------------------------------------------------------------
# Define the LoRA configuration, prepare the model for training, and add the LoRA adaptor to it.
# ------------------------------------------------------------

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.SEQ_2_SEQ_LM
)

model = prepare_model_for_kbit_training(model)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ------------------------------------------------------------
# At the end of the execution of the code cell above, the number of parameters to train should be < 1% of the total for the model.  
# The training process is the same as regular LLM training, the main difference is in model to be trained, which is the one after submission to LoRA.  
# Define a Data Collator for this training:
# ------------------------------------------------------------

from transformers import DataCollatorForSeq2Seq

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

# ------------------------------------------------------------
# Set the training arguments and use them to create a Trainer instance. For this use case training for 3 epochs should be enough.  
# Model warnings have been silenced to make the output at training time less verbose and more readable.
# ------------------------------------------------------------

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

output_dir="lora-flan-t5-small"

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
	auto_find_batch_size=True,
    learning_rate=1e-3,
    num_train_epochs=3,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
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

# ------------------------------------------------------------
# Start the training.
# ------------------------------------------------------------

trainer.train()

# ------------------------------------------------------------
# Save the fine-tuned model to disk.
# ------------------------------------------------------------

lora_model_id="flan_t5_lora"
trainer.model.save_pretrained(lora_model_id)
tokenizer.save_pretrained(lora_model_id)

# ------------------------------------------------------------
# ### Inference and Evaluation
# ------------------------------------------------------------

# ------------------------------------------------------------
# Prepare the model for inference. Load the LoRA configuration and checkpoints, reload the base model and merge the weights.
# ------------------------------------------------------------

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

config = PeftConfig.from_pretrained(lora_model_id)

model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path,  load_in_8bit=True,  device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model, lora_model_id, device_map={"":0})
model.eval()

# ------------------------------------------------------------
# Perform inference (text summarization) on a random subset of the test samples.
# ------------------------------------------------------------

from random import randrange
from datasets import load_dataset

sample = dataset['test'][randrange(len(dataset["test"]))]

input_ids = tokenizer(sample["dialogue"], return_tensors="pt",
                      truncation=True).input_ids.cuda()
outputs = model.generate(input_ids=input_ids, max_new_tokens=10,
                         do_sample=True, top_p=0.9)
print(f"input sentence: {sample['dialogue']}\n{'---'* 20}")

print(f"summary:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")

# ------------------------------------------------------------
# Define a function to evaluate the model.
# ------------------------------------------------------------

import numpy as np

def evaluate_peft_model(sample, max_target_length=50):
    outputs = model.generate(input_ids=sample["input_ids"].unsqueeze(0).cuda(),
                             do_sample=True, top_p=0.9,
                             max_new_tokens=max_target_length)
    prediction = tokenizer.decode(
        outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
    labels = np.where(sample['labels'] != -100,
                      sample['labels'], tokenizer.pad_token_id)
    labels = tokenizer.decode(labels, skip_special_tokens=True)

    return prediction, labels

# ------------------------------------------------------------
# Evaluate the model ([*ROUGE* score](https://huggingface.co/spaces/evaluate-metric/rouge)) on the test dataset.
# ------------------------------------------------------------

import evaluate
from datasets import load_from_disk
from tqdm import tqdm

metric = evaluate.load("rouge")

test_dataset = load_from_disk("data/eval/").with_format("torch")

predictions, references = [] , []
for sample in tqdm(test_dataset):
    p,l = evaluate_peft_model(sample)
    predictions.append(p)
    references.append(l)

rogue = metric.compute(predictions=predictions,
                       references=references,
                       use_stemmer=True)

print(f"Rogue1: {rogue['rouge1']* 100:2f}%")
print(f"rouge2: {rogue['rouge2']* 100:2f}%")
print(f"rougeL: {rogue['rougeL']* 100:2f}%")
print(f"rougeLsum: {rogue['rougeLsum']* 100:2f}%")

