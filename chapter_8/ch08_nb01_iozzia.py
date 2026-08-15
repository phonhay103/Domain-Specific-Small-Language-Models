# ==========================================
# Extracted from CH08_NB01_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Generating Protein Sequences with ProtGPT2 Locally
# This notebook is a companion of chapter 8 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is to generate protein sequences using the [ProtGPT2](https://huggingface.co/nferruz/ProtGPT2) model. It doesn't require hardware acceleration.  
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Download the ProtGPT2 model from the HF Hub and set up an inference pipeline for it.
# ------------------------------------------------------------

from transformers import pipeline

model_id = "nferruz/ProtGPT2"
protgpt2 = pipeline('text-generation', model=model_id)

# ------------------------------------------------------------
# Use the pipeline to start generating protein sequences (10 in this example). At the end of the generation process the protein sequences are displayed on the standard output.
# ------------------------------------------------------------

sequences = protgpt2("<|endoftext|>", max_length=100, do_sample=True, top_k=950,
                     repetition_penalty=1.2, num_return_sequences=10,
                     eos_token_id=0)
for seq in sequences:
  print(seq)

# ------------------------------------------------------------
# Define a function to calculate the perplexity metric for the generated results.
# ------------------------------------------------------------

import torch

def calculate_perplexity(model, tokenizer, text, device):
    encodings = tokenizer(text, return_tensors='pt').to(device)

    input_ids = encodings.input_ids
    target_ids = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

    neg_log_likelihood = outputs.loss

    perplexity = torch.exp(neg_log_likelihood)

    return perplexity

# ------------------------------------------------------------
# Evaluate the generated results by calculating the perplexity metric for them.
# ------------------------------------------------------------

device = 'cpu'
for seq in sequences:
  print(calculate_perplexity(protgpt2.model, protgpt2.tokenizer,
                       seq['generated_text'], device))

# ------------------------------------------------------------
# Alternatively we can calculate perplexity on a batch of generated protein sequences. Let's define a custom function for this.
# ------------------------------------------------------------

protgpt2.tokenizer.pad_token = protgpt2.tokenizer.eos_token

def calculate_batch_perplexity(input_texts, model, tokenizer):
    """
    Calculate perplexity for a batch of input texts using a pretrained language model.

    Args:
    - input_texts (List[str]): A list of input texts to evaluate.

    Returns:
    - List[float]: A list of perplexity scores, one for each input text.
    """
    # Tokenize the batch of texts with padding for uniform length
    inputs = tokenizer(
        input_texts, return_tensors="pt", padding=True, truncation=True
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Pass the input batch through the model to get logits
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Shift the logits and input_ids to align targets correctly
    # Logits dimensions are: (batch_size, seq_length, vocab_size)
    shift_logits = logits[:, :-1, :]  # Ignore the last token's logits
    shift_labels = input_ids[:, 1:]   # Skip the first token in the labels

    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Gather the log probabilities for the correct tokens
    target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask out positions corresponding to padding tokens
    target_log_probs = target_log_probs * attention_mask[:, 1:].to(log_probs.dtype)

    # Compute the mean negative log-likelihood for each sequence
    negative_log_likelihood = -target_log_probs.sum(dim=-1) / attention_mask[:, 1:].sum(dim=-1)

    # Compute perplexity for each sequence
    perplexities = torch.exp(negative_log_likelihood)

    # Take mean of perplexities of each batch
    mean_perplexity_score = torch.mean(perplexities)

    return {"perplexities": perplexities, "mean_perplexity": mean_perplexity_score}

# ------------------------------------------------------------
# Execute the ```calculate_batch_perplexity``` function on the generated protein sequences.
# 
# ------------------------------------------------------------

sequence_texts = [seq['generated_text'] for seq in sequences]
print(f"Perplexity scores: {calculate_batch_perplexity(sequence_texts, protgpt2.model, protgpt2.tokenizer)}")

