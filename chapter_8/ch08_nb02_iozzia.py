# ==========================================
# Extracted from CH08_NB02_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Antibody Generation with AntibodyGPT
# This notebook is a companion of chapter 8 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is to generate antibody sequences using the [AntibodyGPT](https://huggingface.co/AntibodyGeneration/fine-tuned-progen2-small) model. It requires hardware acceleration.  
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Downgrading the HF's Transformers library to ensure compatibility with the AntibodyGPT2's `ProGenForCausalLM` class, as it inherits from `PreTrainedModel`, which, starting from Transformers release 4.50, wouldn't inherit from `GenerationMixin` anymore, in so loosing the availability of the `generate` method.
# ------------------------------------------------------------

# !pip install transformers==4.49.0

# ------------------------------------------------------------
# Clone the official repo.
# ------------------------------------------------------------

# !git clone https://github.com/joethequant/docker_protein_generator.git
# %cd ./docker_protein_generator/

# ------------------------------------------------------------
# Download one of the pretrained models and the associated tokenizer from the HF's Hub. Please note that the AutoClass to use is the custom ```ProGenForCausalLM``` available in the ```docker_protein_generator``` cloned repo.
# 
# ------------------------------------------------------------

from models.progen.modeling_progen import ProGenForCausalLM
import torch
from tokenizers import Tokenizer

model_path = 'AntibodyGeneration/fine-tuned-progen2-small'

model = ProGenForCausalLM.from_pretrained(model_path)
tokenizer = Tokenizer.from_pretrained(model_path)

from pathlib import Path

models_path = Path("antibodygen")
model.save_pretrained(models_path)

# ------------------------------------------------------------
# The save model to disk is 588.6 MB (617 MB in memory after download).
# ------------------------------------------------------------

# ------------------------------------------------------------
# Define a target antigen sequence and the number of antibody sequences you want to generate for it and then start the generation process.
# ------------------------------------------------------------

target_sequence = 'MQIPQAPWPVVWAVLQLGWRPGWFLDSPDRPWNPPTFSPALLVVTEGDNATFTCSFSNTSESFVLNWYRMSPSNQTDKLAAFPEDRSQPGQDCRFRVTQLPNGRDFHMSVVRARRNDSGTYLCGAISLAPKAQIKESLRAELRVTERRAEVPTAHPSPSPRPAGQFQTLVVGVVGGLLGSLVLLVWVLAVICSRAARGTIGARRTGQPLKEDPSAVPVFSVDYGELDFQWREKTPEPPVPCVPEQTEYATIVFPSGMGTSSPARRGSADGPRSAQPLRPEDGHCSWPL'
number_of_sequences = 2

# ------------------------------------------------------------
# Tokenize the prompt sequence and then convert it to PyTorch tensor and move it to the GPU.
# ------------------------------------------------------------

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenized_sequence = tokenizer.encode(target_sequence)
input_tensor = torch.tensor([tokenized_sequence.ids]).to(device)

# ------------------------------------------------------------
# Move the model to GPU.
# ------------------------------------------------------------

model = model.to(device)

# ------------------------------------------------------------
# Start the sequence generation.
# ------------------------------------------------------------

with torch.no_grad():
    output = model.generate(input_tensor, max_length=1024,
                            pad_token_id=tokenizer.encode('<|pad|>').ids[0],
	                          do_sample=True, top_p=0.9, temperature=0.8,
	                          num_return_sequences=number_of_sequences)

# ------------------------------------------------------------
# Decode the generated sequences and display them.
# ------------------------------------------------------------

as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]
sequences = tokenizer.decode_batch(as_lists(output))
if len(sequences) > 0:
    sequences = [x.replace('2', '') for x in sequences]

sequences

