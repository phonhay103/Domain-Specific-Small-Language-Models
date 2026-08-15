# ==========================================
# Extracted from CH08_NB03_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Generating Crystal Structures with CrystaLLM
# This notebook is a companion of chapter 8 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is to generate and evaluate crystal structures using the [CrystaLLM](https://github.com/lantunes/CrystaLLM) model. It doesn't require hardware acceleration.  
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Clone the CrystaLLM repo.
# ------------------------------------------------------------

# !git clone https://github.com/lantunes/CrystaLLM.git

# ------------------------------------------------------------
# Install the missing requirements. Only Pymatgen (Python Materials Genomics, a robust Open Source library for materials analysis) and OmegaConf (a YAML based hierarchical configuration system) are missing in the Colab VMs.
# ------------------------------------------------------------

# !pip install pymatgen==2023.3.23 omegaconf

# ------------------------------------------------------------
# Add the CrystaLLM path to the Python path.
# ------------------------------------------------------------

import sys
import os

sys.path.append('/content/CrystaLLM')
os.environ["PYTHONPATH"] += (":/content/CrystaLLM")

# %cd CrystaLLM

# ------------------------------------------------------------
# Download a CrystLLM pretrained model. They aren't available in the HF's Hub, so we have to use the provided ```download.py``` script.
# ------------------------------------------------------------

# !python bin/download.py crystallm_v1_small.tar.gz
# !tar xvf crystallm_v1_small.tar.gz

# ------------------------------------------------------------
# Create a prompt to be used for the generation process and save it to file.
# ------------------------------------------------------------

# !python bin/make_prompt_file.py Na2Cl2 sample_prompt.txt --spacegroup P4/nmm

# ------------------------------------------------------------
# Do random sampling.
# ------------------------------------------------------------

# !python bin/sample.py \
# out_dir=crystallm_v1_small \
# start=FILE:sample_prompt.txt \
# num_samples=2 \
# top_k=10 \
# max_new_tokens=3000 \
# device=cpu \
# target=file

# ------------------------------------------------------------
# Post-process the generated raw CIF files.
# ------------------------------------------------------------

# !python bin/postprocess.py . colab_processed_cifs

# ------------------------------------------------------------
# Alternatively, we can do Monte Carlo Tree Search decoding to generate CIF files from the given prompt.
# ------------------------------------------------------------

# !python bin/mcts.py \
# out_dir=crystallm_v1_small \
# device=cpu \
# dtype=bfloat16 \
# start=FILE:sample_prompt.txt \
# tree_width=5 \
# max_depth=2000 \
# selector=puct \
# c=1.0 \
# num_simulations=1000 \
# reward_k=2.0 \
# scorer=random \
# top_child_weight_cutoff=0.9999 \
# bypass_only_child=True \
# mcts_out_dir=colab_mcts_cifs

# ------------------------------------------------------------
# ### CIF Files Evaluation
# ------------------------------------------------------------

# ------------------------------------------------------------
# Save the generated CIF files into a tar.gz file, as this is a requirement for the provided Python script for evaluation.
# ------------------------------------------------------------

# !tar -czvf colab_processed_cifs.tar.gz ./colab_processed_cifs/

# ------------------------------------------------------------
# Perform the evaluation and save the results to a CSV file.
# ------------------------------------------------------------

# !python bin/evaluate_cifs.py colab_processed_cifs.tar.gz -o colab_processed_cifs.csv

# ------------------------------------------------------------
# # Optional: Prepare the Trained Model for Push to the Hugging Face Hub
# ------------------------------------------------------------

# ------------------------------------------------------------
# This section is about showcasing the steps to make a CrystaLLM pretrained model available through the HF's Transformers API and share it in the HF's Hub. The code below is just for educational purposes, please refrain to share the original CrystaLLM pretrained model without consent from the authors. You can still tune further the original models on your own CIF files dataset, but please stay compliant to any update about the OS license for the CrystaLLM work before taking any further action. Thanks.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Upgrade Numpy to the latest version.
# ------------------------------------------------------------

# !pip install -U numpy

# ------------------------------------------------------------
# Create the configuration file for the model.
# 
# ------------------------------------------------------------

crystall_small_config = """{
  "bias": true,
  "model_type": "gpt2",
  "block_size": 1024,
  "dropout": 0.1,
  "n_embd": 512,
  "n_head": 8,
  "n_layer": 8,
  "vocab_size": 50257
}"""

with open('/content/CrystaLLM/crystallm_v1_small/config.json', 'a') as f:
    f.write(crystall_small_config)

# ------------------------------------------------------------
# Convert the original model checkpoints from .pt to .bin format.
# ------------------------------------------------------------

import torch

# Load the checkpoint file
checkpoint = torch.load("/content/CrystaLLM/crystallm_v1_small/ckpt.pt", map_location=torch.device('cpu'))

# Extract the model parameters
params = checkpoint["model"]

# Save the parameters to a .bin file
torch.save(params, "/content/CrystaLLM/crystallm_v1_small/pytorch_model.bin")

# ------------------------------------------------------------
# Add configuration file and converted checkpoints to the Transformer API (use the generic AutoConfig and AutoModelForCausalLM classes).
# ------------------------------------------------------------

from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained('/content/CrystaLLM/crystallm_v1_small/config.json')
transformer_model = AutoModelForCausalLM.from_pretrained('/content/CrystaLLM/crystallm_v1_small', config=config)

# ------------------------------------------------------------
# Save the model using the Transformers API (the method invoked in the code cell below takes care of generating also any other required accessory file).
# ------------------------------------------------------------

transformer_model.save_pretrained("/content/CrystaLLM/crystallm_v1_small_hf")

# ------------------------------------------------------------
# The model is now ready to be uploaded to the HF's Hub (assuming you have a valid Hugging Face profile). Please read the statement at the beginning of this section about permission to share the checkpoints through the Hub.
# ------------------------------------------------------------

