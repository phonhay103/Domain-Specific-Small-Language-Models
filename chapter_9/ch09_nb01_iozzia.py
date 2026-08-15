# ==========================================
# Extracted from CH09_NB01_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Using FlexGen to Offload OPT Models' Weights to RAM and Disk
# This notebook is a companion of chapter 9 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is to perform inference with a Meta AI's OPT models, by offloading part of models' weights from VRAM to RAM and/or disk, using the [FlexGen](https://github.com/FMInference/FlexLLMGen/) generation engine programmatically. While the code refers to the [OPT 1.3 B](https://huggingface.co/facebook/opt-1.3b) model, the same applies to any other model from the same family. It requires hardware acceleration to be executed.  
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the FlexGen from source.
# ------------------------------------------------------------

# !git clone https://github.com/FMInference/FlexLLMGen.git
# %cd FlexLLMGen
# !pip install -e .

# ------------------------------------------------------------
# # Using FlexGen and the Transformers library programmatically
# ------------------------------------------------------------

# ------------------------------------------------------------
# Import the required FlexGen classes.
# ------------------------------------------------------------

from flexllmgen.flex_opt import (Policy, OptLM, ExecutionEnv, CompressionConfig,
        str2bool)

# ------------------------------------------------------------
# Download the OPT 1.3 B tokenizer form the Hugging Face's Hub.
# ------------------------------------------------------------

from transformers import AutoTokenizer

model_id = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tokenizer.add_bos_token = False
stop = tokenizer("\n").input_ids[0]

# ------------------------------------------------------------
# Setup the FlexGen execution environment.
# ------------------------------------------------------------

offload_dir = './flexgen_offload'
env = ExecutionEnv.create(offload_dir)

# ------------------------------------------------------------
# Prepare a list of prompts for batch inference.
# ------------------------------------------------------------

prompts = [
    "Question: Where were the 2004 Olympics held?\n"
    "Answer: Athens, Greece\n"
    "Question: What is the longest river on the earth?\n"
    "Answer:",

    "Extract the airport codes from this text.\n"
    "Text: \"I want a flight from New York to San Francisco.\"\n"
    "Airport codes: JFK, SFO.\n"
    "Text: \"I want you to book a flight from Phoenix to Las Vegas.\"\n"
    "Airport codes:",
]

# ------------------------------------------------------------
# Setup an offloading policy.
# ------------------------------------------------------------

policy = Policy(len(prompts), 1,
                70, 30, 70, 30, 100, 0,
                overlap=True, sep_layer=True, pin_weight=True,
                cpu_cache_compute=True, attn_sparsity=1.0,
                compress_weight=True,
                comp_weight_config=CompressionConfig(
                    num_bits=4, group_size=64,
                    group_dim=0, symmetric=False),
                compress_cache=False, # Set compress_cache to False
                comp_cache_config=CompressionConfig(
                    num_bits=4, group_size=64,
                    group_dim=2, symmetric=False)
                )

# ------------------------------------------------------------
# Prepare the model to be executed through the FlexGen inference engine and following the preliminary defined offloading policies. This step also downloads the model's checkpoints from the Hugging Face's Hub and manages the conversion process.
# ------------------------------------------------------------

path = '~/opt_weights'
model = OptLM(model_id, env, path, policy)

# ------------------------------------------------------------
# Generate text for the given set of prompts and then display the generated result for each one.
# ------------------------------------------------------------

print("Generate...")
inputs = tokenizer(prompts, padding="max_length", max_length=128)
output_ids = model.generate(
    inputs.input_ids,
    do_sample=True,
    temperature=0.7,
    max_new_tokens=32,
    stop=stop)
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
print("Outputs:\n" + 70 * '-')
for i in [0, len(outputs)-1]:
    print(f"{i}: {outputs[i]}")
    print("-" * 70)

# ------------------------------------------------------------
# Shutdown the FlexGen execution environment when done.
# ------------------------------------------------------------

print("Shutting down...")
env.close_copy_threads()

