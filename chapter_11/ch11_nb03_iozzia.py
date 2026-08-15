# ==========================================
# Extracted from CH11_NB03_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Small Language Model Conversion and Inference with MLC LLM
# This notebook is a companion of chapter 11 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook shows how to use [MLC LLM](https://llm.mlc.ai/) to convert and compile a Small Language Model hosted in the Hugging Face's Hub and then run inference with it on a Linux system. The model under consideration is of [RedPajama-INCITE-Instruct-3B-v1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1), but the code in this notebook applies to any other Open Source LLM hosted in the HF's Hub. Hardware acceleration is required.   
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the proper MLC LLM wheel for Linux and the CUDA drivers in this system.
# ------------------------------------------------------------

# !python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu122 mlc-ai-nightly-cu122

# ------------------------------------------------------------
# Verify the MLC installation completed successfully.
# ------------------------------------------------------------

# !mlc_llm --help

# ------------------------------------------------------------
# # Model Conversion
# ------------------------------------------------------------

# ------------------------------------------------------------
# To run a model with MLC LLM, we need to convert the model weights into MLC format. Some preliminary actions to be done: create the destination directory for the original model's weights and accessory files, install the Git extension for versioning large files and clone the HF's repo for the target model.
# ------------------------------------------------------------

# !mkdir -p dist/models && cd dist/models
# !git lfs install
# !git clone https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1

# ------------------------------------------------------------
# Convert the model weights to the MLC LLM format. The converted weights are saved into the same directory as for the original model.
# ------------------------------------------------------------

# !mlc_llm convert_weight ./RedPajama-INCITE-Instruct-3B-v1/ \
#     --quantization q4f16_1 \
#     -o dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC \
#     --device cuda:0

# ------------------------------------------------------------
# Generate the chat configuration for the converted model. The generated configuration is saved in the same directory as for the converted weights.
# ------------------------------------------------------------

# !mlc_llm gen_config ./RedPajama-INCITE-Instruct-3B-v1/ \
#     --quantization q4f16_1 --conv-template redpajama_chat \
#     -o dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC/

# ------------------------------------------------------------
# Verify that all the required files (the chat configuration file, the model's weights info, shards and tokenizer files) are within the destination directory.
# ------------------------------------------------------------

# !ls dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC

# ------------------------------------------------------------
# We need now to compile the converted model before we can run inference. Create a destination directory for the compiled model.
# ------------------------------------------------------------

# !mkdir ./dist/libs

# ------------------------------------------------------------
# Compile the model library using the specification in chat configuration file preliminary created (*mlc-chat-config.json*).
# ------------------------------------------------------------

# !mlc_llm compile ./dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC/mlc-chat-config.json \
#     --device cuda -o dist/libs/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-cuda.so

# ------------------------------------------------------------
# Verify that the model compilation completed successfully. For Linux systems and CUDA drivers the compilation directory should contain a single compiled library file. Please refer to the official MLC LLM documentation for other operating systems and hardware accelerators.
# ------------------------------------------------------------

# !ls dist/libs

# ------------------------------------------------------------
# # Chat with the Converted model using the MLC LLM Python API
# ------------------------------------------------------------

# ------------------------------------------------------------
# Create an instace of the MLCEngine for the converted model. This class supports only synchronous chat completions.
# ------------------------------------------------------------

from mlc_llm import MLCEngine

engine = MLCEngine(model="./dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC",
                   model_lib="./dist/libs/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-cuda.so")

# ------------------------------------------------------------
# Start some chat examples.
# ------------------------------------------------------------

for response in engine.chat.completions.create(
    messages=[{"role": "user", "content": "What's the meaning of life?"}],
    model="./dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC",
    stream=False,
):
    print(response)
print("\n")

for response in engine.chat.completions.create(
    messages=[{"role": "user", "content": "What's the meaning of life?"}],
    model="./dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC",
    stream=True,
):
    for choice in response.choices:
        print(choice.delta.content, end="", flush=True)
print("\n")

# ------------------------------------------------------------
# Shutdown the MLC engine.
# ------------------------------------------------------------

engine.terminate()

