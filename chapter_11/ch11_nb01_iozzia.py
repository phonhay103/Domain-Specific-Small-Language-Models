# ==========================================
# Extracted from CH11_NB01_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Small Language Model Offline Serving with vLLM
# This notebook is a companion of chapter 11 of the "Domain Specific LLMs in Action" book, author Guglielmo Iozzia, [Manning Publications](https://www.manning.com/), 2024.  
# The code in this notebook is about serving some example small language models, such as [GPT-2 small](https://huggingface.co/openai-community/gpt2) and [Microsoft's Phi-3 mini 4k Instruct](microsoft/Phi-3-mini-4k-instruct), through the [vLLM](https://github.com/vllm-project/vllm) API. The same code applies to any other Open Source LLM. Hardware acceleration is needed.  
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Install the vLLM package. Installing through `pip` requires the CUDA drivers and an NVIDIA GPU, as there is no pre-built wheel for CPU only available for Linux systems. If you want to run the code below for GPT-2 small without hardware acceleration you have to build vLLM from source instead. Please see the official vLLM documentation for steps on how to do it.  
# Restart the VM once the vLLM installation completes.
# ------------------------------------------------------------

# !pip install vllm

# ------------------------------------------------------------
# # Text Generation
# ------------------------------------------------------------

# ------------------------------------------------------------
# Import the required packages and classes.
# ------------------------------------------------------------

import torch
from vllm import LLM

# ------------------------------------------------------------
# Specify the model to use (GPT-2 small in this case) and where to offload its weights (CPU or GPU) and then create an instance of the `vLLM` engine class for it. This class, after downloading the configuration file for the given model from the HF's Hub, analyzes the configuration and prepare the model setup accordingly, before downloading checkpoitns and tokenizer. Once the download is completed, it laodt the checkpoints in the destinazion device, profile the memory, initializes the KV cache and warms up the model.
# ------------------------------------------------------------

model_id = "openai-community/gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"
vllm_model = LLM(model_id, device=device, enforce_eager=True)

# ------------------------------------------------------------
# Provide a prompt.
# ------------------------------------------------------------

prompt = "Once upon a time in a land far away"

# ------------------------------------------------------------
# Implement a custom function to generate text with the given model in vLLM.
# ------------------------------------------------------------

def generate_text(prompt, max_length=50):
    input_ids = vllm_model.tokenizer.encode(prompt,
                                            return_tensors="pt").to(device)

    generated_ids = vllm_model.generate(
        input_ids=input_ids,
        max_length=max_length,
        do_sample=True,
        num_return_sequences=1
    )

    generated_text = vllm_model.tokenizer.decode(generated_ids[0],
                                                 skip_special_tokens=True)

    return generated_text

# ------------------------------------------------------------
# Call the `generate_text` function and display the generated text.
# ------------------------------------------------------------

output = vllm_model.generate(prompt)
for item in output:
    prompt = item.prompt
    generated_text = item.outputs[0].text
    print(generated_text)

# ------------------------------------------------------------
# # Chat
# ------------------------------------------------------------

del vllm_model

import gc
gc.collect()
torch.cuda.empty_cache()

# ------------------------------------------------------------
# Specify the model to use (Microsoft Phi-3 mini 4k Instruct in this case) and where to offload its weights (CPU or GPU) and then create an instance of the `vLLM` engine class for it. Also using the `SamplingParams` class to set the temperature for the model.
# ------------------------------------------------------------

import torch
from vllm import LLM, SamplingParams

model_id = "microsoft/Phi-3-mini-4k-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
llm = LLM(model=model_id, device=device, dtype=torch.half)

sampling_params = SamplingParams(temperature=0.5)

# ------------------------------------------------------------
# Define a function to print the generated outputs.
# ------------------------------------------------------------

def print_outputs(outputs):
  print("=" * 80)
  for output in outputs:
      prompt = output.prompt
      generated_text = output.outputs[0].text
      print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
  print("-" * 80)

# ------------------------------------------------------------
# Setup the input in a format to chat with the selected model, then run the `chat` command and display the output.
# ------------------------------------------------------------

conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": "Hi"
    },
    {
        "role": "assistant",
        "content": "Hi! How can I assist you today?"
    },
    {
        "role": "user",
        "content": "Write an essay about the Monte Carlo Tree Search algorithm.",
    },
]
outputs = llm.chat(conversation,
                   sampling_params=sampling_params,
                   use_tqdm=False)
print_outputs(outputs)

# ------------------------------------------------------------
# The same way, batch inference can be run with the vLLM's chat API.
# ------------------------------------------------------------

conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": "Hi"
    },
    {
        "role": "assistant",
        "content": "Hi! How can I assist you today?"
    },
    {
        "role": "user",
        "content": "Write an essay about the Monte Carlo Tree Search algorithm.",
    },
]
conversations = [conversation for _ in range(10)]

outputs = llm.chat(messages=conversations,
                   sampling_params=sampling_params,
                   use_tqdm=True)
print_outputs(outputs)

# ------------------------------------------------------------
# It is also possible to specify a chat template to the vLLM's chat API. Download one for the Phi-3 mini model.
# ------------------------------------------------------------

# !wget https://raw.githubusercontent.com/chujiezheng/chat_templates/main/chat_templates/phi-3.jinja

# ------------------------------------------------------------
# Then supply it to the inference engine class for running chats.
# ------------------------------------------------------------

with open('phi-3.jinja', "r") as f:
  chat_template = f.read()

outputs = llm.chat(
  conversations,
  sampling_params=sampling_params,
  use_tqdm=False,
  chat_template=chat_template,
)

outputs

# ------------------------------------------------------------
# # GGUF
# ------------------------------------------------------------

del llm

import gc
gc.collect()
torch.cuda.empty_cache()

# ------------------------------------------------------------
# vLLM supports also models in [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) format. The code example in the cells below show evidence of this using a [GGUF version](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) of the Phi-3 mini 4k Instruct model already available from Microsoft in the HF's Hub. Download it and the tokenizer associated to the vanilla model too.
# ------------------------------------------------------------

from huggingface_hub import hf_hub_download

repo_id = "microsoft/Phi-3-mini-4k-instruct-gguf"
filename = "Phi-3-mini-4k-instruct-q4.gguf"
tokenizer = "microsoft/Phi-3-mini-4k-instruct"
model = hf_hub_download(repo_id, filename=filename)

# ------------------------------------------------------------
# Before starting the chat, uninstall the PyTorch-native training-to-serving model optimization package (lately available by default in the Colab VM), as it interferes with the vLLM GGUF inference.
# ------------------------------------------------------------

# !pip uninstall -y torchao

# ------------------------------------------------------------
# Once the download is completed, create an instance of the vLLM engine starting from the model in the local file system. Also setup some prompt to chat with it and configure some sampling parameters (`temperature` and `max_tokens`).
# ------------------------------------------------------------

from vllm import LLM, SamplingParams

prompts = [
    "How to explain Internet for a medieval knight?",
    "What's the future of AI?",
]
prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]
sampling_params = SamplingParams(temperature=0, max_tokens=128)

llm = LLM(model=model, tokenizer=tokenizer)

# ------------------------------------------------------------
# Start the chat with the GGUF model and display the outputs.
# ------------------------------------------------------------

outputs = llm.chat(prompts, sampling_params)
print_outputs(outputs)

