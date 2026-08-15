# ==========================================
# Extracted from CH15_NB02_Iozzia.ipynb
# ==========================================

# ------------------------------------------------------------
# # Adapt an SLM to reason over a custom domain and dataset using GRPO
# 
# This notebook is a companion of chapter 15 of the "Domain-Specific Small Language Models" book, author Guglielmo Iozzia, Manning Publications, 2026.  
# The code in this notebook is an example of building a reasoning SLM [Qwen 2.5 3B Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), to specialize it on a QA task about additive manufacturing (3D printing) knowledge through GRPO (Group Relative Policy Optimization) and [QLoRA](https://arxiv.org/abs/2305.14314). Hardware acceleration (GPU) is needed.  
# More details about the code can be found in the related book's chapter.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Upgrade the HF's Datasets package to the latest version.
# ------------------------------------------------------------

# !pip install --force-reinstall datasets

# ------------------------------------------------------------
# Install the missing dependencies in the Colab VM (Unsloth, HF's TRL, vLLM).
# ------------------------------------------------------------

import os

# Uninstall the conflicting versions
# !pip uninstall -y huggingface-hub transformers tokenizers

# Install compatible versions for vLLM, Unsloth, and TRL
# !pip install huggingface-hub==0.34.0
# !pip install transformers==4.56.2
# !pip install tokenizers==0.22.2

# ------------------------------------------------------------
# Enable fast RL (GRPO in TRL).
# ------------------------------------------------------------

from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", FastLanguageModel)

# ------------------------------------------------------------
# Download the 4-bit quantized Qwen 2.5 3B Instruct model from the HF's Hub. The ```fast_inference argument``` is set to ```True```, to enable vLLM fast inference. The ```gpu_memory_utilization``` argument is set to use 50% of the available GPU memory, to prevent out of memory error in this use case. Feel free to adjust it to different percentage. ```max_seq_length``` has been set to 2048 as this is enough for this use case, but if you need longer reasoning traces you need to increase it. ```lora_rank``` has been set to 64. Increasing this value would make the model smarter, but slower.
# 
# ------------------------------------------------------------

import torch
from unsloth import is_bfloat16_supported

max_seq_length = 2048
lora_rank = 64

model_id = "Qwen/Qwen2.5-3B-Instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5,
)

# ------------------------------------------------------------
# Prepare the downloaded model for PEFT (QLoRA in this case).
# ------------------------------------------------------------

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ------------------------------------------------------------
# Define the system prompt and the reasoning (model output) templates.
# ------------------------------------------------------------

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# ------------------------------------------------------------
# Define a custom function to load the addictive manufacturing for reasoning dataset from the HF Hub. The function also prepares the prompts fro the training and removes the unused columns.
# ------------------------------------------------------------

from datasets import load_dataset, Dataset

def get_datasets(split = "train") -> Dataset:
  data_qa = load_dataset("g3lu/addictive_manufacturing_reasoning", split=split)

  data_qa = data_qa.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "You are an expert in additive manufacturing.\n" +
                          "\n\nAnswer the following question:\n" +
                          x['question'] +
                          " You need to carefully review the question and reason before answering."
            },
        ],
        'answer': x['answer'],
        'db_set': 'addictive_manufacturing_reasoning'
    })

  data_qa = data_qa.remove_columns(['question', 'reason'])

  return data_qa

# ------------------------------------------------------------
# The dataset contains 10,000 samples, but we don’t need all of them for training and testing in this example. We will use only 100 samples (90 for training and 10 for testing).
# ------------------------------------------------------------

slice_split = "train[0:100]"
dataset = get_datasets(slice_split)
dataset = dataset.shuffle(seed=42)
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]
print(f"train size: {len(train_dataset)}, test size: {len(test_dataset)}")

train_dataset[1]

# ------------------------------------------------------------
# Define a helper function to extract the content within an XML-formatted model's answer.
# ------------------------------------------------------------

def extract_xml_answer(text: str) -> str:
      answer = text.split("<answer>")[-1]
      answer = answer.split("</answer>")[0]

      return answer.strip()

# ------------------------------------------------------------
# Define some reward functions. The first one is related to the correctness of the generated answers.
# ------------------------------------------------------------

def correctness_reward_func(prompts, completions, answer, db_set, **kwargs) -> list[float]:
      responses = [completion[0]['content'] for completion in completions]
      q = prompts[0][-1]['content']
      extracted_responses = [extract_xml_answer(r) for r in responses]
      print('-'*20, f"Question:\\n{q}", f"\\nAnswer:\\n{answer[0]}", f"\\nResponse:\\n{responses[0]}", f"\\nExtracted:\\n{extracted_responses[0]}")
      rewards = []
      for r,a,dt in zip(extracted_responses, answer, db_set):
        if dt == "gsm8k":
          if a in r:
            rewards.append(1.0)
          elif r == a:
            rewards.append(2.0)
          else:
            rewards.append(0.0)
      else:
        rewards.append(2.0 if r.lower() == a.strip().lower() else 0.0)

      return rewards

# ------------------------------------------------------------
# A second defined reward function is about assigning a score to the responses from the model, specific for given datasets.
# ------------------------------------------------------------

def int_reward_func(completions, db_set, **kwargs) -> list[float]:
      responses = [completion[0]['content'] for completion in completions]
      extracted_responses = [extract_xml_answer(r) for r in responses]
      rewards = []
      for r,dt in zip(extracted_responses,db_set):
          if dt == "gsm8k":
            rewards.append(0.5 if r.isdigit() else 0.0)
          elif dt == "pubmedqa":
            rewards.append(0.5 if ('yes' in r.lower() or 'no' in r.lower() or 'maybe' in r.lower()) else 0.0)
      else:
        rewards.append(0.5 if ('a' in r.lower() or 'b' in r.lower() or 'c' in r.lower() or 'd' in r.lower()) else 0.0)

      return rewards

# ------------------------------------------------------------
# Other two reward function are about the validation of the XML formatting of the generated answers (one to perform a strict validation, another to check how close to the expected formatting a generated answer is).
# ------------------------------------------------------------

import re

def strict_format_reward_func(completions, **kwargs) -> list[float]:
      """Reward function that checks if the completion has a specific format."""
      pattern = r"^<reasoning>\\n.*?\\n</reasoning>\\n<answer>\\n.*?\\n</answer>\\n$"
      responses = [completion[0]["content"] for completion in completions]
      matches = [re.match(pattern, r) for r in responses]

      return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
      """Reward function that checks if the completion has a specific format."""
      pattern = r"<reasoning>.*?</reasoning>\\s*<answer>.*?</answer>"
      responses = [completion[0]["content"] for completion in completions]
      matches = [re.match(pattern, r) for r in responses]

      return [0.5 if match else 0.0 for match in matches]

# ------------------------------------------------------------
# A final reward function is about counting the number of opening and closing ```<reasoning>``` and ```<answer>``` tags in a generated answers.
# ------------------------------------------------------------

def count_xml(text) -> float:
      count = 0.0
      if text.count("<reasoning>\\n") == 1:
        count += 0.125
      if text.count("\\n</reasoning>\\n") == 1:
        count += 0.125
      if text.count("\\n<answer>\\n") == 1:
        count += 0.125
        count -= len(text.split("\\n</answer>\\n")[-1])*0.001
      if text.count("\\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\\n</answer>")[-1]) - 1)*0.001

      return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
      contents = [completion[0]["content"] for completion in completions]

      return [count_xml(c) for c in contents]

# ------------------------------------------------------------
# Set up a training configuration. We use the TRL’s GRPO-specific ```GRPOConfig``` to set vLLM as the backend (```use_vllm``` set to ```True``` to make it faster), the maximum prompt and completion lengths, the number of training steps, the checkpoint saving frequency, whether to keep results and checkpoints locally, and the output directory for the experiment. ```gradient_accumulation_steps``` has been set to 1, but you can increase it to 4 for smoother training. ```num_generations``` is set to 6: decrease it if run out of memory.  
# Then create an instance of TRL's GRPO-specific ```GRPOTrainer``` that uses the ```GRPOConfig```. The custom reward functions are assigned to it.
# ------------------------------------------------------------

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    use_vllm = True,
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    num_generations = 6,
    max_prompt_length = 1024,
    max_completion_length = 1024,
    max_steps = 10,
    save_steps = 10,
    max_grad_norm = 0.1,
    report_to = "none",
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset=test_dataset,
)

# ------------------------------------------------------------
# Start the training. The output of the training is captured to be saved to file at the end.
# ------------------------------------------------------------

# %%capture cap
trainer.train()

# ------------------------------------------------------------
# Save the training captured output to file when training is completed.
# ------------------------------------------------------------

f = open("training_output.txt", "w")
print(cap, file=f)
f.close()

# ------------------------------------------------------------
# Evaluate the baseline model.
# ------------------------------------------------------------

test_prompt = """
What strategies can be employed to balance the need for simplification
in simulation models with the preservation of critical geometric features,
such as lattice structures or organic shapes, that are characteristic
of additive manufacturing designs and significantly impact
the product's performance, while also considering the limitations imposed
by computational resources and the potential for introducing significant errors
in the results?
"""
text = tokenizer.apply_chat_template([
      {"role" : "user", "content" : test_prompt}
    ], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
        temperature = 0.8,
        top_p = 0.95,
        max_tokens = 1024,
)
output = model.fast_generate(
        [text],
        sampling_params = sampling_params,
        lora_request = None,
)[0].outputs[0].text

output

# ------------------------------------------------------------
# Save the QLoRA weights.
# ------------------------------------------------------------

model.save_pretrained("grpo_saved_lora")

# ------------------------------------------------------------
# Evaluate the model with the QLoRA weights against the same prompt as for the baseline model.
# ------------------------------------------------------------

text = tokenizer.apply_chat_template(
        [
            {"role" : "system", "content" : SYSTEM_PROMPT},
            {"role" : "user", "content" : test_prompt},
        ], tokenize = False, add_generation_prompt = True)

# Tokenize the input text
inputs = tokenizer(text, return_tensors = "pt").to("cuda")

# Use HuggingFace generate arguments directly since fast_inference is False
outputs = model.generate(
    **inputs,
    max_new_tokens = 1024,
    temperature = 0.8,
    top_p = 0.95,
    do_sample = True,
)

# Decode the generated output
output = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Remove the input prompt part from the output
output = output.replace(text, "")

output

# ------------------------------------------------------------
# Merge the QLoRA weights with the baseline.
# ------------------------------------------------------------

model.save_pretrained_merged("model", tokenizer)

