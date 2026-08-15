"""
Small Language Model Conversion and Inference with MLC LLM.

This script is a companion of chapter 11 of the "Domain Specific LLMs in Action" book,
author Guglielmo Iozzia, Manning Publications, 2024.

The code shows how to use MLC LLM (https://llm.mlc.ai/) to convert and compile a Small
Language Model hosted in the Hugging Face Hub and then run inference with it on a Linux
system. The model under consideration is RedPajama-INCITE-Instruct-3B-v1, but the code
applies to any other Open Source LLM hosted in the HF Hub. Hardware acceleration is
required.

More details about the code can be found in the related book's chapter.

--- Prerequisite shell steps (run once before executing this script) ---
# Install the proper MLC LLM wheel for Linux + CUDA 12.2:
#   python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cu122 mlc-ai-nightly-cu122

# Verify installation:
#   mlc_llm --help

# --- Model Conversion (run once) ---
# Create destination directories and clone model weights:
#   mkdir -p dist/models && cd dist/models
#   git lfs install
#   git clone https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1

# Convert weights to MLC format:
#   mlc_llm convert_weight ./RedPajama-INCITE-Instruct-3B-v1/ \\
#       --quantization q4f16_1 \\
#       -o dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC \\
#       --device cuda:0

# Generate chat configuration:
#   mlc_llm gen_config ./RedPajama-INCITE-Instruct-3B-v1/ \\
#       --quantization q4f16_1 --conv-template redpajama_chat \\
#       -o dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC/

# Compile the model library:
#   mkdir ./dist/libs
#   mlc_llm compile ./dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC/mlc-chat-config.json \\
#       --device cuda -o dist/libs/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-cuda.so
"""

# third-party
from mlc_llm import MLCEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIR = "./dist/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-MLC"
MODEL_LIB = "./dist/libs/RedPajama-INCITE-Instruct-3B-v1-q4f16_1-cuda.so"
CHAT_QUESTION = "What's the meaning of life?"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def run_sync_completion(engine: MLCEngine, question: str) -> None:
    """Run a single synchronous chat completion and print the response."""
    for response in engine.chat.completions.create(
        messages=[{"role": "user", "content": question}],
        model=MODEL_DIR,
        stream=False,
    ):
        print(response)
    print("\n")


def run_streaming_completion(engine: MLCEngine, question: str) -> None:
    """Run a streaming chat completion and print tokens as they arrive."""
    for response in engine.chat.completions.create(
        messages=[{"role": "user", "content": question}],
        model=MODEL_DIR,
        stream=True,
    ):
        for choice in response.choices:
            print(choice.delta.content, end="", flush=True)
    print("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Create an MLCEngine, run example completions, then shut down cleanly."""
    # Create an instance of the MLCEngine for the converted model.
    # This class supports only synchronous chat completions.
    engine = MLCEngine(model=MODEL_DIR, model_lib=MODEL_LIB)

    run_sync_completion(engine, CHAT_QUESTION)
    run_streaming_completion(engine, CHAT_QUESTION)

    engine.terminate()


if __name__ == "__main__":
    main()
