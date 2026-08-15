"""Using FlexGen to Offload OPT Model Weights to RAM and Disk.

Companion script for Chapter 9 of "Domain Specific LLMs in Action"
(Guglielmo Iozzia, Manning Publications, 2024).

Performs batch inference with Meta AI's OPT 1.3 B model using the FlexGen
engine to offload model weights from VRAM to RAM and/or disk.  The same
approach applies to any other model in the OPT family.  Requires GPU
hardware acceleration.

Setup notes (run once before executing this script):
  # git clone https://github.com/FMInference/FlexLLMGen.git
  # cd FlexLLMGen
  # pip install -e .
"""

# third-party
from flexllmgen.flex_opt import (
    CompressionConfig,
    ExecutionEnv,
    OptLM,
    Policy,
    str2bool,  # noqa: F401 – re-exported for callers that may need it
)
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID: str = "facebook/opt-1.3b"
OFFLOAD_DIR: str = "./flexgen_offload"
OPT_WEIGHTS_PATH: str = "~/opt_weights"

# Tokenization / generation settings
TOKENIZER_PADDING_SIDE: str = "left"
INPUT_MAX_LENGTH: int = 128
MAX_NEW_TOKENS: int = 32
TEMPERATURE: float = 0.7

# Offloading policy knobs (GPU%, CPU% for weights and cache respectively)
POLICY_GPU_WEIGHT_PERCENT: int = 70
POLICY_CPU_WEIGHT_PERCENT: int = 30
POLICY_GPU_CACHE_PERCENT: int = 70
POLICY_CPU_CACHE_PERCENT: int = 30
POLICY_GPU_PERCENT: int = 100
POLICY_CPU_PERCENT: int = 0

# Weight compression settings
WEIGHT_NUM_BITS: int = 4
WEIGHT_GROUP_SIZE: int = 64
WEIGHT_GROUP_DIM: int = 0

# Cache compression settings (disabled by default)
CACHE_NUM_BITS: int = 4
CACHE_GROUP_SIZE: int = 64
CACHE_GROUP_DIM: int = 2

# Few-shot prompts for batch inference
PROMPTS: list[str] = [
    (
        "Question: Where were the 2004 Olympics held?\n"
        "Answer: Athens, Greece\n"
        "Question: What is the longest river on the earth?\n"
        "Answer:"
    ),
    (
        'Extract the airport codes from this text.\n'
        'Text: "I want a flight from New York to San Francisco."\n'
        "Airport codes: JFK, SFO.\n"
        'Text: "I want you to book a flight from Phoenix to Las Vegas."\n'
        "Airport codes:"
    ),
]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def build_tokenizer(model_id: str) -> AutoTokenizer:
    """Load the OPT tokenizer from the HF Hub with left-padding enabled."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, padding_side=TOKENIZER_PADDING_SIDE
    )
    tokenizer.add_bos_token = False
    return tokenizer


def build_policy(num_prompts: int) -> Policy:
    """Create a FlexGen offloading policy for the given batch size."""
    return Policy(
        num_prompts,
        1,
        POLICY_GPU_WEIGHT_PERCENT,
        POLICY_CPU_WEIGHT_PERCENT,
        POLICY_GPU_CACHE_PERCENT,
        POLICY_CPU_CACHE_PERCENT,
        POLICY_GPU_PERCENT,
        POLICY_CPU_PERCENT,
        overlap=True,
        sep_layer=True,
        pin_weight=True,
        cpu_cache_compute=True,
        attn_sparsity=1.0,
        compress_weight=True,
        comp_weight_config=CompressionConfig(
            num_bits=WEIGHT_NUM_BITS,
            group_size=WEIGHT_GROUP_SIZE,
            group_dim=WEIGHT_GROUP_DIM,
            symmetric=False,
        ),
        compress_cache=False,  # cache compression disabled
        comp_cache_config=CompressionConfig(
            num_bits=CACHE_NUM_BITS,
            group_size=CACHE_GROUP_SIZE,
            group_dim=CACHE_GROUP_DIM,
            symmetric=False,
        ),
    )


def run_batch_inference(
    model: OptLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    stop_token_id: int,
) -> list[str]:
    """Tokenise *prompts*, run generation, and return decoded output strings."""
    inputs = tokenizer(prompts, padding="max_length", max_length=INPUT_MAX_LENGTH)
    output_ids = model.generate(
        inputs.input_ids,
        do_sample=True,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
        stop=stop_token_id,
    )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def main() -> None:
    """Orchestrate FlexGen setup, batch inference, and environment teardown."""
    tokenizer = build_tokenizer(MODEL_ID)
    stop_token_id: int = tokenizer("\n").input_ids[0]

    env = ExecutionEnv.create(OFFLOAD_DIR)
    policy = build_policy(len(PROMPTS))

    # Download model weights from HF Hub and prepare for FlexGen execution
    model = OptLM(MODEL_ID, env, OPT_WEIGHTS_PATH, policy)

    print("Generate...")
    outputs = run_batch_inference(model, tokenizer, PROMPTS, stop_token_id)

    print("Outputs:\n" + "-" * 70)
    for i in [0, len(outputs) - 1]:
        print(f"{i}: {outputs[i]}")
        print("-" * 70)

    print("Shutting down...")
    env.close_copy_threads()


if __name__ == "__main__":
    main()
