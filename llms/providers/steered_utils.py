"""Local steered model provider for activation-steered inference.

Uses transformers + ActivationSteerer from agency_vectors to run
a locally-loaded model with optional activation steering vectors.
"""

import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow importing from agency_vectors repo
_AGENCY_VECTORS_ROOT = Path(__file__).resolve().parents[2] / "agency_vectors"
if _AGENCY_VECTORS_ROOT.is_dir():
    sys.path.insert(0, str(_AGENCY_VECTORS_ROOT))

from activation_steer import ActivationSteerer

# ---------------------------------------------------------------------------
# Singleton cache – model + tokenizer are expensive to load, reuse across calls
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, Any] = {}


def _get_model_and_tokenizer(model_id: str):
    """Load (or return cached) model and tokenizer."""
    if model_id not in _MODEL_CACHE:
        print(f"[steered] Loading model {model_id} …")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        _MODEL_CACHE[model_id] = (model, tokenizer)
        print(f"[steered] Model loaded on {model.device}")

    return _MODEL_CACHE[model_id]


_VECTOR_CACHE: dict[str, torch.Tensor] = {}


def _get_steering_vector(vector_path: str, layer: int) -> torch.Tensor:
    """Load (or return cached) steering vector for a specific layer."""
    cache_key = f"{vector_path}:{layer}"
    if cache_key not in _VECTOR_CACHE:
        print(f"[steered] Loading vector {vector_path} layer {layer}")
        vectors = torch.load(vector_path, weights_only=False)
        _VECTOR_CACHE[cache_key] = vectors[layer]
    return _VECTOR_CACHE[cache_key]


def generate_from_steered_model(
    prompt: str | list[dict[str, str]],
    model_id: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_new_tokens: int = 384,
    stop_sequences: list[str] | None = None,
    # Steering params
    vector_path: str | None = None,
    steering_layer: int = 20,
    steering_coeff: float = 0.0,
    steering_type: str = "response",
) -> str:
    """Generate a completion from a locally-loaded steered model.

    Args:
        prompt: Either a plain string or a list of chat messages
            [{"role": "system", "content": ...}, {"role": "user", ...}].
        model_id: HuggingFace model ID or local path.
        temperature: Sampling temperature (0 = greedy).
        top_p: Nucleus sampling threshold.
        max_new_tokens: Maximum tokens to generate.
        stop_sequences: Optional stop strings (unused for now; model stop tokens apply).
        vector_path: Path to a .pt persona vector file. None = no steering.
        steering_layer: Which layer to steer (1-indexed, as in agency_vectors).
        steering_coeff: Steering coefficient. 0 = no steering.
        steering_type: "response", "prompt", or "all".

    Returns:
        The generated text (string).
    """
    model, tokenizer = _get_model_and_tokenizer(model_id)

    # Format input – accept both chat messages and raw strings
    if isinstance(prompt, list):
        # Chat message format: convert to model's chat template
        text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    # Determine whether to steer
    use_steering = (
        vector_path is not None
        and steering_coeff != 0.0
        and Path(vector_path).exists()
    )

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "top_p": top_p,
        "use_cache": True,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        if use_steering:
            vector = _get_steering_vector(vector_path, steering_layer)
            # agency_vectors uses 0-indexed layers internally
            with ActivationSteerer(
                model,
                vector,
                coeff=steering_coeff,
                layer_idx=steering_layer - 1,
                positions=steering_type,
            ):
                output = model.generate(**inputs, **gen_kwargs)
        else:
            output = model.generate(**inputs, **gen_kwargs)

    response = tokenizer.decode(
        output[0][prompt_len:], skip_special_tokens=True
    )
    return response
