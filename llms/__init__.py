"""This module is adapt from https://github.com/zeno-ml/zeno-build"""
from .providers.hf_utils import generate_from_huggingface_completion
from .providers.openai_utils import (
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
)
from .utils import call_llm

try:
    from .providers.steered_utils import generate_from_steered_model
except ModuleNotFoundError as exc:
    def generate_from_steered_model(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(
            "Steered inference dependencies are missing in the WebArena venv. "
            "Install torch/transformers in that environment before using "
            "--provider steered."
        ) from exc

__all__ = [
    "generate_from_openai_completion",
    "generate_from_openai_chat_completion",
    "generate_from_huggingface_completion",
    "generate_from_steered_model",
    "call_llm",
]
