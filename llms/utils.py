import argparse
from typing import Any

from llms import (
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    generate_from_steered_model,
    lm_config,
)

APIInput = str | list[Any] | dict[str, Any]


def call_llm(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
) -> str:
    response: str
    if lm_config.provider == "openai":
        if lm_config.mode == "chat":
            assert isinstance(prompt, list)
            response = generate_from_openai_chat_completion(
                messages=prompt,
                model=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                top_p=lm_config.gen_config["top_p"],
                context_length=lm_config.gen_config["context_length"],
                max_tokens=lm_config.gen_config["max_tokens"],
                stop_token=None,
            )
        elif lm_config.mode == "completion":
            assert isinstance(prompt, str)
            response = generate_from_openai_completion(
                prompt=prompt,
                engine=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                max_tokens=lm_config.gen_config["max_tokens"],
                top_p=lm_config.gen_config["top_p"],
                stop_token=lm_config.gen_config["stop_token"],
            )
        else:
            raise ValueError(
                f"OpenAI models do not support mode {lm_config.mode}"
            )
    elif lm_config.provider == "huggingface":
        assert isinstance(prompt, str)
        response = generate_from_huggingface_completion(
            prompt=prompt,
            model_endpoint=lm_config.gen_config["model_endpoint"],
            temperature=lm_config.gen_config["temperature"],
            top_p=lm_config.gen_config["top_p"],
            stop_sequences=lm_config.gen_config["stop_sequences"],
            max_new_tokens=lm_config.gen_config["max_new_tokens"],
        )
    elif lm_config.provider == "steered":
        response = generate_from_steered_model(
            prompt=prompt,
            model_id=lm_config.model,
            temperature=lm_config.gen_config["temperature"],
            top_p=lm_config.gen_config["top_p"],
            max_new_tokens=lm_config.gen_config["max_new_tokens"],
            vector_path=lm_config.gen_config.get("vector_path"),
            steering_layer=lm_config.gen_config.get("steering_layer", 20),
            steering_coeff=lm_config.gen_config.get("steering_coeff", 0.0),
            steering_type=lm_config.gen_config.get("steering_type", "response"),
        )
    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )

    return response
