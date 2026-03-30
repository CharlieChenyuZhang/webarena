import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tiktoken
from beartype import beartype

from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
)
from browser_env.utils import Observation, StateInfo
from llms import (
    call_llm,
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)
from llms.tokenizers import Tokenizer


class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
        result_dir: str | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError


class TeacherForcingAgent(Agent):
    """Agent that follows a pre-defined action sequence"""

    def __init__(self) -> None:
        super().__init__()

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def set_actions(self, action_seq: str | list[str]) -> None:
        if isinstance(action_seq, str):
            action_strs = action_seq.strip().split("\n")
        else:
            action_strs = action_seq
        action_strs = [a.strip() for a in action_strs]

        actions = []
        for a_str in action_strs:
            try:
                if self.action_set_tag == "playwright":
                    cur_action = create_playwright_action(a_str)
                elif self.action_set_tag == "id_accessibility_tree":
                    cur_action = create_id_based_action(a_str)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
            except ActionParsingError as e:
                cur_action = create_none_action()

            cur_action["raw_prediction"] = a_str
            actions.append(cur_action)

        self.actions: list[Action] = actions

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        return self.actions.pop(0)

    def reset(
        self,
        test_config_file: str,
        result_dir: str | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        with open(test_config_file) as f:
            ref_actions = json.load(f)["reference_action_sequence"]
            tag = ref_actions["action_set_tag"]
            action_seq = ref_actions["action_sequence"]
            self.set_action_set_tag(tag)
            self.set_actions(action_seq)


class PromptAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.current_task_id: int | None = None
        self.current_trace_path: Path | None = None
        self.current_intent: str | None = None
        self.current_task_config_file: str | None = None
        self.run_label: str | None = None
        self.model_variant: str | None = None
        self.condition_name: str | None = None
        self.steering_vector_path: str | None = None
        self.steering_layer: int | None = None
        self.steering_coeff: float | None = None
        self.steering_type: str | None = None
        self.model_target: str | None = None
        self.step_idx = 0

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def _to_jsonable(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        return repr(value)

    def _log_model_event(
        self,
        *,
        prompt: Any,
        response: str,
        parsed_response: str | None,
        action: Action,
        parse_error: str | None = None,
    ) -> None:
        if self.current_trace_path is None:
            return

        event = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "step_idx": self.step_idx,
            "task_id": self.current_task_id,
            "intent": self.current_intent,
            "task_config_file": self.current_task_config_file,
            "run_label": self.run_label,
            "model_variant": self.model_variant,
            "condition_name": self.condition_name,
            "provider": self.lm_config.provider,
            "model": self.lm_config.model,
            "model_target": self.model_target,
            "steering_vector_path": self.steering_vector_path,
            "steering_layer": self.steering_layer,
            "steering_coeff": self.steering_coeff,
            "steering_type": self.steering_type,
            "prompt": self._to_jsonable(prompt),
            "response": response,
            "parsed_response": parsed_response,
            "action": self._to_jsonable(action),
            "parse_error": parse_error,
        }
        with self.current_trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    @beartype
    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any]
    ) -> Action:
        prompt = self.prompt_constructor.construct(
            trajectory, intent, meta_data
        )
        lm_config = self.lm_config
        n = 0
        while True:
            response = call_llm(lm_config, prompt)
            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            n += 1
            try:
                parsed_response = self.prompt_constructor.extract_action(
                    response
                )
                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
                action["raw_prediction"] = response
                self._log_model_event(
                    prompt=prompt,
                    response=response,
                    parsed_response=parsed_response,
                    action=action,
                )
                break
            except ActionParsingError as e:
                if n >= lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    self._log_model_event(
                        prompt=prompt,
                        response=response,
                        parsed_response=None,
                        action=action,
                        parse_error=repr(e),
                    )
                    break

        self.step_idx += 1
        return action

    def reset(
        self,
        test_config_file: str,
        result_dir: str | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.step_idx = 0
        with open(test_config_file, "r", encoding="utf-8") as f:
            test_config = json.load(f)
        self.current_task_id = test_config.get("task_id")
        self.current_intent = test_config.get("intent")
        self.current_task_config_file = test_config_file
        self.run_label = None
        self.model_variant = None
        self.condition_name = None
        self.steering_vector_path = None
        self.steering_layer = None
        self.steering_coeff = None
        self.steering_type = None
        self.model_target = None
        if run_metadata is not None:
            self.run_label = run_metadata.get("run_label")
            self.model_variant = run_metadata.get("model_variant")
            self.condition_name = run_metadata.get("condition_name")
            self.steering_vector_path = run_metadata.get("steering_vector_path")
            self.steering_layer = run_metadata.get("steering_layer")
            self.steering_coeff = run_metadata.get("steering_coeff")
            self.steering_type = run_metadata.get("steering_type")
            self.model_target = run_metadata.get("model_target")
        if self.current_task_id is not None and result_dir is not None:
            task_trace_dir = Path(result_dir) / "model_traces"
            task_trace_dir.mkdir(parents=True, exist_ok=True)
            self.current_trace_path = (
                task_trace_dir / f"task_{self.current_task_id}.jsonl"
            )
            if self.current_trace_path.exists():
                self.current_trace_path.unlink()
        else:
            self.current_trace_path = None


def construct_agent(args: argparse.Namespace) -> Agent:
    llm_config = lm_config.construct_llm_config(args)

    agent: Agent
    if args.agent_type == "teacher_forcing":
        agent = TeacherForcingAgent()
    elif args.agent_type == "prompt":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        tokenizer = Tokenizer(args.provider, args.model)
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
        )
    else:
        raise NotImplementedError(
            f"agent type {args.agent_type} not implemented"
        )
    return agent
