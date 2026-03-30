"""Script to run end-to-end evaluation on the benchmark"""
import argparse
import glob
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import openai

from agent import (
    Agent,
    PromptAgent,
    TeacherForcingAgent,
    construct_agent,
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _write_task_timing(
    result_dir: str,
    *,
    task_id: int | str | None,
    task_config_file: str | None,
    intent: str | None,
    run_label: str,
    model_variant: str,
    condition_name: str,
    started_at: str,
    ended_at: str,
    duration_seconds: float,
    status: str,
    score: float | None,
    provider: str,
    model: str,
    model_target: str | None,
    steering_vector_path: str | None,
    steering_layer: int | None,
    steering_coeff: float | None,
    steering_type: str | None,
) -> None:
    if task_id is None:
        return

    timing_dir = Path(result_dir) / "task_timings"
    timing_dir.mkdir(parents=True, exist_ok=True)
    timing_path = timing_dir / f"task_{task_id}.json"
    payload = {
        "task_id": task_id,
        "task_config_file": task_config_file,
        "intent": intent,
        "run_label": run_label,
        "model_variant": model_variant,
        "condition_name": condition_name,
        "provider": provider,
        "model": model,
        "model_target": model_target,
        "steering_vector_path": steering_vector_path,
        "steering_layer": steering_layer,
        "steering_coeff": steering_coeff,
        "steering_type": steering_type,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds,
        "status": status,
        "score": score,
    }
    with timing_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When concesecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When concesecutive repeating action exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )
    parser.add_argument(
        "--model_endpoint",
        help="huggingface model endpoint",
        type=str,
        default="",
    )
    parser.add_argument(
        "--run_label",
        help="Human-readable run label stamped into config and model traces",
        type=str,
        default="",
    )
    parser.add_argument(
        "--model_variant",
        help="Traceable model variant label such as baseline or steered",
        type=str,
        default="",
    )
    parser.add_argument(
        "--condition_name",
        help="Experiment condition label such as goal_persistence_high",
        type=str,
        default="",
    )

    # steered model config
    parser.add_argument(
        "--vector_path",
        help="Path to .pt persona steering vector file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--steering_layer",
        help="Transformer layer to apply steering (1-indexed)",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--steering_coeff",
        help="Steering coefficient (0 = no steering)",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--steering_type",
        help="Where to apply steering: response, prompt, or all",
        type=str,
        choices=["response", "prompt", "all"],
        default="response",
    )

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=1000)
    parser.add_argument(
        "--task_ids_file",
        type=str,
        default="",
        help="Optional JSON file containing a frozen task subset. Each entry may be a task object or task_id.",
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type != "accessibility_tree"
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    if not args.model_variant:
        use_local_steering = (
            args.provider == "steered"
            or (args.vector_path is not None and args.steering_coeff != 0.0)
        )
        args.model_variant = "steered" if use_local_steering else "baseline"

    if not args.condition_name:
        args.condition_name = args.model_variant

    if not args.run_label:
        args.run_label = args.condition_name

    return args


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to early stop"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""


def test(
    args: argparse.Namespace,
    agent: Agent | PromptAgent | TeacherForcingAgent,
    config_file_list: list[str],
) -> None:
    scores = []
    max_steps = args.max_steps
    total_tasks = len(config_file_list)
    run_started_monotonic = time.time()

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
    )

    for task_idx, config_file in enumerate(config_file_list, start=1):
        task_started_monotonic = time.time()
        task_started_at = _iso_utc_now()
        task_id: int | str | None = None
        intent: str | None = None
        task_config_file_for_log: str | None = config_file
        score: float | None = None
        task_status = "started"
        render_helper = None
        try:
            progress_pct = (task_idx / total_tasks) * 100 if total_tasks else 100.0
            logger.info(
                f"[Task Progress] {task_idx}/{total_tasks} "
                f"({progress_pct:.1f}%) run_label={args.run_label}"
            )
            render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )

            # get intent
            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                # automatically login
                if _c["storage_state"]:
                    cookie_file_name = os.path.basename(_c["storage_state"])
                    comb = get_site_comb_from_filepath(cookie_file_name)
                    temp_dir = tempfile.mkdtemp()
                    # subprocess to renew the cookie
                    subprocess.run(
                        [
                            sys.executable,
                            "browser_env/auto_login.py",
                            "--auth_folder",
                            temp_dir,
                            "--site_list",
                            *comb,
                        ]
                    )
                    _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                    assert os.path.exists(_c["storage_state"])
                    # update the config file
                    config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                    with open(config_file, "w") as f:
                        json.dump(_c, f)
                    task_config_file_for_log = config_file

            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            agent.reset(
                config_file,
                result_dir=args.result_dir,
                run_metadata={
                    "run_label": args.run_label,
                    "model_variant": args.model_variant,
                    "condition_name": args.condition_name,
                    "model_target": args.model_endpoint or None,
                    "steering_vector_path": args.vector_path,
                    "steering_layer": args.steering_layer,
                    "steering_coeff": args.steering_coeff,
                    "steering_type": args.steering_type,
                },
            )
            trajectory: Trajectory = []
            obs, info = env.reset(options={"config_file": config_file})
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)

            meta_data = {"action_history": ["None"]}
            while True:
                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )

                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    try:
                        action = agent.next_action(
                            trajectory, intent, meta_data=meta_data
                        )
                    except ValueError as e:
                        # get the error message
                        action = create_stop_action(f"ERROR: {str(e)}")

                trajectory.append(action)

                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=args.action_set_tag,
                    prompt_constructor=agent.prompt_constructor
                    if isinstance(agent, PromptAgent)
                    else None,
                )
                render_helper.render(
                    action, state_info, meta_data, args.render_screenshot
                )
                meta_data["action_history"].append(action_str)

                if action["action_type"] == ActionTypes.STOP:
                    break

                obs, _, terminated, _, info = env.step(action)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    break

            evaluator = evaluator_router(config_file)
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )

            scores.append(score)

            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
                task_status = "pass"
            else:
                logger.info(f"[Result] (FAIL) {config_file}")
                task_status = "fail"

            if args.save_trace_enabled:
                env.save_trace(
                    Path(args.result_dir) / "traces" / f"{task_id}.zip"
                )

        except KeyboardInterrupt:
            task_status = "interrupted"
            raise
        except openai.error.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
            task_status = "openai_error"
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            # write to error file
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())  # write stack trace to file
            task_status = "error"
        finally:
            task_ended_at = _iso_utc_now()
            task_duration_seconds = time.time() - task_started_monotonic
            _write_task_timing(
                args.result_dir,
                task_id=task_id,
                task_config_file=task_config_file_for_log,
                intent=intent,
                run_label=args.run_label,
                model_variant=args.model_variant,
                condition_name=args.condition_name,
                started_at=task_started_at,
                ended_at=task_ended_at,
                duration_seconds=task_duration_seconds,
                status=task_status,
                score=score,
                provider=args.provider,
                model=args.model,
                model_target=args.model_endpoint or None,
                steering_vector_path=args.vector_path,
                steering_layer=args.steering_layer,
                steering_coeff=args.steering_coeff,
                steering_type=args.steering_type,
            )
            completed_tasks = task_idx
            remaining_tasks = max(total_tasks - completed_tasks, 0)
            elapsed_run_seconds = time.time() - run_started_monotonic
            avg_task_seconds = (
                elapsed_run_seconds / completed_tasks if completed_tasks else 0.0
            )
            eta_seconds = remaining_tasks * avg_task_seconds
            logger.info(
                "[Task Timing] "
                f"task_id={task_id} "
                f"duration={_format_duration(task_duration_seconds)} "
                f"elapsed={_format_duration(elapsed_run_seconds)} "
                f"remaining={remaining_tasks} "
                f"eta={_format_duration(eta_seconds)}"
            )
            if render_helper is not None:
                render_helper.close()

    env.close()
    if scores:
        logger.info(f"Average score: {sum(scores) / len(scores)}")
    else:
        logger.info("Average score: no completed tasks")


def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)
    if not (Path(result_dir) / "task_timings").exists():
        (Path(result_dir) / "task_timings").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


def resolve_config_file_list(args: argparse.Namespace) -> list[str]:
    if args.task_ids_file:
        with open(args.task_ids_file, "r", encoding="utf-8") as f:
            task_items = json.load(f)
        task_ids: list[str] = []
        for item in task_items:
            if isinstance(item, dict):
                task_id = item.get("task_id")
            else:
                task_id = item
            if task_id is None:
                raise ValueError(
                    f"Task entry in {args.task_ids_file} is missing task_id: {item!r}"
                )
            task_ids.append(str(task_id))
        return [f"config_files/{task_id}.json" for task_id in task_ids]

    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        test_file_list.append(f"config_files/{i}.json")
    return test_file_list


if __name__ == "__main__":
    args = config()
    args.sleep_after_execution = 2.0
    prepare(args)

    test_file_list = resolve_config_file_list(args)
    if "debug" not in args.result_dir:
        test_file_list = get_unfinished(test_file_list, args.result_dir)

    if len(test_file_list) == 0:
        logger.info("No task left to run")
    else:
        print(f"Total {len(test_file_list)} tasks left")
        args.render = False
        args.render_screenshot = True
        args.save_trace_enabled = True

        args.current_viewport_only = True
        dump_config(args)

        agent = construct_agent(args)
        test(args, agent, test_file_list)
