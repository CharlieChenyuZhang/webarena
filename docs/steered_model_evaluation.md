# Evaluating Steered Models on WebArena

This guide walks through running activation-steered models (from `agency_vectors`) on the WebArena benchmark.

## Overview

The `steered` provider integrates the `ActivationSteerer` from the `agency_vectors` project into WebArena's agent loop. It loads a base model (e.g. Qwen2.5-7B-Instruct) via `transformers`, optionally applies a persona steering vector during inference, and generates actions for the benchmark.

## Prerequisites

### 1. Install dependencies

```bash
# Create environment
conda create -n webarena python=3.10
conda activate webarena

# Install webarena
cd /path/to/webarena
pip install -r requirements.txt
playwright install

# Install additional deps for steered provider
pip install torch transformers accelerate
```

### 2. Symlink agency_vectors

The steered provider imports `ActivationSteerer` from the `agency_vectors` repo. Create a symlink in the webarena root:

```bash
ln -s /path/to/agency_vectors /path/to/webarena/agency_vectors
```

### 3. Set up WebArena environments

WebArena requires local web applications running in Docker. Follow the main WebArena README to start the environment containers (shopping, GitLab, Reddit, Wikipedia, etc.) and set the environment URL variables:

```bash
export SHOPPING="http://localhost:7770"
export SHOPPING_ADMIN="http://localhost:7780/admin"
export REDDIT="http://localhost:9999"
export GITLAB="http://localhost:8023"
export MAP="http://localhost:3000"
export WIKIPEDIA="http://localhost:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kev/Wikipedia:Village_Pump/Proposals/Paragraph_break"
export HOMEPAGE="http://localhost:4399"
```

### 4. Generate config files

```bash
python scripts/generate_test_data.py
```

This creates `config_files/*.json` with one JSON file per task.

## Running Evaluations

### Baseline (no steering)

Run the base Qwen model without any activation steering to establish a baseline:

```bash
python run.py \
  --provider steered \
  --model Qwen/Qwen2.5-7B-Instruct \
  --mode chat \
  --temperature 0.0 \
  --max_tokens 384 \
  --max_obs_length 1920 \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
  --test_start_idx 0 \
  --test_end_idx 50 \
  --result_dir results/baseline
```

### Steered run (with persona vector)

Apply a steering vector during inference. Available traits in `agency_vectors/persona_vectors/Qwen2.5-7B-Instruct/`:

- `goal_persistence` - persist through multi-step tasks
- `independence` - act decisively without seeking confirmation
- `evil` - (control/comparison)
- `rigidity` - (control/comparison, multiple versions)

Example with `goal_persistence` at coefficient 2.0:

```bash
python run.py \
  --provider steered \
  --model Qwen/Qwen2.5-7B-Instruct \
  --mode chat \
  --temperature 0.0 \
  --max_tokens 384 \
  --max_obs_length 1920 \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
  --vector_path agency_vectors/persona_vectors/Qwen2.5-7B-Instruct/goal_persistence_response_avg_diff.pt \
  --steering_layer 20 \
  --steering_coeff 2.0 \
  --steering_type response \
  --test_start_idx 0 \
  --test_end_idx 50 \
  --result_dir results/goal_persistence_coef2
```

### Coefficient sweep

To systematically compare steering strengths, run multiple coefficients:

```bash
for coef in 0.5 1.0 2.0 3.0 5.0; do
  python run.py \
    --provider steered \
    --model Qwen/Qwen2.5-7B-Instruct \
    --mode chat \
    --temperature 0.0 \
    --max_tokens 384 \
    --max_obs_length 1920 \
    --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
    --vector_path agency_vectors/persona_vectors/Qwen2.5-7B-Instruct/goal_persistence_response_avg_diff.pt \
    --steering_layer 20 \
    --steering_coeff $coef \
    --steering_type response \
    --test_start_idx 0 \
    --test_end_idx 50 \
    --result_dir "results/goal_persistence_coef${coef}"
done
```

## CLI Arguments Reference

| Argument | Description | Default |
|----------|-------------|---------|
| `--provider steered` | Use the local steered model provider | (required) |
| `--model` | HuggingFace model ID or local path | `gpt-3.5-turbo-0613` |
| `--mode` | `chat` or `completion` | `chat` |
| `--vector_path` | Path to `.pt` persona vector file | `None` (no steering) |
| `--steering_layer` | Transformer layer to steer (1-indexed) | `20` |
| `--steering_coeff` | Steering magnitude (0 = disabled) | `0.0` |
| `--steering_type` | Where to steer: `response`, `prompt`, `all` | `response` |
| `--temperature` | Sampling temperature (0 = greedy) | `1.0` |
| `--max_tokens` | Max tokens to generate per step | `384` |
| `--max_obs_length` | Truncate observations to this many tokens | `1920` |
| `--test_start_idx` | First task index to run | `0` |
| `--test_end_idx` | Last task index (exclusive) | `1000` |
| `--result_dir` | Where to save results | auto-generated |

## Architecture

```
run.py (CLI args)
  -> construct_llm_config()  [llms/lm_config.py]
       Builds LMConfig with steering params in gen_config
  -> construct_agent()       [agent/agent.py]
       Creates PromptAgent with the LMConfig
  -> agent.next_action()
       -> prompt_constructor.construct()  -> chat messages
       -> call_llm()                      [llms/utils.py]
            -> generate_from_steered_model()  [llms/providers/steered_utils.py]
                 Loads model (cached), applies ActivationSteerer, generates
       -> extract_action()                -> parsed browser action
  -> env.step(action)
       Browser executes the action
```

## Tips

- **Start small**: Run 5-10 tasks first (`--test_start_idx 0 --test_end_idx 10`) to verify everything works before scaling up.
- **GPU memory**: Qwen2.5-7B in bfloat16 needs ~14GB VRAM. The model is loaded once and cached.
- **Greedy decoding**: Use `--temperature 0.0` for reproducible results.
- **Prompt format**: `p_cot_id_actree_2s.json` uses chain-of-thought with accessibility tree observations — the recommended default.
- **Expected scores**: Vanilla 7B models typically score 0-6% on WebArena. Even a 2-3 percentage point improvement from steering is a meaningful signal.

## Comparing Results

Results are saved as HTML files in the result directory. To compute scores:

```bash
# Count passing tasks
grep -l "PASS" results/baseline/*.html | wc -l
grep -l "PASS" results/goal_persistence_coef2/*.html | wc -l
```

The average score is also logged at the end of each run.
