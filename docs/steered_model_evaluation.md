# Evaluating Steered Models on WebArena

End-to-end guide for running activation-steered models (from `agency_vectors`) on the WebArena benchmark to measure how steering affects web agent task performance.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Set Up WebArena Docker Environments](#3-set-up-webarena-docker-environments)
4. [Install Python Dependencies](#4-install-python-dependencies)
5. [Set Environment Variables](#5-set-environment-variables)
6. [Generate Auth Cookies and Config Files](#6-generate-auth-cookies-and-config-files)
7. [Link the agency_vectors Repo](#7-link-the-agency_vectors-repo)
8. [Run Baseline (No Steering)](#8-run-baseline-no-steering)
9. [Run Steered Evaluation](#9-run-steered-evaluation)
10. [Run a Coefficient Sweep](#10-run-a-coefficient-sweep)
11. [Compare Results](#11-compare-results)
12. [CLI Arguments Reference](#12-cli-arguments-reference)
13. [Architecture](#13-architecture)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Overview

The `steered` provider plugs into WebArena's existing agent loop. It loads a base model (e.g. `Qwen/Qwen2.5-7B-Instruct`) via HuggingFace `transformers`, optionally applies an activation steering vector from `agency_vectors` during inference, and generates browser actions for each benchmark task. The benchmark evaluates whether the agent completed the task correctly.

**What you're measuring:** the delta in task success rate between the base model and the steered model. Even a small improvement (2-3 percentage points) is meaningful given that vanilla 7B models score 0-6% on WebArena.

---

## 2. Prerequisites

- **GPU**: Qwen2.5-7B in bfloat16 requires ~14GB VRAM. A single A100/A6000/4090 works.
- **Disk**: ~50GB for the model weights (downloaded on first run), plus ~200GB+ for Docker images.
- **Docker**: Installed and running.
- **Conda**: For Python environment management.
- **Repos cloned**:
  - This repo (`webarena`)
  - `agency_vectors` (contains steering vectors and `ActivationSteerer`)

---

## 3. Set Up WebArena Docker Environments

WebArena requires 6 web applications running locally in Docker containers. The agent interacts with these during evaluation.

### Option A: Use the Pre-Built AWS AMI (Recommended)

This is the fastest path — everything is pre-installed.

1. **Launch an EC2 instance** in `us-east-2` (Ohio):
   - AMI ID: `ami-08a862bf98e3bd7aa` (name: `webarena-with-configurable-map-backend`)
   - Instance type: `t3a.xlarge`
   - Root volume: 1000GB EBS
   - Security group: allow inbound on ports `22, 80, 3000, 7770, 7780, 8023, 8888, 9999`

2. **Assign an Elastic IP** and note the hostname (e.g. `ec2-xx-xx-xx-xx.us-east-2.compute.amazonaws.com`).

3. **SSH in and start the services:**
   ```bash
   docker start gitlab shopping shopping_admin forum kiwix33
   cd /home/ubuntu/openstreetmap-website/ && docker compose start
   ```
   Wait ~1 minute for all services to boot.

4. **Configure hostnames** (replace `<your-server-hostname>` everywhere below):
   ```bash
   HOSTNAME="<your-server-hostname>"

   # Shopping
   docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://${HOSTNAME}:7770"
   docker exec shopping mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value='http://${HOSTNAME}:7770/' WHERE path = 'web/secure/base_url';"
   docker exec shopping /var/www/magento2/bin/magento cache:flush

   # Shopping Admin
   docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://${HOSTNAME}:7780"
   docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value='http://${HOSTNAME}:7780/' WHERE path = 'web/secure/base_url';"
   docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
   docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
   docker exec shopping_admin /var/www/magento2/bin/magento cache:flush

   # GitLab
   docker exec gitlab update-permissions
   docker exec gitlab sed -i "s|^external_url.*|external_url 'http://${HOSTNAME}:8023'|" /etc/gitlab/gitlab.rb
   docker exec gitlab gitlab-ctl reconfigure
   ```

5. **Verify all services return HTTP 200:**
   ```bash
   for port in 7770 7780 9999 8888 3000 8023; do
     curl -s -o /dev/null -w "Port ${port}: %{http_code}\n" http://${HOSTNAME}:${port}
   done
   ```

### Option B: Manual Docker Setup

If you can't use AWS, download and run each Docker image individually. See `environment_docker/README.md` for detailed instructions per service:

| Service | Port | Image |
|---------|------|-------|
| Shopping | 7770 | `shopping_final_0712` |
| Shopping Admin | 7780 | `shopping_admin_final_0719` |
| Reddit Forum | 9999 | `postmill-populated-exposed-withimg` |
| GitLab | 8023 | `gitlab-populated-final-port8023` |
| Wikipedia | 8888 | `kiwix-serve` + `.zim` file |
| Map | 3000 | OpenStreetMap docker-compose |

Docker images can be downloaded from:
- http://metis.lti.cs.cmu.edu/webarena-images/
- https://archive.org/download/webarena-env-shopping-image (and similar for other images)

---

## 4. Install Python Dependencies

```bash
# Create and activate environment
conda create -n webarena python=3.10 -y
conda activate webarena

# Install webarena
cd /path/to/webarena
pip install -r requirements.txt
playwright install
pip install -e .

# Install additional deps for steered provider
pip install torch transformers accelerate
```

---

## 5. Set Environment Variables

Use the provided helper script:

```bash
source setup_env.sh <your-server-hostname>
# Example:
source setup_env.sh ec2-xx-xx-xx-xx.us-east-2.compute.amazonaws.com
```

Or set them manually:

```bash
export SHOPPING="http://<your-server-hostname>:7770"
export SHOPPING_ADMIN="http://<your-server-hostname>:7780/admin"
export REDDIT="http://<your-server-hostname>:9999"
export GITLAB="http://<your-server-hostname>:8023"
export MAP="http://<your-server-hostname>:3000"
export WIKIPEDIA="http://<your-server-hostname>:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="http://<your-server-hostname>:4399"
```

> **Note:** If you're not using the Homepage service, you can set `export HOMEPAGE="PASS"` as a placeholder.

---

## 6. Generate Auth Cookies and Config Files

```bash
# Generate browser auth cookies for auto-login
mkdir -p .auth
python browser_env/auto_login.py

# Generate per-task config files (one JSON per task, 812 total)
python scripts/generate_test_data.py
```

Verify config files were created:
```bash
ls config_files/*.json | wc -l
# Should output: 812 (or similar)
```

---

## 7. Link the agency_vectors Repo

The steered provider needs to import `ActivationSteerer` from your `agency_vectors` repo:

```bash
cd /path/to/webarena
ln -s /path/to/agency_vectors ./agency_vectors
```

Verify the steering vectors are accessible:
```bash
ls agency_vectors/persona_vectors/Qwen2.5-7B-Instruct/
```

Available traits (using `*_response_avg_diff.pt` vectors):
- `goal_persistence` — persist through multi-step tasks
- `independence` — act decisively without seeking confirmation
- `evil` — negative control
- `rigidity` (v1-v4) — stick to initial approach

---

## 8. Run Baseline (No Steering)

Start with a small subset to verify the setup works:

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
  --test_end_idx 5 \
  --result_dir results/baseline_smoke_test
```

If this runs successfully (the model loads, generates actions, and the browser executes them), scale up:

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
  --test_end_idx 812 \
  --result_dir results/baseline
```

The average score is logged at the end of the run.

---

## 9. Run Steered Evaluation

Apply a persona steering vector during inference. Example with `goal_persistence` at coefficient 2.0:

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
  --test_end_idx 812 \
  --result_dir results/goal_persistence_coef2
```

### Key steering parameters

| Parameter | What it controls | Recommended |
|-----------|-----------------|-------------|
| `--vector_path` | Which trait to steer toward | Use `*_response_avg_diff.pt` variants |
| `--steering_layer` | Which transformer layer to hook (1-indexed) | `20` for Qwen2.5-7B |
| `--steering_coeff` | How strongly to steer (0 = off) | Start with `1.0`-`3.0` |
| `--steering_type` | Which tokens get steered | `response` (recommended) |

---

## 10. Run a Coefficient Sweep

Systematically compare steering strengths:

```bash
TRAIT="goal_persistence"
VECTOR="agency_vectors/persona_vectors/Qwen2.5-7B-Instruct/${TRAIT}_response_avg_diff.pt"

for coef in 0.5 1.0 2.0 3.0 5.0; do
  python run.py \
    --provider steered \
    --model Qwen/Qwen2.5-7B-Instruct \
    --mode chat \
    --temperature 0.0 \
    --max_tokens 384 \
    --max_obs_length 1920 \
    --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
    --vector_path "$VECTOR" \
    --steering_layer 20 \
    --steering_coeff $coef \
    --steering_type response \
    --test_start_idx 0 \
    --test_end_idx 812 \
    --result_dir "results/${TRAIT}_coef${coef}"
done
```

To sweep multiple traits:

```bash
for trait in goal_persistence independence; do
  for coef in 1.0 2.0 3.0; do
    python run.py \
      --provider steered \
      --model Qwen/Qwen2.5-7B-Instruct \
      --mode chat \
      --temperature 0.0 \
      --max_tokens 384 \
      --max_obs_length 1920 \
      --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
      --vector_path "agency_vectors/persona_vectors/Qwen2.5-7B-Instruct/${trait}_response_avg_diff.pt" \
      --steering_layer 20 \
      --steering_coeff $coef \
      --steering_type response \
      --test_start_idx 0 \
      --test_end_idx 812 \
      --result_dir "results/${trait}_coef${coef}"
  done
done
```

---

## 11. Compare Results

The average score is printed at the end of each run. You can also count passing tasks from the result HTML files:

```bash
echo "=== Results Summary ==="
for dir in results/*/; do
  total=$(ls "$dir"/*.html 2>/dev/null | wc -l)
  pass=$(grep -rl "PASS" "$dir" 2>/dev/null | wc -l)
  if [ "$total" -gt 0 ]; then
    pct=$(echo "scale=1; $pass * 100 / $total" | bc)
    echo "$(basename $dir): ${pass}/${total} (${pct}%)"
  fi
done
```

---

## 12. CLI Arguments Reference

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
| `--top_p` | Nucleus sampling threshold | `0.9` |
| `--max_tokens` | Max tokens to generate per agent step | `384` |
| `--max_obs_length` | Truncate observations to this many tokens | `1920` |
| `--max_steps` | Max browser actions per task | `30` |
| `--max_retry` | Retries on action parse failure | `1` |
| `--test_start_idx` | First task index to run | `0` |
| `--test_end_idx` | Last task index (exclusive) | `1000` |
| `--result_dir` | Where to save results | auto-generated |

---

## 13. Architecture

```
run.py (CLI args)
  |
  |-- construct_llm_config()           [llms/lm_config.py]
  |     Packs steering params into LMConfig.gen_config
  |
  |-- construct_agent()                [agent/agent.py]
  |     Creates PromptAgent with the LMConfig + tokenizer
  |
  |-- test() loop: for each task config
        |
        |-- env.reset()                Start browser at task URL
        |
        |-- while not done:
        |     |
        |     |-- agent.next_action()
        |     |     |-- prompt_constructor.construct()
        |     |     |     Formats observation + history into chat messages
        |     |     |
        |     |     |-- call_llm()     [llms/utils.py]
        |     |     |     |-- generate_from_steered_model()
        |     |     |           [llms/providers/steered_utils.py]
        |     |     |           Loads model (cached), applies ActivationSteerer,
        |     |     |           runs model.generate(), returns text
        |     |     |
        |     |     |-- extract_action()
        |     |           Parses LLM output into browser action
        |     |
        |     |-- env.step(action)     Execute action in browser
        |
        |-- evaluator(trajectory)      Score: PASS (1) or FAIL (0)
```

---

## 14. Troubleshooting

### Model doesn't load / CUDA out of memory
- Qwen2.5-7B in bfloat16 needs ~14GB VRAM. Check with `nvidia-smi`.
- If you have <14GB, try a smaller model or use CPU (very slow).

### `AssertionError` about environment URLs
- All 7 environment variables must be set. Check with `env | grep -E 'SHOPPING|REDDIT|GITLAB|WIKIPEDIA|MAP|HOMEPAGE'`.

### GitLab returns 502
```bash
docker exec gitlab rm -f /var/opt/gitlab/postgresql/data/postmaster.pid
docker exec -u gitlab-psql gitlab /opt/gitlab/embedded/bin/pg_resetwal -f /var/opt/gitlab/postgresql/data
docker exec gitlab gitlab-ctl restart
```

### Services not accessible externally (AWS)
```bash
for port in 7770 7780 3000 8888 9999 8023; do
  sudo iptables -t nat -A PREROUTING -p tcp --dport $port -j REDIRECT --to-port $port
done
```

### `ImportError: cannot import name 'ActivationSteerer'`
- Check the symlink: `ls -la agency_vectors/activation_steer.py`
- If broken, recreate: `ln -sf /path/to/agency_vectors ./agency_vectors`

### Resetting environments after a full run
After all 812 tasks, the web apps may have been modified by the agent. Reset:
```bash
docker stop shopping shopping_admin forum gitlab
docker rm shopping shopping_admin forum gitlab
docker run --name shopping -p 7770:80 -d shopping_final_0712
docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719
docker run --name forum -p 9999:80 -d postmill-populated-exposed-withimg
docker run --name gitlab -d -p 8023:8023 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start
# Then re-run the hostname configuration commands from Step 3
```

### Prompt JSON files not found
The prompt JSONs are auto-generated from Python files. If missing:
```bash
cd agent/prompts/raw && python p_cot_id_actree_2s.py
```
Or they are generated automatically when `run.py` calls `prepare()`.
