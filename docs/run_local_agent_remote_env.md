# Running the Agent Locally with Remote WebArena Environments

This guide runs the steered model agent on your local machine (with a GPU) while using the WebArena Docker environments hosted on a remote EC2 instance.

## Prerequisites

- Local NVIDIA GPU with 14GB+ VRAM (e.g. RTX 4090, A6000, A100)
- CUDA and PyTorch installed locally
- Remote EC2 instance running WebArena Docker services (already set up)

## Your Remote Environment

| Field | Value |
|-------|-------|
| EC2 Instance | `i-0d485a079fc402fd8` |
| Elastic IP | `3.130.111.132` |
| Hostname | `ec2-3-130-111-132.us-east-2.compute.amazonaws.com` |
| SSH Key | `~/.ssh/webarena-key.pem` |
| AWS Profile | `chenyu` |

### Services running on the remote instance

| Service | URL |
|---------|-----|
| Shopping | http://ec2-3-130-111-132.us-east-2.compute.amazonaws.com:7770 |
| Shopping Admin | http://ec2-3-130-111-132.us-east-2.compute.amazonaws.com:7780 |
| Reddit/Forum | http://ec2-3-130-111-132.us-east-2.compute.amazonaws.com:9999 |
| Wikipedia | http://ec2-3-130-111-132.us-east-2.compute.amazonaws.com:8888 |
| Map | http://ec2-3-130-111-132.us-east-2.compute.amazonaws.com:3000 |
| GitLab | http://ec2-3-130-111-132.us-east-2.compute.amazonaws.com:8023 |

---

## Step 1: Install local dependencies

```bash
conda create -n webarena python=3.10 -y
conda activate webarena

cd /path/to/webarena
pip install -r requirements.txt
playwright install
pip install -e .

# Steered model deps
pip install torch transformers accelerate
```

## Step 2: Symlink agency_vectors

```bash
cd /path/to/webarena
ln -sf /path/to/agency_vectors ./agency_vectors
```

Verify:
```bash
ls agency_vectors/activation_steer.py
ls agency_vectors/persona_vectors/Qwen2.5-7B-Instruct/
```

## Step 3: Set environment variables

```bash
source setup_env.sh ec2-3-130-111-132.us-east-2.compute.amazonaws.com
```

Verify they are set:
```bash
echo $SHOPPING    # should print http://ec2-3-130-111-132.us-east-2.compute.amazonaws.com:7770
echo $GITLAB      # should print http://ec2-3-130-111-132.us-east-2.compute.amazonaws.com:8023
```

## Step 4: Generate auth cookies

This launches a local browser (via Playwright) that logs into each remote service and saves cookies:

```bash
mkdir -p .auth
python browser_env/auto_login.py
```

You should see output confirming cookies were saved for each site. If a site fails, verify it's accessible:
```bash
curl -s -o /dev/null -w "%{http_code}" http://ec2-3-130-111-132.us-east-2.compute.amazonaws.com:7770
```

## Step 5: Generate task config files

```bash
python scripts/generate_test_data.py
```

Verify:
```bash
ls config_files/*.json | head -5
# Should show: config_files/0.json, config_files/1.json, etc.
```

## Step 6: Run a smoke test (1 task, no steering)

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
  --test_end_idx 1 \
  --result_dir results/smoke_test
```

On first run, the model (~14GB) will be downloaded from HuggingFace. Subsequent runs use the cached model.

Check the result:
```bash
cat results/smoke_test/*.html  # shows PASS or FAIL
ls results/smoke_test/traces/  # shows browser trace
```

## Step 7: Run baseline (no steering, more tasks)

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

## Step 8: Run steered evaluation

Example with `goal_persistence` trait:

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

## Step 9: Compare results

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

## Managing the Remote EC2 Instance

### Start/stop to save costs

```bash
# Stop (saves money, keeps data)
aws ec2 stop-instances --profile chenyu --region us-east-2 --instance-ids i-0d485a079fc402fd8

# Start (Elastic IP stays attached)
aws ec2 start-instances --profile chenyu --region us-east-2 --instance-ids i-0d485a079fc402fd8
```

After restarting the instance, SSH in and restart Docker services:

```bash
ssh -i ~/.ssh/webarena-key.pem ubuntu@3.130.111.132
docker start gitlab shopping shopping_admin forum kiwix33
cd /home/ubuntu/openstreetmap-website/ && docker compose start
```

Wait ~1 minute, then re-run the hostname configuration commands from `docs/steered_model_evaluation.md` Step 3.

### SSH into the instance

```bash
ssh -i ~/.ssh/webarena-key.pem ubuntu@3.130.111.132
```

### Check instance status

```bash
aws ec2 describe-instances --profile chenyu --region us-east-2 \
  --instance-ids i-0d485a079fc402fd8 \
  --query 'Reservations[0].Instances[0].State.Name' --output text
```
