# Wandb Setup Guide for RAD Embeddings

This guide explains how to use Weights & Biases (Wandb) for experiment tracking and visualization in the RAD Embeddings project.

## Table of Contents

1. [Initial Setup](#initial-setup)
2. [Running Experiments](#running-experiments)
3. [Visualizing Results](#visualizing-results)
4. [Understanding the Modified Files](#understanding-the-modified-files)
5. [Troubleshooting](#troubleshooting)

---

## Initial Setup

### 1. Install Dependencies

On your server with 2 A100 GPUs, install the required dependencies:

```bash
# If using uv (recommended for this project)
uv sync

# Or if using pip
pip install -e .
```

This will install all dependencies including:
- `wandb>=0.16.0`
- `pandas>=2.0.0`
- `matplotlib>=3.7.0`
- All other existing dependencies

### 2. Set Up Wandb

First, create a Wandb account at [wandb.ai](https://wandb.ai) if you don't have one.

Then, log in on your server:

```bash
wandb login
```

This will prompt you for your API key. You can find your API key at: https://wandb.ai/authorize

Alternatively, you can set the API key as an environment variable:

```bash
export WANDB_API_KEY=your_api_key_here
```

### 3. Configure Wandb Projects

The code is set up to log to two Wandb projects:
- **`rad-embeddings-encoder`**: For encoder training logs
- **`rad-embeddings-policy`**: For policy training logs

These projects will be created automatically when you run your first experiment.

---

## Running Experiments

### Option 1: Run All 5 Seeds Automatically (Recommended)

The master script will handle everything for you:

```bash
cd rad_embeddings
python run_all_experiments.py
```

**For parallel execution on 2 GPUs:**

```bash
python run_all_experiments.py --parallel
```

This will:
1. Train encoder for seed 1, then train policy with that encoder
2. Train encoder for seed 2, then train policy with that encoder
3. ... and so on for all 5 seeds

When using `--parallel`, it will run 2 experiments simultaneously (one per GPU).

**Custom seeds:**

```bash
python run_all_experiments.py --seeds 1 2 3 4 5 6 7 8 9 10
```

### Option 2: Run Individual Experiments

If you prefer more control, you can run each step manually:

**Train an encoder (seed 1):**

```bash
cd rad_embeddings
python train_encoder.py 1
```

**Train policy using that encoder (seed 1):**

```bash
python train_token_env_policy.py --seed 1 --encoder-file storage/DFABisimEnv-v1-encoder_1.zip
```

**Repeat for other seeds (2, 3, 4, 5):**

```bash
# Seed 2
python train_encoder.py 2
python train_token_env_policy.py --seed 2 --encoder-file storage/DFABisimEnv-v1-encoder_2.zip

# Seed 3
python train_encoder.py 3
python train_token_env_policy.py --seed 3 --encoder-file storage/DFABisimEnv-v1-encoder_3.zip

# And so on...
```

### Option 3: Use GPU Assignment Manually

If you want to control which GPU each experiment runs on:

```bash
# Run seed 1 on GPU 0
CUDA_VISIBLE_DEVICES=0 python train_encoder.py 1
CUDA_VISIBLE_DEVICES=0 python train_token_env_policy.py --seed 1 --encoder-file storage/DFABisimEnv-v1-encoder_1.zip

# Run seed 2 on GPU 1 (in another terminal)
CUDA_VISIBLE_DEVICES=1 python train_encoder.py 2
CUDA_VISIBLE_DEVICES=1 python train_token_env_policy.py --seed 2 --encoder-file storage/DFABisimEnv-v1-encoder_2.zip
```

---

## Visualizing Results

### Real-time Monitoring

While experiments are running, you can monitor them in real-time:

1. Go to [wandb.ai](https://wandb.ai)
2. Navigate to your workspace
3. Open the `rad-embeddings-policy` project
4. You'll see live plots of all metrics including `rollout/ep_rew_disc_mean`

### Generate Learning Curves with Mean ± Std

After all 5 seeds have completed training, generate publication-quality learning curves:

```bash
cd rad_embeddings/scripts
python plot_wandb_learning_curve.py \
    --entity YOUR_WANDB_USERNAME \
    --project rad-embeddings-policy \
    --metric rollout/ep_rew_disc_mean \
    --output learning_curve.png
```

Replace `YOUR_WANDB_USERNAME` with your Wandb username or team name.

**Customize the plot:**

```bash
python plot_wandb_learning_curve.py \
    --entity YOUR_USERNAME \
    --project rad-embeddings-policy \
    --metric rollout/ep_rew_disc_mean \
    --title "Policy Learning Curve (5 Seeds)" \
    --output figures/policy_learning_curve.png
```

**Plot other metrics:**

```bash
# Plot episode reward mean
python plot_wandb_learning_curve.py \
    --entity YOUR_USERNAME \
    --project rad-embeddings-policy \
    --metric rollout/ep_rew_mean \
    --output figures/ep_rew_mean.png

# Plot episode length
python plot_wandb_learning_curve.py \
    --entity YOUR_USERNAME \
    --project rad-embeddings-policy \
    --metric rollout/ep_len_mean \
    --output figures/ep_len.png
```

### Export Data to CSV (Optional)

If you want to analyze the data offline or archive it:

```bash
cd rad_embeddings/scripts
python export_wandb_to_csv.py \
    --entity YOUR_USERNAME \
    --project rad-embeddings-policy \
    --output ../wandb_exports/
```

This will download all run data and save it as CSV files in `wandb_exports/`.

---

## Understanding the Modified Files

### 1. `train_encoder.py`

**Changes:**
- Added `import wandb`
- Added `wandb.init()` to initialize logging
- Added `wandb.finish()` to close the run

**What it logs:**
- All metrics from `LoggerCallback`
- Training configuration (seed, env_id, algorithm)
- Model checkpoints (via `WandbCallback`)

### 2. `train_token_env_policy.py`

**Changes:**
- Added `import wandb` and `from wandb.integration.sb3 import WandbCallback`
- Added `wandb.init()` with configuration
- Added `WandbCallback` to the callback list
- Added `wandb.finish()` at the end

**What it logs:**
- All metrics from `LoggerCallback` including `rollout/ep_rew_disc_mean`
- Training configuration (seed, encoder file, features extractor type)
- Model architecture (total parameters)
- Model checkpoints

### 3. `encoder.py`

**Changes:**
- Added `import wandb` and `from wandb.integration.sb3 import WandbCallback`
- Modified `train()` method to include `WandbCallback` in the callbacks list

**What it logs:**
- Same as train_encoder.py, but integrated into the Encoder class

### 4. `utils/sb3/logger_callback.py`

**Changes:**
- Added additional metrics: `ep_len_mean`, `ep_rew_mean`, `ep_disc_rew_std`
- Added safety check for empty episode lists
- Metrics are automatically picked up by WandbCallback

**What it logs:**
- `rollout/ep_len_min`, `rollout/ep_len_max`, `rollout/ep_len_mean`, `rollout/ep_len_std`
- `rollout/ep_rew_min`, `rollout/ep_rew_max`, `rollout/ep_rew_mean`, `rollout/ep_rew_std`
- `rollout/ep_rew_disc_mean`, `rollout/ep_rew_disc_std` (key metric!)

### 5. New Files Created

**`run_all_experiments.py`**
- Master script to orchestrate all experiments
- Supports sequential or parallel execution
- Handles GPU assignment automatically in parallel mode

**`scripts/plot_wandb_learning_curve.py`**
- Downloads data from Wandb
- Aggregates across seeds
- Plots mean with shaded standard deviation region

**`scripts/export_wandb_to_csv.py`**
- Exports Wandb data to CSV files
- Useful for offline analysis or archiving

---

## Logged Metrics

### Key Metrics (from LoggerCallback)

| Metric | Description |
|--------|-------------|
| `rollout/ep_rew_disc_mean` | **Main metric**: Mean discounted episode reward |
| `rollout/ep_rew_disc_std` | Standard deviation of discounted rewards |
| `rollout/ep_rew_mean` | Mean episode reward (undiscounted) |
| `rollout/ep_rew_std` | Standard deviation of episode rewards |
| `rollout/ep_rew_min` | Minimum episode reward |
| `rollout/ep_rew_max` | Maximum episode reward |
| `rollout/ep_len_mean` | Mean episode length |
| `rollout/ep_len_std` | Standard deviation of episode lengths |
| `rollout/ep_len_min` | Minimum episode length |
| `rollout/ep_len_max` | Maximum episode length |

### Standard SB3 Metrics (automatically logged)

- `train/learning_rate`: Current learning rate
- `train/loss`: Total loss
- `train/policy_loss`: Policy loss
- `train/value_loss`: Value function loss
- `train/entropy_loss`: Entropy loss
- `train/explained_variance`: Explained variance of value function
- `rollout/ep_rew_mean`: Mean episode reward (SB3 default)
- `rollout/ep_len_mean`: Mean episode length (SB3 default)
- `time/fps`: Frames per second
- `time/total_timesteps`: Total timesteps trained

---

## Troubleshooting

### Issue: "wandb: ERROR Error uploading"

**Solution:** This is usually a network issue. Wandb will retry automatically. If it persists, check your internet connection or firewall settings.

### Issue: "No module named 'wandb'"

**Solution:** Install wandb:
```bash
pip install wandb
# or
uv sync
```

### Issue: "wandb.errors.UsageError: api_key not configured"

**Solution:** Run `wandb login` and enter your API key.

### Issue: Experiments are slow

**Solution:** 
- Use `--parallel` mode to run 2 experiments simultaneously
- Check GPU utilization with `nvidia-smi`
- Consider reducing `n_envs` if memory is an issue

### Issue: "Metric 'rollout/ep_rew_disc_mean' not found"

**Solution:** This metric is logged by `LoggerCallback`. Make sure:
1. The callback is included in the training
2. At least one episode has completed
3. You're looking at the correct project (`rad-embeddings-policy`, not `rad-embeddings-encoder`)

### Issue: Plots show fewer than 5 seeds

**Solution:** 
- Check that all 5 experiments completed successfully
- Verify all runs are in the same `group` (should be `policy_training`)
- Check the Wandb web interface to see which runs are available

### Issue: Want to disable Wandb temporarily

**Solution:** Set the environment variable:
```bash
export WANDB_MODE=disabled
```

Then run your experiments as usual. Nothing will be logged to Wandb.

---

## Advanced Usage

### Comparing Different Feature Extractors

You can compare encoder-based vs LLM-based features:

```bash
# Run with encoder (what you're doing)
python train_token_env_policy.py --seed 1 --encoder-file storage/DFABisimEnv-v1-encoder_1.zip

# Run with LLM features
python train_token_env_policy.py --seed 1 --llm

# Run without encoder (baseline)
python train_token_env_policy.py --seed 1
```

All will be logged to Wandb with appropriate tags for easy comparison.

### Custom Wandb Configuration

You can modify the `wandb.init()` calls in the training scripts to:
- Change project names
- Add custom tags
- Add notes or descriptions
- Change run names

Example:

```python
wandb.init(
    project="my-custom-project",
    name=f"my_experiment_seed_{SEED}",
    config={...},
    group="my_group",
    tags=["tag1", "tag2"],
    notes="This is a test run",
)
```

### Sweeps (Hyperparameter Tuning)

Wandb supports automated hyperparameter sweeps. See [Wandb Sweeps documentation](https://docs.wandb.ai/guides/sweeps) for details.

---

## Summary of Steps

1. **Setup:**
   ```bash
   uv sync  # Install dependencies
   wandb login  # Authenticate
   ```

2. **Run Experiments:**
   ```bash
   cd rad_embeddings
   python run_all_experiments.py --parallel  # Run all 5 seeds
   ```

3. **Monitor:** Check [wandb.ai](https://wandb.ai) for real-time progress

4. **Visualize:**
   ```bash
   cd scripts
   python plot_wandb_learning_curve.py --entity YOUR_USERNAME --project rad-embeddings-policy
   ```

5. **Done!** You now have:
   - All experiments logged to Wandb
   - Learning curves with mean ± std across 5 seeds
   - Easy comparison and analysis

---

## Questions or Issues?

If you encounter any issues not covered here, check:
- [Wandb Documentation](https://docs.wandb.ai)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- The code comments in the modified files

Happy experimenting! 🚀

