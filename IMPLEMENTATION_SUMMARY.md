# Wandb Integration - Implementation Summary

This document summarizes all changes made to integrate Wandb into your RAD Embeddings project.

---

## ✅ Completed Changes

### 1. Modified Existing Files

#### `train_encoder.py`
- **Added**: Wandb initialization with project `rad-embeddings-encoder`
- **Added**: Seed tracking and group organization
- **Added**: `wandb.finish()` at end
- **Logs to**: `rad-embeddings-encoder` project

#### `train_token_env_policy.py`
- **Added**: Wandb initialization with project `rad-embeddings-policy`
- **Added**: WandbCallback integration
- **Added**: Experiment name based on features (encoder/llm/no_encoder)
- **Added**: Configuration tracking (seed, encoder file, etc.)
- **Added**: `wandb.finish()` at end
- **Logs to**: `rad-embeddings-policy` project

#### `encoder.py`
- **Added**: Wandb imports
- **Modified**: `Encoder.train()` method to include WandbCallback
- **Added**: Model checkpoint saving to wandb

#### `utils/sb3/logger_callback.py`
- **Enhanced**: Added more metrics (`ep_len_mean`, `ep_rew_mean`, etc.)
- **Added**: Safety check for empty episode lists
- **Fixed**: Added `ep_disc_rew_std` metric
- **Compatible**: Works with both Tensorboard and Wandb

#### `utils/sb3/token_env_llm_features_extractor.py`
- **Fixed**: Removed debug print statements and input() calls

#### `pyproject.toml`
- **Added**: `wandb>=0.16.0`
- **Added**: `pandas>=2.0.0`
- **Added**: `matplotlib>=3.7.0`

---

### 2. New Files Created

#### `run_all_experiments.py`
**Purpose**: Master orchestration script for running all 5-seed experiments

**Features**:
- Sequential or parallel execution
- Automatic GPU assignment in parallel mode
- Progress tracking and error handling
- Runs both encoder and policy training for each seed

**Usage**:
```bash
python run_all_experiments.py  # Sequential
python run_all_experiments.py --parallel  # Parallel (2 GPUs)
python run_all_experiments.py --seeds 1 2 3  # Custom seeds
```

#### `scripts/plot_wandb_learning_curve.py`
**Purpose**: Generate publication-quality learning curves with mean ± std

**Features**:
- Downloads data from Wandb API
- Aggregates across multiple seeds
- Interpolates to common time grid
- Plots mean with shaded std deviation
- Saves high-resolution figures

**Usage**:
```bash
python plot_wandb_learning_curve.py \
    --entity YOUR_USERNAME \
    --project rad-embeddings-policy \
    --metric rollout/ep_rew_disc_mean \
    --output learning_curve.png
```

#### `scripts/export_wandb_to_csv.py`
**Purpose**: Export Wandb data to CSV for offline analysis

**Features**:
- Downloads all runs in a group
- Saves metrics as CSV files
- Saves run configurations as JSON
- Useful for archiving and custom analysis

**Usage**:
```bash
python export_wandb_to_csv.py \
    --entity YOUR_USERNAME \
    --project rad-embeddings-policy \
    --output wandb_exports/
```

#### `WANDB_SETUP_GUIDE.md`
**Purpose**: Comprehensive documentation

**Contents**:
- Initial setup instructions
- Detailed usage guide
- Troubleshooting section
- Advanced usage examples

#### `QUICK_START.md`
**Purpose**: Quick reference for common tasks

**Contents**:
- TL;DR command sequence
- File structure overview
- Expected runtimes
- Key metrics explanation

---

## 📊 Logged Metrics

### Custom Metrics (from LoggerCallback)

All logged under `rollout/` namespace:

| Metric | Description |
|--------|-------------|
| `ep_rew_disc_mean` | **PRIMARY**: Mean discounted episode reward |
| `ep_rew_disc_std` | Std deviation of discounted rewards |
| `ep_rew_mean` | Mean undiscounted episode reward |
| `ep_rew_std` | Std deviation of episode rewards |
| `ep_rew_min` | Minimum episode reward |
| `ep_rew_max` | Maximum episode reward |
| `ep_len_mean` | Mean episode length |
| `ep_len_std` | Std deviation of episode lengths |
| `ep_len_min` | Minimum episode length |
| `ep_len_max` | Maximum episode length |

### Standard SB3 Metrics

Automatically logged by Stable-Baselines3:

**Training metrics** (`train/` namespace):
- `learning_rate`, `loss`, `policy_loss`, `value_loss`
- `entropy_loss`, `explained_variance`

**Timing metrics** (`time/` namespace):
- `fps`, `total_timesteps`

---

## 🔄 Workflow

### Complete 5-Seed Experiment Workflow

```
For each seed (1-5):
  ┌─────────────────────────┐
  │ 1. Train Encoder        │
  │    - 1M steps           │
  │    - Logs to Wandb      │
  │    - Saves to storage/  │
  └──────────┬──────────────┘
             │
             ▼
  ┌─────────────────────────┐
  │ 2. Train Policy         │
  │    - Uses saved encoder │
  │    - 1M steps           │
  │    - Logs to Wandb      │
  │    - Saves to exps_*/   │
  └─────────────────────────┘

After all seeds complete:
  ┌─────────────────────────┐
  │ 3. Generate Plot        │
  │    - Download from API  │
  │    - Compute mean & std │
  │    - Save figure        │
  └─────────────────────────┘
```

---

## 🚀 Usage Instructions

### Step 1: Initial Setup (One-Time)

```bash
# Install dependencies
uv sync  # or: pip install -e .

# Login to Wandb
wandb login
```

### Step 2: Run Experiments

**Option A: Automatic (Recommended)**
```bash
cd rad_embeddings
python run_all_experiments.py --parallel
```

**Option B: Manual**
```bash
# For each seed 1-5:
python train_encoder.py {SEED}
python train_token_env_policy.py --seed {SEED} --encoder-file storage/DFABisimEnv-v1-encoder_{SEED}.zip
```

### Step 3: Generate Learning Curves

```bash
cd scripts
python plot_wandb_learning_curve.py \
    --entity YOUR_WANDB_USERNAME \
    --project rad-embeddings-policy \
    --metric rollout/ep_rew_disc_mean \
    --output learning_curve.png
```

---

## 📁 Output Structure

```
rad_embeddings/
│
├── storage/                           # Encoder outputs
│   ├── DFABisimEnv-v1-encoder_1.zip
│   ├── DFABisimEnv-v1-encoder_2.zip
│   ├── ...
│   ├── DFABisimEnv-v1-encoder_5.zip
│   └── wandb_models/                  # Wandb checkpoints (optional)
│
├── exps_no_embed/                     # Policy outputs
│   ├── token_env_reach_avoid_policy_seed1.zip
│   ├── token_env_reach_avoid_policy_seed2.zip
│   ├── ...
│   ├── token_env_reach_avoid_policy_seed5.zip
│   ├── wandb_models/                  # Wandb checkpoints (optional)
│   └── runs/                          # Tensorboard logs (still created)
│
├── scripts/
│   ├── learning_curve.png             # Your final plot!
│   └── ...
│
└── wandb_exports/                     # CSV exports (if using export script)
    ├── policy_encoder_seed_1_seed1.csv
    ├── policy_encoder_seed_1_seed1_config.json
    └── ...
```

---

## 🔍 Verification Checklist

After running experiments, verify:

- [ ] All 5 encoder runs appear in `rad-embeddings-encoder` Wandb project
- [ ] All 5 policy runs appear in `rad-embeddings-policy` Wandb project
- [ ] Each run has `rollout/ep_rew_disc_mean` metric
- [ ] All runs are in the same group (`policy_training`)
- [ ] Model files exist in `storage/` and `exps_no_embed/`
- [ ] Learning curve plot shows mean line with shaded std region
- [ ] Plot has data from all 5 seeds

---

## 🎯 Key Decisions Made

1. **Two separate projects**: Encoder and policy training are logged to different projects for clarity
2. **Groups**: All policy runs are in the same group (`policy_training`) for easy aggregation
3. **Tensorboard preserved**: Tensorboard logging is still active (doesn't hurt to have both)
4. **WandbCallback**: Used for automatic integration with SB3
5. **LoggerCallback enhanced**: Added more metrics without breaking existing functionality
6. **Parallel support**: Master script can utilize both GPUs efficiently

---

## 🐛 Common Issues & Solutions

### Issue: Import errors when running
**Solution**: Run `uv sync` or `pip install -e .` first

### Issue: "api_key not configured"
**Solution**: Run `wandb login`

### Issue: Can't find runs when plotting
**Solution**: Check entity name with `wandb whoami`, verify project name matches

### Issue: Metrics not appearing
**Solution**: Wait for at least one episode to complete, check that LoggerCallback is in the callback list

### Issue: Out of memory on GPU
**Solution**: Reduce `n_envs` in the config, or use one GPU at a time (remove `--parallel`)

---

## 📈 Expected Results

After successful completion, you should have:

1. **On Wandb Dashboard**:
   - 5 encoder training runs
   - 5 policy training runs
   - Real-time plots of all metrics
   - Ability to compare runs side-by-side

2. **Local Files**:
   - 5 trained encoder models
   - 5 trained policy models
   - Learning curve plot (mean ± std)
   - (Optional) CSV exports of all data

3. **Key Plot**:
   - X-axis: Training steps (0 to 1,000,000)
   - Y-axis: `rollout/ep_rew_disc_mean`
   - Blue line: Mean across 5 seeds
   - Shaded region: ± 1 standard deviation

---

## 🎓 What You Learned

This implementation demonstrates:

- ✅ Integrating Wandb with Stable-Baselines3
- ✅ Multi-seed experiment orchestration
- ✅ Custom callback integration
- ✅ Efficient GPU utilization (parallel training)
- ✅ Automated result aggregation and visualization
- ✅ Production-ready experiment tracking

---

## 📝 Notes

- **Backwards Compatible**: All existing functionality preserved
- **Flexible**: Can disable Wandb with `export WANDB_MODE=disabled`
- **Extensible**: Easy to add more metrics in LoggerCallback
- **Reproducible**: Seeds are tracked, configs are logged
- **Scalable**: Works with any number of seeds (not just 5)

---

## 🤝 Contributing

To add new metrics:

1. Modify `utils/sb3/logger_callback.py`
2. Use `self.logger.record("namespace/metric_name", value)`
3. WandbCallback will automatically pick it up

To add new experiments:

1. Follow the pattern in `train_encoder.py` or `train_token_env_policy.py`
2. Add `wandb.init()` at the start
3. Add `WandbCallback` to callbacks list
4. Add `wandb.finish()` at the end

---

## 📚 References

- [Wandb Documentation](https://docs.wandb.ai)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Wandb + SB3 Integration](https://docs.wandb.ai/guides/integrations/other/stable-baselines-3)

---

**Status**: ✅ All implementation complete and ready to use!

**Last Updated**: November 18, 2025

