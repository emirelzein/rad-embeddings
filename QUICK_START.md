# Quick Start: Running 5-Seed Experiments with Wandb

## TL;DR

```bash
# 1. Setup (one-time)
uv sync
wandb login

# 2. Run all experiments (recommended for 2 GPUs)
cd rad_embeddings
python run_all_experiments.py --parallel

# 3. Plot results
cd scripts
python plot_wandb_learning_curve.py \
    --entity YOUR_WANDB_USERNAME \
    --project rad-embeddings-policy \
    --output learning_curve.png
```

---

## What Happens

### For Each Seed (1-5):

1. **Train Encoder** → Saves to `storage/DFABisimEnv-v1-encoder_{seed}.zip`
2. **Train Policy** with that encoder → Saves to `exps_no_embed/token_env_reach_avoid_policy_seed{seed}.zip`

### Both Steps Log to Wandb:
- Real-time metrics during training
- `rollout/ep_rew_disc_mean` (your key metric!)
- All other rollout statistics
- Model checkpoints

---

## File Structure After Running

```
rad_embeddings/
├── storage/
│   ├── DFABisimEnv-v1-encoder_1.zip  # Trained encoders
│   ├── DFABisimEnv-v1-encoder_2.zip
│   ├── ...
│   └── DFABisimEnv-v1-encoder_5.zip
│
├── exps_no_embed/
│   ├── token_env_reach_avoid_policy_seed1.zip  # Trained policies
│   ├── token_env_reach_avoid_policy_seed2.zip
│   ├── ...
│   └── token_env_reach_avoid_policy_seed5.zip
│
└── scripts/
    └── learning_curve.png  # Your final plot!
```

---

## Modified Files Summary

### Training Scripts (now with Wandb):
- ✅ `train_encoder.py` - Logs encoder training
- ✅ `train_token_env_policy.py` - Logs policy training
- ✅ `encoder.py` - Integrates Wandb into Encoder.train()
- ✅ `utils/sb3/logger_callback.py` - Enhanced metrics logging

### New Scripts:
- 🆕 `run_all_experiments.py` - Master orchestration script
- 🆕 `scripts/plot_wandb_learning_curve.py` - Generate mean ± std plots
- 🆕 `scripts/export_wandb_to_csv.py` - Export data for offline analysis

### Configuration:
- ✅ `pyproject.toml` - Added wandb, pandas, matplotlib dependencies

---

## Expected Runtime

- **Encoder training**: ~1 hour per seed (1M steps)
- **Policy training**: ~1 hour per seed (1M steps)
- **Total per seed**: ~2 hours
- **All 5 seeds (sequential)**: ~10 hours
- **All 5 seeds (parallel, 2 GPUs)**: ~6 hours

*Note: Exact times depend on your hardware and environment.*

---

## Key Metrics to Watch

| Metric | What it Means |
|--------|---------------|
| `rollout/ep_rew_disc_mean` | **PRIMARY**: Mean discounted reward (what you're plotting!) |
| `rollout/ep_len_mean` | How long episodes last on average |
| `train/policy_loss` | Policy gradient loss (should decrease) |
| `train/value_loss` | Value function loss (should decrease) |

---

## Troubleshooting

### Experiments failing?
```bash
# Check GPU availability
nvidia-smi

# Run with verbose output
python run_all_experiments.py 2>&1 | tee experiment_log.txt
```

### Wandb not logging?
```bash
# Check login status
wandb status

# Re-login if needed
wandb login
```

### Plot script can't find data?
```bash
# Check your Wandb username
wandb whoami

# List your projects
wandb projects
```

---

## Next Steps After Experiments Complete

1. **View on Wandb Web Interface:**
   - Go to https://wandb.ai
   - Navigate to `rad-embeddings-policy` project
   - Compare all 5 runs

2. **Generate Publication Plot:**
   ```bash
   cd scripts
   python plot_wandb_learning_curve.py \
       --entity YOUR_USERNAME \
       --project rad-embeddings-policy \
       --metric rollout/ep_rew_disc_mean \
       --title "Policy Learning Curve (5 Seeds)" \
       --output ../figures/learning_curve_high_res.png
   ```

3. **Export Raw Data (optional):**
   ```bash
   python export_wandb_to_csv.py \
       --entity YOUR_USERNAME \
       --project rad-embeddings-policy \
       --output ../results/
   ```

---

## Need More Details?

See `WANDB_SETUP_GUIDE.md` for comprehensive documentation.

