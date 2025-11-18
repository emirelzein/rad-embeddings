"""
Plot learning curves from Wandb with mean and standard deviation across multiple seeds.

This script downloads data from Wandb and creates publication-quality learning curves
showing the mean performance across seeds with shaded standard deviation regions.

Usage:
    python plot_wandb_learning_curve.py --entity YOUR_WANDB_ENTITY --project rad-embeddings-policy
    python plot_wandb_learning_curve.py --entity YOUR_WANDB_ENTITY --project rad-embeddings-policy --metric rollout/ep_rew_disc_mean
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from pathlib import Path
from typing import List, Dict, Tuple

def download_runs_data(
    entity: str,
    project: str,
    group: str,
    metric: str = "rollout/ep_rew_disc_mean"
) -> Dict[int, pd.DataFrame]:
    """
    Download data from Wandb for all runs in a group.
    
    Args:
        entity: Wandb entity (username or team)
        project: Wandb project name
        group: Group name to filter runs
        metric: Metric to extract
        
    Returns:
        Dictionary mapping seed to dataframe with columns [step, metric_value]
    """
    api = wandb.Api()
    
    # Get all runs in the project with the specified group
    runs = api.runs(
        f"{entity}/{project}",
        filters={"group": group}
    )
    
    print(f"Found {len(runs)} runs in group '{group}'")
    
    seed_data = {}
    
    for run in runs:
        # Extract seed from config or run name
        seed = run.config.get("seed")
        if seed is None:
            # Try to extract from name
            import re
            match = re.search(r"seed[_\s](\d+)", run.name, re.IGNORECASE)
            if match:
                seed = int(match.group(1))
            else:
                print(f"Warning: Could not extract seed from run {run.name}, skipping")
                continue
        
        print(f"Downloading data for seed {seed} (run: {run.name})...")
        
        # Download the history (metrics over time)
        history = run.history(keys=["_step", metric])
        
        if history.empty:
            print(f"Warning: No data found for metric '{metric}' in run {run.name}")
            continue
        
        # Clean up the data
        history = history.dropna(subset=[metric])
        
        if len(history) == 0:
            print(f"Warning: No valid data for seed {seed}")
            continue
        
        seed_data[seed] = history[["_step", metric]].rename(columns={metric: "value"})
        print(f"  ✓ Downloaded {len(history)} data points")
    
    return seed_data

def align_and_aggregate_data(
    seed_data: Dict[int, pd.DataFrame],
    step_column: str = "_step"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align data from different seeds and compute mean and std.
    
    Args:
        seed_data: Dictionary mapping seed to dataframe
        step_column: Name of the step column
        
    Returns:
        Tuple of (steps, mean_values, std_values)
    """
    if not seed_data:
        raise ValueError("No data to aggregate")
    
    # Find common steps (intersection of all seeds)
    # Or we can interpolate to a common grid
    
    # Get all unique steps across all seeds
    all_steps = set()
    for df in seed_data.values():
        all_steps.update(df[step_column].values)
    all_steps = sorted(all_steps)
    
    # Create a regular grid for interpolation
    min_step = min(all_steps)
    max_step = max(all_steps)
    
    # Use a regular grid with reasonable number of points
    n_points = min(1000, len(all_steps))
    step_grid = np.linspace(min_step, max_step, n_points)
    
    # Interpolate each seed's data onto this grid
    interpolated_values = []
    
    for seed, df in seed_data.items():
        steps = df[step_column].values
        values = df["value"].values
        
        # Sort by steps
        sort_idx = np.argsort(steps)
        steps = steps[sort_idx]
        values = values[sort_idx]
        
        # Interpolate
        interp_values = np.interp(step_grid, steps, values)
        interpolated_values.append(interp_values)
    
    # Stack and compute statistics
    value_matrix = np.stack(interpolated_values, axis=0)  # Shape: (n_seeds, n_points)
    
    mean_values = np.mean(value_matrix, axis=0)
    std_values = np.std(value_matrix, axis=0)
    
    return step_grid, mean_values, std_values

def plot_learning_curve(
    steps: np.ndarray,
    mean_values: np.ndarray,
    std_values: np.ndarray,
    metric_name: str,
    output_file: str = None,
    title: str = None
):
    """
    Plot learning curve with mean and shaded std region.
    
    Args:
        steps: Array of step values
        mean_values: Array of mean metric values
        std_values: Array of std metric values
        metric_name: Name of the metric for labeling
        output_file: Path to save the figure (optional)
        title: Plot title (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot mean line
    ax.plot(steps, mean_values, linewidth=2, label="Mean", color="blue")
    
    # Plot shaded std region
    ax.fill_between(
        steps,
        mean_values - std_values,
        mean_values + std_values,
        alpha=0.3,
        color="blue",
        label="± 1 std"
    )
    
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
    ax.set_title(title or f"Learning Curve: {metric_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {output_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Plot learning curves from Wandb with mean and std across seeds"
    )
    parser.add_argument(
        "--entity",
        type=str,
        required=True,
        help="Wandb entity (username or team name)"
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Wandb project name"
    )
    parser.add_argument(
        "--group",
        type=str,
        default="policy_training",
        help="Group name to filter runs (default: policy_training)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="rollout/ep_rew_disc_mean",
        help="Metric to plot (default: rollout/ep_rew_disc_mean)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for the plot (e.g., learning_curve.png)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("Wandb Learning Curve Plotter")
    print(f"{'='*80}")
    print(f"Entity: {args.entity}")
    print(f"Project: {args.project}")
    print(f"Group: {args.group}")
    print(f"Metric: {args.metric}")
    print(f"{'='*80}\n")
    
    # Download data
    print("Downloading data from Wandb...")
    seed_data = download_runs_data(
        entity=args.entity,
        project=args.project,
        group=args.group,
        metric=args.metric
    )
    
    if not seed_data:
        print("Error: No data found!")
        return 1
    
    print(f"\n✓ Downloaded data for {len(seed_data)} seeds: {sorted(seed_data.keys())}\n")
    
    # Aggregate data
    print("Aggregating data across seeds...")
    steps, mean_values, std_values = align_and_aggregate_data(seed_data)
    print(f"✓ Aggregated {len(steps)} data points\n")
    
    # Plot
    print("Creating plot...")
    plot_learning_curve(
        steps=steps,
        mean_values=mean_values,
        std_values=std_values,
        metric_name=args.metric,
        output_file=args.output,
        title=args.title
    )
    
    print("\n✓ Done!\n")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

