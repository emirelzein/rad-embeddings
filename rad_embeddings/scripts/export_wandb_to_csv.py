"""
Export Wandb data to CSV files for offline analysis.

This script downloads metrics from Wandb and saves them as CSV files,
which can be useful for offline analysis or archiving.

Usage:
    python export_wandb_to_csv.py --entity YOUR_ENTITY --project rad-embeddings-policy --output results/
"""

import argparse
import pandas as pd
import wandb
from pathlib import Path

def export_runs_to_csv(
    entity: str,
    project: str,
    group: str,
    output_dir: str,
    metrics: list = None
):
    """
    Export all runs in a group to CSV files.
    
    Args:
        entity: Wandb entity
        project: Wandb project name
        group: Group name to filter runs
        output_dir: Directory to save CSV files
        metrics: List of metrics to export (None = all)
    """
    api = wandb.Api()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all runs in the group
    runs = api.runs(
        f"{entity}/{project}",
        filters={"group": group}
    )
    
    print(f"Found {len(runs)} runs in group '{group}'")
    
    for run in runs:
        seed = run.config.get("seed", "unknown")
        run_name = run.name.replace("/", "_").replace(" ", "_")
        
        print(f"Exporting run: {run_name} (seed: {seed})...")
        
        # Download history
        if metrics:
            history = run.history(keys=["_step"] + metrics)
        else:
            history = run.history()
        
        if history.empty:
            print(f"  Warning: No data found for run {run_name}")
            continue
        
        # Save to CSV
        csv_file = output_path / f"{run_name}_seed{seed}.csv"
        history.to_csv(csv_file, index=False)
        print(f"  ✓ Saved to: {csv_file}")
        
        # Also save config
        config_file = output_path / f"{run_name}_seed{seed}_config.json"
        import json
        with open(config_file, "w") as f:
            json.dump(dict(run.config), f, indent=2)
        print(f"  ✓ Saved config to: {config_file}")
    
    print(f"\n✓ All data exported to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Export Wandb runs to CSV files"
    )
    parser.add_argument("--entity", type=str, required=True, help="Wandb entity")
    parser.add_argument("--project", type=str, required=True, help="Wandb project name")
    parser.add_argument("--group", type=str, default="policy_training", help="Group name")
    parser.add_argument("--output", type=str, default="wandb_exports/", help="Output directory")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Specific metrics to export (default: all)"
    )
    
    args = parser.parse_args()
    
    export_runs_to_csv(
        entity=args.entity,
        project=args.project,
        group=args.group,
        output_dir=args.output,
        metrics=args.metrics
    )
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

