"""
Master script to run all experiments with multiple seeds.
This script orchestrates training of encoders and policies for 5 different seeds.

Usage:
    python run_all_experiments.py --parallel  # Run 2 seeds in parallel (for 2 GPUs)
    python run_all_experiments.py  # Run sequentially
"""

import subprocess
import argparse
import os
import sys
from pathlib import Path

def run_experiment(seed: int, gpu_id: int = None):
    """
    Run a complete experiment (encoder + policy) for a given seed.
    
    Args:
        seed: Random seed for the experiment
        gpu_id: GPU ID to use (0 or 1), or None for CPU/default
    """
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"\n{'='*80}")
    print(f"Starting Experiment - Seed: {seed}, GPU: {gpu_id if gpu_id is not None else 'default'}")
    print(f"{'='*80}\n")
    
    # Step 1: Train encoder
    print(f"[Seed {seed}] Step 1/2: Training encoder...")
    encoder_cmd = [sys.executable, "train_encoder.py", str(seed)]
    result = subprocess.run(encoder_cmd, env=env, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"[Seed {seed}] ERROR: Encoder training failed!")
        return False
    
    print(f"[Seed {seed}] ✓ Encoder training completed")
    
    # Step 2: Train policy with the encoder
    encoder_file = f"storage/DFABisimEnv-v1-encoder_{seed}.zip"
    print(f"[Seed {seed}] Step 2/2: Training policy with encoder...")
    policy_cmd = [
        sys.executable,
        "train_token_env_policy.py",
        "--seed", str(seed),
        "--encoder-file", encoder_file
    ]
    result = subprocess.run(policy_cmd, env=env, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"[Seed {seed}] ERROR: Policy training failed!")
        return False
    
    print(f"[Seed {seed}] ✓ Policy training completed")
    print(f"[Seed {seed}] ✓✓ Experiment completed successfully!\n")
    
    return True

def run_sequential(seeds):
    """Run experiments sequentially."""
    print("Running experiments SEQUENTIALLY")
    for seed in seeds:
        success = run_experiment(seed)
        if not success:
            print(f"Experiment with seed {seed} failed. Stopping.")
            return False
    return True

def run_parallel(seeds):
    """Run experiments in parallel (2 at a time for 2 GPUs)."""
    import concurrent.futures
    
    print("Running experiments IN PARALLEL (2 at a time)")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        # Map seeds to GPUs in a round-robin fashion
        futures = []
        for i, seed in enumerate(seeds):
            gpu_id = i % 2  # Alternate between GPU 0 and 1
            future = executor.submit(run_experiment, seed, gpu_id)
            futures.append((seed, future))
        
        # Wait for all to complete
        all_success = True
        for seed, future in futures:
            try:
                success = future.result()
                if not success:
                    print(f"Experiment with seed {seed} failed.")
                    all_success = False
            except Exception as e:
                print(f"Experiment with seed {seed} raised an exception: {e}")
                all_success = False
        
        return all_success

def main():
    parser = argparse.ArgumentParser(
        description="Run all experiments with 5 seeds"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel (2 at a time for 2 GPUs)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="List of seeds to run (default: 1 2 3 4 5)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("RAD Embeddings Experiment Suite")
    print(f"{'='*80}")
    print(f"Seeds: {args.seeds}")
    print(f"Mode: {'PARALLEL (2 GPUs)' if args.parallel else 'SEQUENTIAL'}")
    print(f"{'='*80}\n")
    
    # Run experiments
    if args.parallel:
        success = run_parallel(args.seeds)
    else:
        success = run_sequential(args.seeds)
    
    # Summary
    print(f"\n{'='*80}")
    if success:
        print("✓✓✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY! ✓✓✓")
    else:
        print("✗✗✗ SOME EXPERIMENTS FAILED ✗✗✗")
    print(f"{'='*80}\n")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

