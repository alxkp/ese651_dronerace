#!/usr/bin/env python3
"""Script to inspect _best_model.pt checkpoint."""

import torch
import argparse
from pathlib import Path


def inspect_checkpoint(checkpoint_path: str):
    """Load and print information from a checkpoint file."""

    print(f"\n{'='*80}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Print top-level keys
    print("Top-level keys in checkpoint:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    print()

    # Print model architecture info
    if 'model_state_dict' in checkpoint:
        print(f"{'='*80}")
        print("Model Architecture (state_dict keys):")
        print(f"{'='*80}")
        state_dict = checkpoint['model_state_dict']

        # Group by module
        actor_params = {}
        critic_params = {}
        other_params = {}

        for name, tensor in state_dict.items():
            if 'actor' in name.lower():
                actor_params[name] = tensor.shape
            elif 'critic' in name.lower() or 'value' in name.lower():
                critic_params[name] = tensor.shape
            else:
                other_params[name] = tensor.shape

        print("\nActor parameters:")
        for name, shape in sorted(actor_params.items()):
            print(f"  {name:60s} {str(shape):20s} ({tensor.numel():,} params)")

        print("\nCritic parameters:")
        for name, shape in sorted(critic_params.items()):
            print(f"  {name:60s} {str(shape):20s} ({tensor.numel():,} params)")

        if other_params:
            print("\nOther parameters:")
            for name, shape in sorted(other_params.items()):
                print(f"  {name:60s} {str(shape):20s} ({tensor.numel():,} params)")

        # Count total parameters
        total_actor = sum(t.numel() for t in actor_params.values())
        total_critic = sum(t.numel() for t in critic_params.values())
        total_other = sum(t.numel() for t in other_params.values())
        total = total_actor + total_critic + total_other

        print(f"\nParameter counts:")
        print(f"  Actor:  {total_actor:,}")
        print(f"  Critic: {total_critic:,}")
        if total_other > 0:
            print(f"  Other:  {total_other:,}")
        print(f"  Total:  {total:,}")

    # Print optimizer info
    if 'optimizer_state_dict' in checkpoint:
        print(f"\n{'='*80}")
        print("Optimizer State:")
        print(f"{'='*80}")
        opt_state = checkpoint['optimizer_state_dict']

        # Check if it's a JointOptimizer (multiple optimizers)
        if isinstance(opt_state, dict):
            print(f"\nOptimizer state keys: {list(opt_state.keys())}")

            for key in opt_state.keys():
                if key.startswith('opt_'):
                    print(f"\n{key}:")
                    sub_opt = opt_state[key]
                    if 'param_groups' in sub_opt:
                        for i, pg in enumerate(sub_opt['param_groups']):
                            print(f"  Param group {i}:")
                            for k, v in pg.items():
                                if k != 'params':  # Skip param IDs
                                    print(f"    {k}: {v}")

    # Print training metadata
    if 'iteration' in checkpoint:
        print(f"\n{'='*80}")
        print("Training Metadata:")
        print(f"{'='*80}")
        print(f"  Iteration: {checkpoint['iteration']}")

    if 'infos' in checkpoint:
        print(f"\nTraining info:")
        infos = checkpoint['infos']
        for key, value in infos.items():
            print(f"  {key}: {value}")

    # Check for any other interesting keys
    other_keys = [k for k in checkpoint.keys()
                  if k not in ['model_state_dict', 'optimizer_state_dict', 'iteration', 'infos']]
    if other_keys:
        print(f"\n{'='*80}")
        print("Other checkpoint data:")
        print(f"{'='*80}")
        for key in other_keys:
            value = checkpoint[key]
            if isinstance(value, (int, float, str, bool)):
                print(f"  {key}: {value}")
            elif isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor{tuple(value.shape)}")
            else:
                print(f"  {key}: {type(value).__name__}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect PyTorch checkpoint file")
    parser.add_argument(
        "checkpoint",
        type=str,
        nargs="?",
        default="/workspace/ese651_dronerace/logs/rsl_rl/quadcopter_direct/2025-11-11_20-52-55/_best_model.pt",
        help="Path to checkpoint file (supports wildcards)"
    )

    args = parser.parse_args()

    # Handle wildcards
    from glob import glob
    matches = glob(args.checkpoint)

    if not matches:
        print(f"Error: No checkpoint found matching: {args.checkpoint}")
        exit(1)

    if len(matches) > 1:
        print(f"Found {len(matches)} matching checkpoints:")
        for i, path in enumerate(matches):
            print(f"  {i+1}. {path}")
        print(f"\nInspecting most recent: {matches[-1]}\n")
        checkpoint_path = matches[-1]
    else:
        checkpoint_path = matches[0]

    inspect_checkpoint(checkpoint_path)
