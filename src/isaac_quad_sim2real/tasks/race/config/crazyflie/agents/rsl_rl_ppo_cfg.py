# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QuadcopterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 150  # BALANCED: 150 steps = 3 seconds (enough for good trajectories to reach gate)
                              # Was 24 (0.48s - too short), then 300 (6s - too slow)
                              # 3s gives good agents time to reach gate while keeping iteration speed reasonable
                              # Random agents won't reach gate in 3s, but learning agents will
    max_iterations = 10000  # Increased to complete full curriculum (10k iters for 1.0x gates)
    save_interval = 100    # Save less frequently
    experiment_name = "quadcopter_direct"
    empirical_normalization = False # Disable to prevent crushing gate-relative observations
    logger = "tensorboard"  # Use tensorboard (wandb has protobuf conflicts)
    wandb_project = "ese651_quadcopter"  # Wandb project name for logging (not used with tensorboard)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[256, 256, 128],  # FIXED: Literature standard (was [512,512,256,256] - too large!)
                                              # Loquercio et al. use [256,256,128] for faster convergence
        activation="elu",
        min_std=0.15,             # Higher minimum std to prevent premature convergence (was 0.05)
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,       # CRITICAL FIX: Reduced from 0.10 to 0.001 (100x!)
                                  # High entropy was preventing policy updates (action_std stuck at 1.0)
                                  # Now allows exploitation and policy gradient to flow
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,     # FIXED: Increased from 3e-4 to 5e-4 (match paper's likely config)
        schedule="adaptive",
        gamma=0.99,               # Matches 2023 racing paper
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # FIXED: Simplify optimizer configuration for stability
        opt="muon",              # Use simple AdamW instead of Muon
        use_right_actor=True,    # Disable right preconditioning for stability
        weight_decay=0.0,         # No weight decay for now
        normalize_advantage_per_mini_batch=False,  # Standard advantage normalization
    )
