# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QuadcopterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000  # Increased to accommodate longer curriculum
    save_interval = 100    # Save less frequently
    experiment_name = "quadcopter_direct"
    empirical_normalization = False # Disable to prevent crushing gate-relative observations
    wandb_project = "ese651_quadcopter"  # Wandb project name for logging
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[512, 256, 128, 128],
        activation="elu",
        min_std=0.15,             # Higher minimum std to prevent premature convergence (was 0.05)
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.05,        # Strong exploration to prevent premature convergence (was 0.02)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,     # Slightly lower LR (was 5e-4) for more stable learning
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
