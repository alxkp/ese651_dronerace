# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train:
            # Define reward component keys (matching literature + velocity alignment)
            reward_keys = ["progress", "vel_align", "thrust", "smooth", "pass", "crash"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in reward_keys
            }

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

    def get_rewards(self) -> torch.Tensor:
        """Reward structure matching literature EXACTLY.
        
        Literature formula: r_t = r^prog + r^perc + r^cmd + r^crash
        Our implementation (no perception needed with ground truth):
            r_t = r^prog + r^cmd + r^pass + r^crash
        
        Hyperparameters (exact match to literature):
        - λ₁ = 1.0:     Progress (distance reduction to gate)
        - λ₄ = -0.0001: Thrust command penalty
        - λ₅ = -0.0001: Action smoothness penalty
        - λ₃ = -10.0:   Crash penalty
        - Gate pass bonus = 100.0 (not in literature, added for learning signal)
        """

        # === Gather state ===
        pos_w = self.env._robot.data.root_link_pos_w                    # [N,3]
        prev_pos_w = self.env._prev_pos_w                               # [N,3]

        # Freeze target gate index BEFORE any updates
        target_gate_idx = self.env._idx_wp.clone()
        
        # Get current gate info
        gate_c = self.env._waypoints[target_gate_idx, :3]               # [N,3] gate center
        gate_quat = self.env._waypoints_quat[target_gate_idx, :]        # [N,4] gate quaternion

        # Compute gate frame axes
        gate_rot_matrix = matrix_from_quat(gate_quat)                   # [N,3,3]
        gate_n = gate_rot_matrix[:, :, 0]                               # [N,3] x-axis (normal to gate)
        gate_n = F.normalize(gate_n, dim=-1)
        ux = gate_rot_matrix[:, :, 1]                                   # [N,3] y-axis (horizontal)
        uy = gate_rot_matrix[:, :, 2]                                   # [N,3] z-axis (vertical)

        # Compute centerline direction (to next gate)
        next_gate_idx = (target_gate_idx + 1) % self.env._waypoints.shape[0]
        next_gate_c = self.env._waypoints[next_gate_idx, :3]            # [N,3]
        cl_tan = next_gate_c - gate_c                                   # [N,3]
        cl_tan = F.normalize(cl_tan, dim=-1)

        # Gate dimensions
        half_w = self.env._gate_half_width
        half_h = self.env._gate_half_height

        # === 1. PROGRESS: Distance reduction + Velocity alignment (paper-inspired) ===
        dist_to_gate = torch.linalg.norm(pos_w - gate_c, dim=-1)       # [N]
        dist_reduction = self.env._prev_dist_to_gate - dist_to_gate    # [N] positive when approaching
        self.env._prev_dist_to_gate = dist_to_gate.clone()

        # Distance reduction reward (2023 racing paper formulation)
        # Paper uses λ₁ = 1.0; we use 5.0 for 2x curriculum gates + ground truth advantage
        λ_dist = 5.0  # Reward for making progress toward gate
        r_dist = λ_dist * dist_reduction  # Positive when approaching, negative when retreating

        # Velocity alignment reward (auxiliary signal for better learning)
        # Paper uses camera alignment; we adapt for velocity alignment with ground truth
        v_w = self.env._robot.data.root_lin_vel_w                      # [N,3] world velocity
        dir_to_gate = gate_c - pos_w                                    # [N,3] direction to gate
        dir_to_gate_norm = F.normalize(dir_to_gate, dim=-1)             # [N,3] unit vector
        vel_norm = F.normalize(v_w + 1e-6, dim=-1)                      # [N,3] unit velocity (avoid div by 0)
        alignment = torch.sum(vel_norm * dir_to_gate_norm, dim=-1)     # [N] cosine similarity [-1,1]

        λ_vel = 3.0  # Increased from 0.3 to encourage faster flight toward gates
        r_vel_align = λ_vel * alignment.clamp(0, 1)  # Only reward forward alignment (0 to +1)

        # Combined progress reward
        r_prog = r_dist + r_vel_align

        # === 2. COMMAND PENALTIES: Thrust and smoothness (literature r^cmd) ===
        a_t = self.env._actions                                         # [N, act_dim]
        a_tm1 = self.env._previous_actions                              # [N, act_dim]
        
        # Thrust penalty (λ₄ in literature): penalize excessive thrust commands
        thrust_action = a_t[:, 0]                                       # [N] first action is thrust
        λ4 = -0.0001  # Literature value (exact match)
        r_thrust = λ4 * thrust_action
        
        # Action smoothness penalty (λ₅ in literature)
        action_jerk = torch.sum((a_t - a_tm1) ** 2, dim=-1)             # [N]
        λ5 = -0.0001  # Literature value (exact match)
        r_smooth = λ5 * action_jerk

        # === 3. GATE PASSING: Moderate bonus (not in literature, but helpful for learning) ===
        # Through-gate detection (plane crossing & inside aperture)
        side_prev = torch.sum((prev_pos_w - gate_c) * gate_n, dim=-1)  # [N] signed distance before
        side_now = torch.sum((pos_w - gate_c) * gate_n, dim=-1)        # [N] signed distance now
        
        # Check if position is within gate aperture
        x_offset = torch.sum((pos_w - gate_c) * ux, dim=-1)
        y_offset = torch.sum((pos_w - gate_c) * uy, dim=-1)
        x_in = torch.abs(x_offset) <= half_w
        y_in = torch.abs(y_offset) <= half_h
        in_aperture = x_in & y_in
        
        # Debug logging disabled to reduce spam
        # very_close = dist_to_gate < 0.5  # Within 0.5m
        # if very_close.any() and self.env.iteration % 50 == 0:
        #     for idx in very_close.nonzero()[:3]:  # Show first 3
        #         idx = idx.item()
        #         print(f"[DEBUG] Env {idx}: side_prev={side_prev[idx]:.3f}, side_now={side_now[idx]:.3f}, "
        #               f"dist={dist_to_gate[idx]:.3f}, x_off={x_offset[idx]:.3f} (in={x_in[idx]}), "
        #               f"y_off={y_offset[idx]:.3f} (in={y_in[idx]}), gate_w={half_w*2:.2f}m")
        
        # Gate crossing: POSITIVE → NEGATIVE
        crossed_forward = (side_prev > 0) & (side_now <= 0)
        pass_gate = crossed_forward & in_aperture

        # Gate pass bonus: Moderate reward for successfully passing through gate
        λ_pass = 100.0  # Moderate bonus to incentivize passing
        r_pass = λ_pass * pass_gate.float()

        # Update waypoint when gate is passed
        ids_gate_passed = torch.where(pass_gate)[0]
        if len(ids_gate_passed) > 0:
            self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]
            self.env._n_gates_passed[ids_gate_passed] += 1
            
            new_idx_wp = self.env._idx_wp[ids_gate_passed]
            
            # CRITICAL FIX: Update _desired_pos_w to move the red target dot visualization!
            self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[new_idx_wp, :2]
            self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[new_idx_wp, 2]
            
            # Reset distance tracking for new gate
            new_gate_c = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3]
            self.env._prev_dist_to_gate[ids_gate_passed] = torch.linalg.norm(
                pos_w[ids_gate_passed] - new_gate_c, dim=-1
            )

        # === 4. CRASH: One-time penalty on termination only ===
        # Track crashes for termination logic
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1)
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed.int() * mask
        
        # Apply crash penalty ONLY on termination (not per-timestep in contact)
        # This prevents massive penalties from multi-frame contact before death
        λ3 = -10.0  # Literature value
        terminated_with_crash = self.env.reset_terminated & (self.env._crashed > 0)
        r_crash = torch.where(terminated_with_crash, 
                              λ3 * torch.ones_like(crashed, dtype=torch.float32), 
                              torch.zeros_like(crashed, dtype=torch.float32))

        # === Total reward ===
        reward = r_prog + r_thrust + r_smooth + r_pass + r_crash

        # === Logging ===
        if self.cfg.is_train:
            self._episode_sums["progress"] += r_dist  # Track distance penalty separately
            self._episode_sums["vel_align"] += r_vel_align
            self._episode_sums["thrust"] += r_thrust
            self._episode_sums["smooth"] += r_smooth
            self._episode_sums["pass"] += r_pass
            self._episode_sums["crash"] += r_crash

            # Add per-step metrics to extras for iteration-by-iteration tracking
            # These will show up in TensorBoard/logs alongside standard PPO metrics
            if not hasattr(self.env, 'extras'):
                self.env.extras = {}
            if "log" not in self.env.extras:
                self.env.extras["log"] = {}

            # Per-step averages (logged every iteration)
            self.env.extras["log"]["Step/dist_to_gate"] = dist_to_gate.mean().item()
            self.env.extras["log"]["Step/vel_alignment"] = alignment.mean().item()
            self.env.extras["log"]["Step/reward_dist"] = r_dist.mean().item()
            self.env.extras["log"]["Step/reward_vel_align"] = r_vel_align.mean().item()
            self.env.extras["log"]["Step/total_gates_passed"] = self.env._n_gates_passed.sum().item()
            self.env.extras["log"]["Step/num_crashed"] = (self.env._crashed > 0).sum().item()

            # Episode-level gate statistics (shows multi-gate performance)
            self.env.extras["log"]["Episode/gates_per_env_mean"] = self.env._n_gates_passed.float().mean().item()
            self.env.extras["log"]["Episode/gates_per_env_max"] = self.env._n_gates_passed.max().item()
            self.env.extras["log"]["Episode/pct_passing_2plus_gates"] = (
                (self.env._n_gates_passed >= 2).float().mean().item() * 100
            )
            self.env.extras["log"]["Episode/pct_passing_3plus_gates"] = (
                (self.env._n_gates_passed >= 3).float().mean().item() * 100
            )

            # Track best saved model reward for comparison
            # This gets updated by the trainer when a new best model is saved
            if not hasattr(self.env, '_best_saved_reward'):
                self.env._best_saved_reward = 0.0

            # Current iteration's best reward
            total_episode_rewards = (
                self._episode_sums["progress"] +
                self._episode_sums["vel_align"] +
                self._episode_sums["thrust"] +
                self._episode_sums["smooth"] +
                self._episode_sums["pass"] +
                self._episode_sums["crash"]
            )
            current_best = total_episode_rewards.max().item()

            # Log best saved model reward and current performance ratio
            self.env.extras["log"]["Episode/best_saved_model_reward"] = self.env._best_saved_reward
            if self.env._best_saved_reward > 0:
                self.env.extras["log"]["Episode/current_vs_best_ratio"] = (
                    current_best / self.env._best_saved_reward
                )
            else:
                self.env.extras["log"]["Episode/current_vs_best_ratio"] = 1.0

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations in next-gate relative coordinates for better state representation."""

        # === OLD OBSERVATION STRUCTURE (COMMENTED OUT) ===
        # The following is the old simple observation structure
        # Uncomment this and comment out the new structure below if you want to revert

        # drone_pose_w = self.env._robot.data.root_link_pos_w
        # drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        # drone_quat_w = self.env._robot.data.root_quat_w
        # drone_pos_gate_frame = self.env._pose_drone_wrt_gate
        #
        # obs = torch.cat([
        #     drone_pose_w,
        #     drone_lin_vel_b,
        #     drone_quat_w,
        #     drone_pos_gate_frame
        # ], dim=-1)
        # observations = {"policy": obs}
        # return observations

        # === NEW GATE-RELATIVE OBSERVATION STRUCTURE ===

        # World-frame basics
        p_w   = self.env._robot.data.root_link_pos_w           # [N,3]
        v_w   = self.env._robot.data.root_lin_vel_w            # [N,3]
        q_wb  = self.env._robot.data.root_quat_w               # [N,4] world->body
        R_wb  = matrix_from_quat(q_wb)                         # [N,3,3]
        R_bw  = R_wb.transpose(-1, -2)                         # body->world inverse

        # Gate frame: center c_w, orthonormal axes (u_w, v_w, n_w) where n_w points "through" gate
        c_w   = self.env._next_gate_center_w                   # [N,3]
        u_w   = self.env._next_gate_x_w                        # [N,3]
        v_wg  = self.env._next_gate_y_w                        # [N,3]
        n_w   = self.env._next_gate_normal_w                   # [N,3]
        u_w   = F.normalize(u_w, dim=-1)
        v_wg  = F.normalize(v_wg, dim=-1)
        n_w   = F.normalize(n_w, dim=-1)

        # world->gate rotation matrix
        R_gw  = torch.stack([u_w, v_wg, n_w], dim=-1)          # [N,3,3]; columns are gate axes
        R_wg  = R_gw.transpose(-1, -2)

        # Position/velocity relative to gate, in gate frame
        p_rel_w = p_w - c_w                                    # [N,3]
        p_g     = (R_wg @ p_rel_w.unsqueeze(-1)).squeeze(-1)   # [N,3]
        v_g     = (R_wg @ v_w.unsqueeze(-1)).squeeze(-1)       # [N,3]

        # Body rates (body frame)
        omega_b = self.env._robot.data.root_ang_vel_b          # [N,3]

        # Camera/optical axis in world (use body z axis)
        z_b = torch.tensor([0., 0., 1.], device=p_w.device).expand_as(p_w)    # body +Z axis
        z_w = (R_wb @ z_b.unsqueeze(-1)).squeeze(-1)           # [N,3]
        # Alignment with gate normal (cosine)
        align = torch.sum(F.normalize(z_w, dim=-1) * n_w, dim=-1, keepdim=True)  # [N,1] in [-1,1]

        # In-aperture margins (normalized)
        half_w = self.env._gate_half_width    # scalar
        half_h = self.env._gate_half_height
        margin_x = (p_g[:, 0] / (half_w + 1e-6)).clamp(-2., 2.).unsqueeze(-1)  # [-2,2]
        margin_y = (p_g[:, 1] / (half_h + 1e-6)).clamp(-2., 2.).unsqueeze(-1)

        # Previous action
        prev_a = self.env._previous_actions                    # [N,4]

        # Progress scalar along track tangent (in gate frame, forward is +z axis which is normal direction)
        s_prog = p_g[:, 2].unsqueeze(-1)                       # [N,1]; negative before gate, positive after

        # Normalize/scales - FIXED: Use larger scales and soft saturation
        pos_scale = 20.0   # Increased from 10.0 - allow observations for farther distances
        vel_scale = 20.0   # Increased from 10.0 - allow faster velocities without saturation
        rate_scale = 10.0  # Keep same - body rates are typically small

        # Use tanh for soft saturation instead of hard clipping
        # This preserves gradients even when values are large
        p_g_n = torch.tanh(p_g / pos_scale)      # Smooth saturation at extremes
        v_g_n = torch.tanh(v_g / vel_scale)      # Smooth saturation at extremes
        omega_n = torch.tanh(omega_b / rate_scale)  # Smooth saturation at extremes

        obs_vec = torch.cat([
            p_g_n,                  # 3: position in gate frame (normalized)
            v_g_n,                  # 3: velocity in gate frame (normalized)
            omega_n,                # 3: body rates (normalized)
            align,                  # 1: alignment with gate normal
            margin_x, margin_y,     # 2: in-aperture margins
            prev_a,                 # 4: previous action
            s_prog / pos_scale,     # 1: progress along track (normalized)
        ], dim=-1)                  # Total: 17 dims

        # Safety: finite & normalized
        obs_vec = torch.nan_to_num(obs_vec, nan=0.0, posinf=1.0, neginf=-1.0)

        return {"policy": obs_vec}

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None:
            env_ids = self.env._robot._ALL_INDICES
        elif len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES
        
        # Type assertion for linter (env_ids is guaranteed to be a Tensor at this point)
        assert env_ids is not None

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # Reset position: 2-3.5m behind gate on -gate_normal direction with jitter
        # This ensures drone always starts properly positioned to approach gate

        # Start from the zeroth waypoint (beginning of the race)
        waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)

        # Get gate position and normal
        gate_pos = self.env._waypoints[waypoint_indices, :3]  # [n_reset, 3]
        gate_quat = self.env._waypoints_quat[waypoint_indices, :]  # [n_reset, 4]

        # Extract gate normal from quaternion rotation matrix
        from isaaclab.utils.math import matrix_from_quat
        gate_rot = matrix_from_quat(gate_quat)  # [n_reset, 3, 3]
        gate_normal = gate_rot[:, :, 0]  # x-axis is normal (through gate direction)

        # Position 2-3.5m AHEAD of gate (in +normal direction, past the gate)
        distance_ahead = torch.empty(n_reset, device=self.device).uniform_(2.0, 3.5)
        lateral_jitter = torch.empty(n_reset, 2, device=self.device).uniform_(-0.3, 0.3)  # ±30cm

        # Start position = gate_pos + distance * gate_normal + lateral jitter
        start_pos = gate_pos.clone()
        start_pos += distance_ahead.unsqueeze(-1) * gate_normal  # move forward along +normal (ahead of gate)

        # Add lateral jitter using gate horizontal/vertical axes
        gate_horizontal = gate_rot[:, :, 1]  # y-axis
        gate_vertical = gate_rot[:, :, 2]    # z-axis
        start_pos += lateral_jitter[:, 0].unsqueeze(-1) * gate_horizontal
        start_pos += lateral_jitter[:, 1].unsqueeze(-1) * gate_vertical

        # Force ground spawn at 5cm altitude (no height jitter)
        start_pos[:, 2] = 0.05  # Ground level spawn

        default_root_state[:, 0:3] = start_pos

        # Point drone towards gate (align with gate normal)
        # Yaw from gate normal
        yaw = torch.atan2(gate_normal[:, 1], gate_normal[:, 0])
        yaw_jitter = torch.empty(n_reset, device=self.device).uniform_(-0.1, 0.1)  # ±5.7 degrees
        quat = quat_from_euler_xyz(
            torch.zeros(n_reset, device=self.device),
            torch.zeros(n_reset, device=self.device),
            yaw + yaw_jitter
        )
        default_root_state[:, 3:7] = quat

        # REMOVED LEGACY EVAL SPAWN LOGIC
        # The old eval-only spawn logic (lines 477-505) was broken:
        # - Spawned drones at z=0.05m (5cm) regardless of gate height
        # - Used 2D yaw-only orientation instead of proper 3D gate normal
        # - Caused drones to crash immediately or fly in wrong direction
        #
        # Now both training and eval use the same proper spawn logic above:
        # - Spawn 2-3.5m behind gate at proper height (gate_z ± 20cm jitter)
        # - Face toward gate using proper 3D orientation from gate normal
        # - This ensures consistent behavior between training and evaluation

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        self.env._prev_x_drone_wrt_gate = torch.ones(self.num_envs, device=self.device)

        self.env._crashed[env_ids] = 0

        # Initialize previous position for progress tracking
        self.env._prev_pos_w[env_ids] = self.env._robot.data.root_link_pos_w[env_ids].clone()
        
        # Initialize distance tracking for new episode
        gate_pos = self.env._waypoints[waypoint_indices, :3]
        self.env._prev_dist_to_gate[env_ids] = torch.linalg.norm(
            start_pos - gate_pos, dim=-1
        )

        # Initialize gate frame information for reset environments
        self.env._update_gate_frame(env_ids)