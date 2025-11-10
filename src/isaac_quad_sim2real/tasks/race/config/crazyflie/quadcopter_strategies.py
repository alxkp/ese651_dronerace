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
            # Define reward component keys for the new reward structure
            reward_keys = ["progress", "align", "pass", "speed", "smooth", "wrong_way", "time", "crash", "height", "survival"]
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
        """Comprehensive reward structure for drone racing with progress, alignment, gate passing,
        speed shaping, and smoothness penalties."""

        # === OLD REWARD STRUCTURE (COMMENTED OUT) ===
        # The following is the old simple reward structure that hovers near gates
        # Uncomment this and comment out the new structure below if you want to revert

        # # check to change waypoint
        # dist_to_gate = torch.linalg.norm(self.env._pose_drone_wrt_gate, dim=1)
        # gate_passed = dist_to_gate < 0.1
        # ids_gate_passed = torch.where(gate_passed)[0]
        # self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]
        #
        # # set desired positions in the world frame
        # self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :2]
        # self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]
        #
        # # calculate progress via distance to goal
        # distance_to_goal = torch.linalg.norm(self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1)
        # distance_to_goal = torch.tanh(distance_to_goal/3.0)
        # progress = 1 - distance_to_goal
        #
        # # compute crashed environments
        # contact_forces = self.env._contact_sensor.data.net_forces_w
        # crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        # mask = (self.env.episode_length_buf > 100).int()
        # self.env._crashed = self.env._crashed + crashed * mask
        #
        # if self.cfg.is_train:
        #     rewards = {
        #         "progress_goal": progress * self.env.rew['progress_goal_reward_scale'],
        #         "crash": crashed * self.env.rew['crash_reward_scale'],
        #     }
        #     reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        #     reward = torch.where(self.env.reset_terminated,
        #                         torch.ones_like(reward) * self.env.rew['death_cost'], reward)
        #     for key, value in rewards.items():
        #         self._episode_sums[key] += value
        # else:
        #     reward = torch.zeros(self.num_envs, device=self.device)
        # return reward

        # === NEW COMPREHENSIVE REWARD STRUCTURE ===

        # === Gather state ===
        pos_w   = self.env._robot.data.root_link_pos_w                  # [N,3]
        vel_w   = self.env._robot.data.root_lin_vel_w                   # [N,3]
        omega_b = self.env._robot.data.root_ang_vel_b                   # [N,3]

        # Get current gate info
        current_gate_idx = self.env._idx_wp
        gate_c  = self.env._waypoints[current_gate_idx, :3]             # [N,3] gate center
        gate_yaw = self.env._waypoints[current_gate_idx, -1]            # [N] gate yaw
        gate_quat = self.env._waypoints_quat[current_gate_idx, :]       # [N,4] gate quaternion

        # Compute gate normal from quaternion (forward direction is x-axis rotated by quat)
        # For a gate, the normal is the direction perpendicular to the gate plane
        gate_rot_matrix = matrix_from_quat(gate_quat)                   # [N,3,3]
        gate_n = gate_rot_matrix[:, :, 0]                               # [N,3] x-axis (normal to gate)
        gate_n = F.normalize(gate_n, dim=-1)

        # Gate in-plane axes (y and z axes of gate frame)
        ux = gate_rot_matrix[:, :, 1]                                   # [N,3] y-axis (horizontal)
        uy = gate_rot_matrix[:, :, 2]                                   # [N,3] z-axis (vertical)

        # Compute centerline tangent (direction to next gate)
        next_gate_idx = (current_gate_idx + 1) % self.env._waypoints.shape[0]
        next_gate_c = self.env._waypoints[next_gate_idx, :3]            # [N,3]
        cl_tan = next_gate_c - gate_c                                   # [N,3]
        cl_tan = F.normalize(cl_tan, dim=-1)

        # Previous position for progress tracking
        prev_pos_w = self.env._prev_pos_w                               # [N,3]

        # Gate dimensions
        half_w = self.env._gate_half_width
        half_h = self.env._gate_half_height

        # DEBUG: Periodically log gate size and positions
        if self.env.iteration % 50 == 0 and self.num_envs > 0:
            print(f"[DEBUG] Iter {self.env.iteration}: Gate width={half_w*2:.2f}m, First gate at {gate_c[0].cpu().numpy()}, Drone at {pos_w[0].cpu().numpy()}")

        # === In-plane distance to gate center (remove normal component) ===
        to_gate = pos_w - gate_c
        in_plane = to_gate - torch.sum(to_gate * gate_n, dim=-1, keepdim=True) * gate_n
        d_ip = torch.linalg.norm(in_plane, dim=-1)                      # [N]

        # === Progress by projected displacement ===
        disp = pos_w - prev_pos_w
        ds   = torch.sum(disp * cl_tan, dim=-1)                         # [N]
        ds   = torch.clamp(ds, -0.5, 0.5)

        # NO progress shaping - just reward forward movement toward next gate
        ds_shaped = ds

        # === Through-gate detection (plane crossing & inside aperture) ===
        side_prev = torch.sum((prev_pos_w - gate_c) * gate_n, dim=-1)  # [N]
        side_now  = torch.sum((pos_w - gate_c)  * gate_n, dim=-1)      # [N]
        crossed = (side_prev < 0) & (side_now >= 0)
        x_in = torch.abs(torch.sum((pos_w - gate_c) * ux, dim=-1)) <= half_w
        y_in = torch.abs(torch.sum((pos_w - gate_c) * uy, dim=-1)) <= half_h
        pass_gate = crossed & x_in & y_in

        # Update waypoint index when gate is passed
        ids_gate_passed = torch.where(pass_gate)[0]
        if len(ids_gate_passed) > 0:
            # DEBUG: Print when gates are passed
            if self.env.iteration % 10 == 0:  # Only log every 10 iterations
                print(f"[DEBUG] Iter {self.env.iteration}: {len(ids_gate_passed)} envs passed gates! Gate size: {self.env._gate_half_width*2:.2f}m")
            self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]
            self.env._n_gates_passed[ids_gate_passed] += 1
            # Update gate frame information for environments that passed a gate
            self.env._update_gate_frame(ids_gate_passed)

        # === Speed shaping ===
        v_along   = torch.sum(vel_w * cl_tan, dim=-1)                   # [N]
        v_side    = vel_w - v_along.unsqueeze(-1) * cl_tan
        v_side_n  = torch.linalg.norm(v_side, dim=-1)
        v_scale   = 6.0
        speed_term   = torch.tanh(v_along / v_scale)
        side_penalty = torch.tanh(v_side_n / v_scale)

        # === Smoothness (actions and rates) ===
        a_t   = self.env._actions                                       # [N, act_dim]
        a_tm1 = self.env._previous_actions                              # [N, act_dim]
        da = a_t - a_tm1
        act_jerk = torch.sum(da * da, dim=-1)                           # [N]
        rates_pen = torch.sum(omega_b * omega_b, dim=-1)

        # === Crash / bounds ===
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        # === Reward scales ===
        # ULTRA SIMPLIFIED: Just reward getting close to gates and passing through them
        if self.env.iteration < 2000:
            # Phase 1: Learn to navigate toward and through gates
            λ1, smax = 2.0, 0.5      # Progress toward gate
            λ2, σp   = 5.0, 1.0      # STRONG reward for being near gate center
            λ4       = 10000.0       # Massive gate passing reward
            λ5, λ6   = 1.0, 0.2      # Forward speed
            λ11      = 0.5           # Height maintenance
            λ12      = 0.1           # Survival bonus
        else:
            # Phase 2: Refine to faster, cleaner passes
            λ1, smax = 3.0, 0.5      # Progress
            λ2, σp   = 2.0, 0.8      # Moderate alignment
            λ4       = 10000.0       # Gate passing
            λ5, λ6   = 2.0, 0.3      # Speed
            λ11      = 0.3           # Height
            λ12      = 0.05          # Survival

        λ3       = 0.0           # Camera alignment disabled
        λ7, λ8   = 0.0, 0.0      # NO smoothness penalties - allow any maneuvers
        λ9, λ10  = 0.5, 0.0      # Light wrong-way penalty
        death_cost = self.env.rew.get('death_cost', -50.0) if hasattr(self.env, 'rew') else -50.0  # Light penalty

        # === Compute reward components ===
        r_prog   = λ1 * ds_shaped  # use shaped progress that requires aiming through gate
        r_align  = λ2 * torch.exp(-(d_ip * d_ip) / (σp * σp))
        r_pass   = λ4 * pass_gate.float()
        r_speed  =  λ5 * speed_term - λ6 * side_penalty
        r_smooth = -λ7 * act_jerk     - λ8 * rates_pen
        r_back   = -λ9 * F.relu(-v_along)
        r_time   = -λ10 * torch.ones_like(ds)

        # Add altitude maintenance reward to keep drone flying
        target_height = 1.0  # target height in meters
        height_error = torch.abs(pos_w[:, 2] - target_height)
        r_height = λ11 * torch.exp(-height_error)  # reward for staying near target height

        # Add survival bonus to encourage longer episodes
        r_survival = λ12 * torch.ones_like(ds)  # small reward per timestep for surviving

        reward_vec = r_prog + r_align + r_pass + r_speed + r_smooth + r_back + r_time + r_height + r_survival

        # Apply crash penalty & terminate
        reward_vec = torch.where(crashed.bool(), torch.full_like(reward_vec, death_cost), reward_vec)
        reward_vec = torch.where(self.env.reset_terminated, torch.full_like(reward_vec, death_cost), reward_vec)

        # === Logging ===
        if self.cfg.is_train:
            self._episode_sums["progress"]     += r_prog
            self._episode_sums["align"]        += r_align
            self._episode_sums["pass"]         += r_pass
            self._episode_sums["speed"]        += r_speed
            self._episode_sums["smooth"]       += r_smooth
            self._episode_sums["wrong_way"]    += r_back
            self._episode_sums["time"]         += r_time
            self._episode_sums["crash"]        += (crashed.float() * death_cost)
            self._episode_sums["height"]       += r_height
            self._episode_sums["survival"]     += r_survival

        return reward_vec

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

        # Normalize/scales
        pos_scale = 10.0
        vel_scale = 10.0
        rate_scale = 10.0

        p_g_n = (p_g / pos_scale).clamp(-1., 1.)
        v_g_n = (v_g / vel_scale).clamp(-1., 1.)
        omega_n = (omega_b / rate_scale).clamp(-1., 1.)

        obs_vec = torch.cat([
            p_g_n,                  # 3: position in gate frame (normalized)
            v_g_n,                  # 3: velocity in gate frame (normalized)
            omega_n,                # 3: body rates (normalized)
            align,                  # 1: alignment with gate normal
            margin_x, margin_y,     # 2: in-aperture margins
            prev_a,                 # 4: previous action
            s_prog / pos_scale      # 1: progress along track (normalized)
        ], dim=-1)                  # Total: 17 dims

        # Safety: finite & normalized
        obs_vec = torch.nan_to_num(obs_vec, nan=0.0, posinf=1.0, neginf=-1.0)

        return {"policy": obs_vec}

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

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

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        # This example code initializes the drone 2m behind the first gate. You should delete it or heavily
        # modify it once you begin the racing task.

        # start from the zeroth waypoint (beginning of the race)
        waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)

        # get starting poses behind waypoints
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        theta = self.env._waypoints[waypoint_indices][:, -1]
        z_wp = self.env._waypoints[waypoint_indices][:, 2]

        x_local = -2.0 * torch.ones(n_reset, device=self.device)
        y_local = torch.zeros(n_reset, device=self.device)
        z_local = torch.zeros(n_reset, device=self.device)

        # rotate local pos to global frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = z_local + z_wp

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # point drone towards the zeroth gate
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        quat = quat_from_euler_xyz(
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            initial_yaw + torch.empty(1, device=self.device).uniform_(-0.15, 0.15)
        )
        default_root_state[:, 3:7] = quat
        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

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

        # Initialize gate frame information for reset environments
        self.env._update_gate_frame(env_ids)