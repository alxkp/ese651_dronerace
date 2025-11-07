# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
from typing import List, Optional, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

# --- Optional Muon import -----------------------------------------------------
try:
    # Adjust to your Muon package if needed
    from muon import SingleDeviceMuonWithAuxAdam
    MUON_AVAILABLE = True
except Exception:
    SingleDeviceMuonWithAuxAdam = None
    MUON_AVAILABLE = False


# =============================================================================
# Right-preconditioning hooks (Linear only for MLPs)
# =============================================================================
def _symmetrize(M): return 0.5 * (M + M.t())

class _RightHooks(nn.Module):
    """
    Collects layer inputs X for right-preconditioning on Linear layers.
    Designed to be attached to the *actor* submodule only for PPO.
    """
    def __init__(self):
        super().__init__()
        self.layers: List[nn.Module] = []
        self._handles = []

    def _flatten_in(self, m, x):
        # MLP: nn.Linear input already [N, in_dim]
        if isinstance(m, nn.Linear):
            return x
        return None

    def attach(self, model: nn.Module):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                self.layers.append(m)
                def fwd_pre(mod, inp, _):
                    mod._pc_x = self._flatten_in(mod, inp[0].detach())
                self._handles.append(m.register_forward_pre_hook(fwd_pre, with_kwargs=True))
        return self

    def close(self):
        for h in self._handles:
            try: h.remove()
            except: pass
        self._handles.clear()


class RightPrecondWrapper(Optimizer):
    """
    Wraps a base optimizer and right-preconditions weight gradients of Linear layers by
    (X^T X + λ I)^{-1} on each step. Biases (ndim < 2) are untouched.
    Applies decoupled weight decay inside the wrapper.
    NOTE: Attach this wrapper to the *actor module only* for PPO stability.
    """
    def __init__(self,
        model: nn.Module,                   # <— pass actor submodule here
        base_optimizer: Optimizer,
        *,
        damping: float = 1e-4,
        damping_mode: str = 'trace',        # 'trace' or 'constant'
        jitter_scale: float = 1e-6,
        max_backoff: int = 3,
        approx_diag: bool = False,
        weight_decay: float = 0.0,          # decoupled WD (wrapper-applied)
        decay_bias: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__(base_optimizer.param_groups, dict(_dummy=True))
        assert damping_mode in ('trace','constant')
        self.model = model
        self.base_optimizer = base_optimizer
        self.damping = float(damping)
        self.damping_mode = damping_mode
        self.jitter_scale = float(jitter_scale)
        self.max_backoff = int(max_backoff)
        self.approx_diag = bool(approx_diag)
        self.weight_decay = float(weight_decay)
        self.decay_bias = bool(decay_bias)
        self.device_override = device
        self._hooks = _RightHooks().attach(model)  # <— only actor
        self._last_log: Optional[Dict] = None

        # Ensure base optimizer isn't also doing WD (we apply decoupled WD here)
        if any(g.get("weight_decay", 0.0) for g in self.base_optimizer.param_groups):
            print("[RightPrecondWrapper] WARNING: set base optimizer weight_decay=0; wrapper applies decoupled WD.")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        # for simple update norm logging
        before = {p: p.data.detach().clone()
                  for group in self.base_optimizer.param_groups
                  for p in group['params'] if p.grad is not None}

        logs = dict(total_grad_norm=0.0, total_precond_norm=0.0, update_norm=0.0)

        # Precondition only the attached module (actor) layers
        for m in self._hooks.layers:
            W = getattr(m, 'weight', None)
            gW = getattr(W, 'grad', None)
            if W is None or gW is None or W.ndim < 2:
                continue

            X = getattr(m, '_pc_x', None)
            if X is None:
                continue  # no cached input for this forward

            dev = self.device_override or gW.device
            G = gW.to(dev)
            X = X.to(dev)

            pre_before = torch.linalg.vector_norm(G).item()
            N = max(1, X.shape[0])
            Sin = _symmetrize((X.t() @ X) / float(N))  # [in_dim, in_dim]
            tr = torch.trace(Sin).item()
            lam = (self.damping * tr) if self.damping_mode == 'trace' else self.damping

            if self.approx_diag:
                d = torch.diagonal(Sin) + lam
                G = G * d.clamp_min(1e-12).reciprocal().unsqueeze(0)
            else:
                I = torch.eye(Sin.shape[0], device=Sin.device, dtype=Sin.dtype)
                alpha = self.jitter_scale * (tr if self.damping_mode == 'trace' else 1.0)
                for _ in range(self.max_backoff + 1):
                    try:
                        L = torch.linalg.cholesky(Sin + (lam + alpha) * I, upper=False)
                        G = torch.cholesky_solve(G.t(), L, upper=False).t()
                        break
                    except torch._C._LinAlgError:
                        alpha = max(1e-12, alpha * 10.0 if alpha > 0 else 1e-12)
                else:
                    d = torch.diagonal(Sin) + lam
                    G = G * d.clamp_min(1e-12).reciprocal().unsqueeze(0)

            pre_after = torch.linalg.vector_norm(G).item()
            logs['total_grad_norm'] += pre_before
            logs['total_precond_norm'] += pre_after
            W.grad.copy_(G.to(W.grad.device))

        # Decoupled WD (weights only unless decay_bias=True)
        if self.weight_decay > 0.0:
            for group in self.base_optimizer.param_groups:
                lr = group.get('lr', 1.0)
                for p in group['params']:
                    if p.grad is None: continue
                    if (not self.decay_bias) and (p.ndim < 2): continue
                    p.add_(p, alpha=-lr * self.weight_decay)

        self.base_optimizer.step()

        # Update norm log
        upd = 0.0
        for p, pre in before.items():
            upd += torch.linalg.vector_norm(p.data - pre).detach().cpu().item()
        logs['update_norm'] = upd
        self._last_log = logs
        return loss

    def zero_grad(self, set_to_none=True):
        return self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def pop_last_log(self) -> Optional[Dict]:
        out, self._last_log = self._last_log, None
        return out

    def close(self): self._hooks.close()
class JointOptimizer:
    """Thin wrapper so runners expecting a single optimizer still work."""
    def __init__(self, *optimizers):
        self.optimizers = list(optimizers)
        # minimal attr for compatibility
        self.param_groups = []
        for opt in self.optimizers:
            # extend but keep references (not copies)
            self.param_groups.extend(opt.param_groups)

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            if hasattr(opt, "zero_grad"):
                opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        for opt in self.optimizers:
            if hasattr(opt, "step"):
                opt.step()

    def state_dict(self):
        # preserve order; each optimizer’s own state dict kept under a key
        return {
            f"opt_{i}": opt.state_dict()
            for i, opt in enumerate(self.optimizers)
        }

    def load_state_dict(self, state):
        # tolerate missing or extra keys gracefully
        for i, opt in enumerate(self.optimizers):
            key = f"opt_{i}"
            if key in state:
                opt.load_state_dict(state[key])

# =============================================================================
# (Optional) Muon helpers
# =============================================================================
def build_muon_param_groups(model: nn.Module, *, hidden_lr: float, aux_lr: float, wd: float,
                            aux_betas: Tuple[float,float]=(0.9,0.95)) -> List[Dict]:
    """
    Split params so matrix weights (ndim>=2) use Muon, others use aux Adam.
    """
    body = list(model.parameters())
    hidden_w = [p for p in body if p.ndim >= 2]
    hidden_gb = [p for p in body if p.ndim <  2]
    return [
        dict(params=hidden_w, use_muon=True,  lr=hidden_lr, weight_decay=wd),
        dict(params=hidden_gb, use_muon=False, lr=aux_lr, betas=aux_betas, weight_decay=wd),
    ]


# =============================================================================
# PPO with separate actor/critic optimizers
# =============================================================================
class PPO:
    """Proximal Policy Optimization (https://arxiv.org/abs/1707.06347)."""

    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,         # <-- original Adam LR you used
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,

        # optimizer selection
        opt: str = "muon",         # 'adamw' or 'muon'
        weight_decay: float = 0.0,

        # right preconditioning (actor only)
        use_right_actor: bool = True,
        pc_damping: float = 1e-4,
        pc_diag: bool = False,
        decay_bias: bool = False,

        # critic lr & wd (often smaller / safer)
        critic_lr_scale: float = 0.5,
        critic_weight_decay: float = 0.0,

        # muon-specific (if opt='muon')
        muon_hidden_lr: Optional[float] = None,
        muon_aux_lr: Optional[float] = None,
        muon_wd: float = 0.00,
        muon_aux_betas: tuple = (0.9, 0.95),
    ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # modules
        self.actor_critic = actor_critic.to(self.device)

        # ---- split params into actor vs critic ----
        if hasattr(self.actor_critic, "actor") and hasattr(self.actor_critic, "critic"):
            actor_module = self.actor_critic.actor
            critic_module = self.actor_critic.critic
            actor_params  = list(actor_module.parameters())
            critic_params = list(critic_module.parameters())
        else:
            # fallback: by name
            actor_params, critic_params = [], []
            for n, p in self.actor_critic.named_parameters():
                (actor_params if ("actor" in n.lower()) else
                 critic_params if ("critic" in n.lower() or "value" in n.lower())
                 else actor_params).append(p)
            # create pseudo submodules for wrappers
            actor_module = nn.Module()
            critic_module = nn.Module()
            actor_module._modules["wrapped"] = self.actor_critic  # placeholder to attach hooks safely

        # ---- build actor optimizer (with optional right preconditioning) ----
        if opt.lower() == "muon":
            assert MUON_AVAILABLE, "Muon requested but not available."
            a_hidden_lr = muon_hidden_lr or learning_rate
            a_aux_lr    = muon_aux_lr    or learning_rate
            base_actor = SingleDeviceMuonWithAuxAdam(
                build_muon_param_groups(actor_module, hidden_lr=a_hidden_lr, aux_lr=a_aux_lr, wd=muon_wd, aux_betas=muon_aux_betas)
            )
            # keep wd in base for muon (or set to 0 if wrapping)
        elif opt.lower() == "adamw":
            base_actor = optim.AdamW(actor_params, lr=learning_rate, weight_decay=0.0)  # wd via wrapper decoupled
        else:
            raise ValueError(f"Unknown opt '{opt}'. Use 'adamw' or 'muon'.")

        if use_right_actor:
            # move WD into wrapper as decoupled WD
            for g in base_actor.param_groups:
                g["weight_decay"] = 0.0
            self.opt_actor = RightPrecondWrapper(
                model=actor_module,                 # <— attach hooks only to actor
                base_optimizer=base_actor,
                damping=pc_damping,
                damping_mode='trace',
                jitter_scale=1e-6,
                max_backoff=3,
                approx_diag=pc_diag,
                weight_decay=weight_decay,         # decoupled WD here
                decay_bias=decay_bias,
                device=torch.device(device),
            )
        else:
            self.opt_actor = base_actor


        # ---- build critic optimizer (NO right preconditioning) ----
        # c_lr = max(1e-6, float(critic_lr_scale) * float(learning_rate))
        # if opt.lower() == "muon":
        #     c_hidden_lr = muon_hidden_lr or c_lr
        #     c_aux_lr    = muon_aux_lr    or c_lr
        #     self.opt_critic = SingleDeviceMuonWithAuxAdam(
        #         build_muon_param_groups(critic_module, hidden_lr=c_hidden_lr, aux_lr=c_aux_lr, wd=critic_weight_decay, aux_betas=muon_aux_betas)
        #     )
        # else:
        self.opt_critic = optim.AdamW(critic_params, lr=learning_rate, weight_decay=critic_weight_decay)

        # rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()
        self.optimizer = JointOptimizer(self.opt_actor, self.opt_critic)

        # PPO scalars
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # for separate clipping
        self.actor_clip_norm = float(max_grad_norm)
        self.critic_clip_norm = 0.3  # more conservative for value net

    # -------------------------------------------------------------------------
    # Standard PPO API
    # -------------------------------------------------------------------------
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            None,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0

        # minibatch generator
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            observations,
            critic_observations,
            sampled_actions,
            value_targets,
            advantage_estimates,
            discounted_returns,
            prev_log_probs,
            prev_mean_actions,
            prev_action_stds,
            hidden_states,
            episode_masks,
            _,
        ) in generator:

            # Rebuild CURRENT policy distribution on this minibatch
            self.actor_critic.update_distribution(observations)

            # ---- Value loss (with optional clipping) ----
            estimated_values = self.actor_critic.evaluate(critic_observations)
            if self.use_clipped_value_loss:
                value_pred_clipped = value_targets + (estimated_values - value_targets).clamp(-self.clip_param, self.clip_param)
                v1 = (estimated_values - discounted_returns).pow(2)
                v2 = (value_pred_clipped - discounted_returns).pow(2)
                value_loss = 0.5 * torch.max(v1, v2).mean()
            else:
                value_loss = 0.5 * (estimated_values - discounted_returns).pow(2).mean()
            mean_value_loss += float(value_loss)

            # ---- PPO clipped surrogate for actor ----
            curr_log_probs = self.actor_critic.get_actions_log_prob(sampled_actions).squeeze(-1)  # [mb]
            old_log_probs  = prev_log_probs.squeeze(-1)                                           # [mb]
            ratio = torch.exp(curr_log_probs - old_log_probs)
            adv = advantage_estimates.squeeze(-1)
            unclipped = ratio * adv
            clipped   = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
            policy_obj = torch.min(unclipped, clipped)
            policy_loss = -policy_obj.mean()
            mean_surrogate_loss += float(policy_obj.mean())

            # ---- Entropy bonus ----
            entropy_vec = self.actor_critic.entropy
            if entropy_vec.dim() > 1: entropy_vec = entropy_vec.squeeze(-1)
            entropy = entropy_vec.mean()
            mean_entropy += float(entropy)

            total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            # ---- Separate optimizers and clipping ----
            self.opt_actor.zero_grad(set_to_none=True)
            self.opt_critic.zero_grad(set_to_none=True)
            total_loss.backward()

            # separate clip for actor vs critic
            # locate params robustly:
            if hasattr(self.actor_critic, "actor") and hasattr(self.actor_critic, "critic"):
                actor_params  = list(self.actor_critic.actor.parameters())
                critic_params = list(self.actor_critic.critic.parameters())
            else:
                actor_params, critic_params = [], []
                for n, p in self.actor_critic.named_parameters():
                    ((actor_params if "actor" in n.lower() else
                      critic_params if ("critic" in n.lower() or "value" in n.lower()) else actor_params)
                     .append(p))

            torch.nn.utils.clip_grad_norm_(actor_params,  self.actor_clip_norm)
            torch.nn.utils.clip_grad_norm_(critic_params, self.critic_clip_norm)

            self.opt_actor.step()   # right-preconditioned (actor only)
            self.opt_critic.step()  # plain optimizer (critic)

        num_updates = max(1, self.num_learning_epochs * self.num_mini_batches)
        mean_value_loss       /= num_updates
        mean_surrogate_loss   /= num_updates
        mean_entropy          /= num_updates

        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss, mean_entropy
