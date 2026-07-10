import torch
import torch.nn as nn
import torch.nn.functional as F

from torchrl.data import CompositeSpec, TensorSpec
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs.transforms import CatTensors

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union
import einops
import math

from ..utils.valuenorm import ValueNorm1
from .common import GAE, make_mlp
from .ppo import PPOPolicy, PPOConfig, Actor, make_batch


@dataclass
class PPOLagrangianConfig:
    name: str = "ppo_lagrangian"
    train_every: int = 64
    ppo_epochs: int = 4
    num_minibatches: int = 16

    priv_actor: bool = False
    priv_critic: bool = False

    checkpoint_path: Union[str, None] = None

    lr: float = 5e-4

    # Lagrangian-specific
    # Average-power budget in Watts. Per-vehicle values from the paper (T200 thrusters):
    #   BlueROV 1300, BlueROV-Heavy 1200, CUREE-AUV 1300.
    # Default is the BlueROV/CUREE value; override per run, e.g. algo.cost_limit=1200
    cost_limit: float = 1300.0
    lagrange_lr: float = 5e-3
    init_log_lambda: float = -2
    cost_gamma: float = 0.99
    cost_lmbda: float = 0.95
    lambda_min: float = 0.05
    lambda_max: float = 2.0

    # Actuator model selection (auto-set from task config in train.py)
    drone_model_name: str = "BlueROV2"


cs = ConfigStore.instance()
cs.store("ppo_lagrangian", node=PPOLagrangianConfig, group="algo")
cs.store("ppo_lagrangian_priv", node=PPOLagrangianConfig(priv_actor=True, priv_critic=True), group="algo")
cs.store("ppo_lagrangian_priv_critic", node=PPOLagrangianConfig(priv_critic=True), group="algo")


class PPOLagrangianPolicy(PPOPolicy):

    def __init__(
        self,
        cfg: PPOLagrangianConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device
    ):
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device)

        fake_input = observation_spec.zero()

        if self.cfg.priv_critic:
            intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]
            self.cost_critic = TensorDictSequential(
                TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["feature"]),
                TensorDictModule(
                    nn.Sequential(nn.LayerNorm(intrinsics_dim), make_mlp([64, 64])),
                    [("agents", "intrinsics")], ["context"]
                ),
                CatTensors(["feature", "context"], "feature"),
                TensorDictModule(
                    nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(1)),
                    ["feature"], ["cost_value"]
                )
            ).to(self.device)
        else:
            self.cost_critic = TensorDictModule(
                nn.Sequential(make_mlp([256, 256, 256]), nn.LazyLinear(1)),
                [("agents", "observation")], ["cost_value"]
            ).to(self.device)

        self.cost_critic(fake_input)

        if self.cfg.checkpoint_path is None:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)
            self.cost_critic.apply(init_)

        self.cost_gae = GAE(cfg.cost_gamma, cfg.cost_lmbda)
        self.cost_value_norm = ValueNorm1(reward_spec.shape[-2:]).to(self.device)

        self.log_lambda = nn.Parameter(
            torch.tensor(cfg.init_log_lambda, dtype=torch.float32, device=self.device)
        )

        self.cost_critic_opt = torch.optim.Adam(self.cost_critic.parameters(), lr=5e-4)
        self.lambda_opt = torch.optim.Adam([self.log_lambda], lr=cfg.lagrange_lr)

    def __call__(self, tensordict: TensorDict):
        self.actor(tensordict)
        self.critic(tensordict)
        self.cost_critic(tensordict)
        tensordict.exclude("loc", "scale", "feature", inplace=True)
        return tensordict

    @staticmethod
    def _t200_thrust_to_power(thrust_n: torch.Tensor) -> torch.Tensor:
        """
        Convert T200 thrust (N) to electrical power (W) using 16V datasheet power law fit.

          Forward (T >= 0): P = 0.758 * T^1.574
          Reverse (T <  0): P = 0.851 * |T|^1.654
        """
        abs_t = thrust_n.abs()
        power_fwd = 0.758 * abs_t.pow(1.574)
        power_rev = 0.851 * abs_t.pow(1.654)
        return torch.where(thrust_n >= 0.0, power_fwd, power_rev)

    @staticmethod
    def _t200_action_to_thrust(actions: torch.Tensor) -> torch.Tensor:
        """
        Approximate T200 thrust (N) from raw action commands [-1, 1].

        Uses the T200 RPM model from t200.py (throttle = (action+1)/2 for positive range),
        then the quadratic RPM-to-thrust polynomial from the same file.
        This is a steady-state approximation (ignores rotor time constants).
        """
        # T200 maps action ∈ [-1,1] directly to throttle (bidirectional)
        throttle = actions  # T200 throttle is signed in [-1, 1]
        rpm = torch.where(
            throttle > 0.075,
            3.6599e3 * throttle + 3.4521e2,
            torch.where(throttle < -0.075,
                3.4944e3 * throttle - 4.3350e2,
                torch.zeros_like(throttle))
        ).clamp(-3900, 3900)  # match T200 hardware limit
        # T200 thrust polynomial from t200.py (force_constants normalised to 4.4e-7)
        thrust = 9.81 * torch.where(
            rpm > 0,
            4.7368e-7 * rpm.pow(2) - 1.9275e-4 * rpm + 8.4452e-2,
            -3.8442e-7 * rpm.pow(2) - 1.6186e-4 * rpm - 3.9139e-2,
        )
        # Datasheet hard limits: 5.25 kgf forward, 4.1 kgf reverse (at 16V)
        return thrust.clamp(-40.22, 51.50)

    def _compute_power_cost(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute instantaneous electrical power (W) from raw action commands.

        actions: [T, num_envs, 1, num_act_dims] in [-1, 1]
        Returns: [T, num_envs, 1, 1] Watts

        Actuator models:
          T200 (BlueROV2, BlueROV2Heavy, HAUV): action→thrust via T200 model,
              then thrust→power from 16V datasheet (P = a*|T|^1.5, asymmetric fwd/rev)
          M600 + fin200 (iAUV): M600 rotor (action[...,0]) + fin pair (action[...,1:])
        """
        drone = self.cfg.drone_model_name.lower()

        if "iauv" in drone:
            # iAUV: 1 M600 rotor (action[...,0]) + 2 fin control dims (action[...,1:])
            # M600: max 600 RPM, ~50 W at max → K = 50/600^2 = 1.389e-4 W/RPM²
            rotor_act = actions[..., :1]
            fin_act   = actions[..., 1:]
            rpm_r = torch.where(
                rotor_act > 0.075,  5.6599e2 * rotor_act + 3.4521e1,
                torch.where(rotor_act < -0.075, 5.4944e2 * rotor_act - 4.3350e1,
                    torch.zeros_like(rotor_act))
            )
            power_rotor = 1.389e-4 * rpm_r.pow(2)                       # [..., 1]
            power_fins  = 10.0 * fin_act.pow(2).sum(-1, keepdim=True)   # [..., 1]
            power = power_rotor + power_fins                             # [..., 1]
        else:
            # T200 thrusters (BlueROV2, BlueROV2Heavy, HAUV, TiltRotor, ...)
            thrust = self._t200_action_to_thrust(actions)                # [..., num_rotors]
            power  = self._t200_thrust_to_power(thrust).sum(-1, keepdim=True)  # [..., 1]

        # actions: [T, envs, 1, num_act]; power: [T, envs, 1, 1]
        return power

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_values = self.critic(next_tensordict)["state_value"]
            next_cost_values = self.cost_critic(next_tensordict)["cost_value"]

        rewards = tensordict[("next", "agents", "reward")]
        actions = tensordict[("agents", "action")]
        costs = self._compute_power_cost(actions)    # [T, num_envs, 1, 1] Watts
        dones = einops.repeat(
            tensordict[("next", "terminated")],
            "t e 1 -> t e a 1",
            a=self.n_agents
        )

        # Reward GAE
        values = tensordict["state_value"]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        # Cost GAE
        cost_values = tensordict["cost_value"]
        cost_values = self.cost_value_norm.denormalize(cost_values)
        next_cost_values = self.cost_value_norm.denormalize(next_cost_values)

        cost_adv, cost_ret = self.cost_gae(costs - self.cfg.cost_limit, dones, cost_values, next_cost_values)
        cost_adv = (cost_adv - cost_adv.mean()) / cost_adv.std().clip(1e-7)
        self.cost_value_norm.update(cost_ret)
        cost_ret = self.cost_value_norm.normalize(cost_ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)
        tensordict.set("cost_adv", cost_adv)
        tensordict.set("cost_ret", cost_ret)

        # Update Lagrange multiplier
        mean_cost = costs.mean()
        lambda_loss = -self.log_lambda * (mean_cost - self.cfg.cost_limit)
        self.lambda_opt.zero_grad()
        lambda_loss.backward()
        self.lambda_opt.step()
        with torch.no_grad():
            self.log_lambda.clamp_(
                min=math.log(self.cfg.lambda_min),
                max=math.log(self.cfg.lambda_max)
            )

        # PPO epochs
        infos = []
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))

        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        result = {k: v.item() for k, v in infos.items()}
        result["lagrange_lambda"] = self.log_lambda.exp().item()
        result["mean_cost"] = mean_cost.item()
        return result

    def _update(self, tensordict: TensorDict):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy()

        # Lagrangian-modified advantage
        lam = self.log_lambda.exp().detach()
        adv = tensordict["adv"]
        cost_adv = tensordict["cost_adv"]
        modified_adv = adv - lam * cost_adv

        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = modified_adv * ratio
        surr2 = modified_adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * torch.mean(entropy)

        # Value loss (reward critic)
        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        # Cost value loss (cost critic)
        b_cost_values = tensordict["cost_value"]
        b_cost_returns = tensordict["cost_ret"]
        cost_values = self.cost_critic(tensordict)["cost_value"]
        cost_values_clipped = b_cost_values + (cost_values - b_cost_values).clamp(
            -self.clip_param, self.clip_param
        )
        cost_value_loss_clipped = self.critic_loss_fn(b_cost_returns, cost_values_clipped)
        cost_value_loss_original = self.critic_loss_fn(b_cost_returns, cost_values)
        cost_value_loss = torch.max(cost_value_loss_original, cost_value_loss_clipped)

        loss = policy_loss + entropy_loss + value_loss + cost_value_loss

        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        self.cost_critic_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        cost_critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.cost_critic.parameters(), 5)
        # Skip optimizer step if any gradients are NaN (bad minibatch guard)
        all_params = (
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + list(self.cost_critic.parameters())
        )
        has_nan_grad = any(p.grad is not None and p.grad.isnan().any() for p in all_params)
        if not has_nan_grad:
            self.actor_opt.step()
            self.critic_opt.step()
            self.cost_critic_opt.step()

        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        if has_nan_grad:
            z = torch.tensor(0.0, device=self.device)
            policy_loss = z
            value_loss = z
            cost_value_loss = z
            actor_grad_norm = z
            critic_grad_norm = z
            cost_critic_grad_norm = z
            explained_var = z
        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "cost_value_loss": cost_value_loss,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "cost_critic_grad_norm": cost_critic_grad_norm,
            "explained_var": explained_var,
        }, [])
