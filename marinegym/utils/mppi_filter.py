"""
MPPI predictive safety filter for keep-out avoidance (arm B).

Model Predictive Path Integral control used as a *safety filter*: around the policy's
proposed action u_rl, sample K action sequences over an N-step horizon, roll them out on a
nominal translational model, score each by (deviation from u_rl) + (keep-out penalty over the
whole horizon), and return the importance-weighted first action.  Unlike the single-step CBF
shield (arm A) this is predictive (won't engage past the point of no return) and rolls out the
*whole* net-force response (so it favours gentle avoidance instead of the tumbling saturation
that broke arm A).  Pure torch, GPU-batched over envs x samples — no external QP/MPC solver.

Nominal model (world frame, orientation held constant over the short horizon — MarineGym's
get_state() gives world position & linear velocity):
    a_world = k * R (Blin^T a) / M_lin  -  drag * v          (thrust + linear drag)
    v <- v + dt a_world ;   p <- p + dt v
Cost per sample: w_track * sum_t ||a_t - u_rl||^2  +  w_coll * sum_t relu(r - ||p_t - p_o||)^2.
Weights: softmax(-cost / lambda).   u_safe = sum_k w_k * a_seq[k, 0].

Default disabled; validate collision-rate drop vs the filter-off baseline before trusting it.
"""

import torch

from marinegym.utils.torch import quat_rotate, quat_rotate_inverse, quat_axis


class MPPIShield:
    def __init__(
        self,
        drone,
        keepout_radius: float,
        vehicle_radius: float,
        dt: float,
        horizon: int = 20,
        num_samples: int = 128,
        noise_sigma: float = 0.4,
        temperature: float = 0.05,
        w_coll: float = 5.0,
        w_track: float = 1.0,
        drag: float = 1.0,
        thrust_gain: float = 40.0,
        activation_margin: float = 1.5,
    ):
        self.device = drone.device
        self.r = float(keepout_radius) + float(vehicle_radius)
        self.dt = float(dt)
        self.N = int(horizon)
        self.K = int(num_samples)
        self.sigma = float(noise_sigma)
        self.lam = float(temperature)
        self.w_coll = float(w_coll)
        self.w_track = float(w_track)
        self.drag = float(drag)
        self.k = float(thrust_gain)
        self.activation_margin = float(activation_margin)
        self.M = drone.num_rotors

        m = drone.masses.reshape(-1)[0].to(self.device)
        added = drone.ADDED_MASS_0[:3].to(self.device)
        self.M_lin = (m + added).view(1, 3)                          # (1,3)

        rotor_rot_w = drone.rotors_view.get_world_poses()[1].reshape(*drone.shape, self.M, 4)
        axis_w = quat_axis(rotor_rot_w.flatten(end_dim=-2), axis=0).unflatten(0, (*drone.shape, self.M))
        base_rot = drone.rot.reshape(*drone.shape, 1, 4).expand(*drone.shape, self.M, 4)
        axis_b = quat_rotate_inverse(base_rot.reshape(-1, 4), axis_w.reshape(-1, 3)).reshape(*drone.shape, self.M, 3)
        self.Blin = axis_b[0, 0].to(self.device)                     # (M,3) body-frame thrust axes

    @torch.no_grad()
    def filter(self, drone_state: torch.Tensor, obstacle_pos: torch.Tensor, u_rl: torch.Tensor,
               active_mask=None, obstacle_vel=None, blend=None) -> torch.Tensor:
        # active_mask/obstacle_vel/blend accepted for call-site compatibility with
        # MPPIExactShield; this coarse variant keeps its own distance gate and ignores them.
        E = drone_state.shape[0]
        p0 = drone_state[..., 0:3].reshape(E, 3)
        rot = drone_state[..., 3:7].reshape(E, 4)
        v0 = drone_state[..., 7:10].reshape(E, 3)
        p_o = obstacle_pos.reshape(E, -1, 3)                           # [E,N_obs,3]
        u0 = u_rl.reshape(E, self.M)

        # world-frame per-rotor acceleration directions G[E,M,3]: a_world_thrust = einsum('emd,ekm->ekd', G, a)
        body_acc = self.Blin / self.M_lin                            # (M,3) body accel per unit thrust
        rotM = rot.unsqueeze(1).expand(E, self.M, 4).reshape(-1, 4)
        G = self.k * quat_rotate(rotM, body_acc.unsqueeze(0).expand(E, self.M, 3).reshape(-1, 3)).reshape(E, self.M, 3)

        # sample K action sequences around u_rl (thruster box respected)
        noise = self.sigma * torch.randn(E, self.K, self.N, self.M, device=self.device)
        a_seq = (u0.view(E, 1, 1, self.M) + noise).clamp(-1.0, 1.0)  # [E,K,N,M]

        p = p0.unsqueeze(1).expand(E, self.K, 3).clone()
        v = v0.unsqueeze(1).expand(E, self.K, 3).clone()
        cost = torch.zeros(E, self.K, device=self.device)
        for t in range(self.N):
            a_t = a_seq[:, :, t]                                     # [E,K,M]
            acc = torch.einsum('emd,ekm->ekd', G, a_t) - self.drag * v
            v = v + self.dt * acc
            p = p + self.dt * v
            # multi-obstacle: min distance to ANY obstacle
            d_all = (p.unsqueeze(-2) - p_o.unsqueeze(1)).norm(dim=-1)  # [E,K,N_obs]
            pen = (self.r - d_all.min(dim=-1).values).clamp_min(0.0)   # [E,K]
            cost = cost + self.w_coll * pen * pen
        cost = cost + self.w_track * ((a_seq - u0.view(E, 1, 1, self.M)) ** 2).sum((-1, -2))

        weights = torch.softmax(-(cost - cost.min(dim=1, keepdim=True).values) / self.lam, dim=1)  # [E,K]
        u_safe = (weights.unsqueeze(-1) * a_seq[:, :, 0]).sum(dim=1)  # [E,M]

        # activate only near the CLOSEST obstacle; leave the policy untouched far away
        dist = (p0.unsqueeze(1) - p_o).norm(dim=-1).min(dim=-1).values  # [E]
        active = dist.unsqueeze(-1) < (self.r + self.activation_margin)
        out = torch.where(active, u_safe, u0)
        return out.view_as(u_rl)


class MPPIExactShield:
    """MPPI safety filter using the validated <2% NominalDynamics model (arm A, official-code rollout).

    Rolls out K imagined action sequences through the exact 6-DoF vehicle model (T200 lag + full hydro,
    matched to the sim), scores each by keep-out penetration over the horizon + deviation from u_rl, and
    returns the importance-weighted first action.  Orientation is held over the (short) horizon.
    """
    def __init__(self, drone, nominal, keepout_radius, vehicle_radius,
                 horizon=20, num_samples=128, noise_sigma=0.4, temperature=0.05,
                 w_coll=5.0, w_track=1.0, activation_margin=1.5):
        self.drone = drone
        self.nd = nominal
        self.device = drone.device
        self.r = float(keepout_radius) + float(vehicle_radius)
        self.N, self.K = int(horizon), int(num_samples)
        self.sigma, self.lam = float(noise_sigma), float(temperature)
        self.w_coll, self.w_track = float(w_coll), float(w_track)
        self.margin = float(activation_margin)
        self.M = drone.num_rotors

    @torch.no_grad()
    def filter_corridor(self, drone_state, half_width, u_rl, axis: int = 1, center: float = 0.0,
                        vehicle_radius: float = 0.3, active_mask=None, blend=None, d_hat=None,
                        ref_traj=None, w_ref: float = 0.0, center_action=None):
        """论文②：keep-in 走廊版本。与 filter() 同一套 MPPI，只换代价里的安全项。

        cost = w_coll · Σ_t relu(侵入深度)²  +  w_track · Σ_t ‖a_t − u_rl‖²

        侵入深度 = −(W − |p_axis − c| − r_v)，即预测越过侧壁的部分。w_track 那项保持
        "最小介入"语义（u_safe 不要离策略动作太远），与论文①一致。

        d_hat [E,1,6]：在线估的体系等效外力残差，零阶保持注入前滚。名义模型对未建模
        洋流是错的，不注入的话 MPPI 采样出的"安全"动作同样不可信。
        """
        E, K, N, M = drone_state.shape[0], self.K, self.N, self.M
        if active_mask is not None and not active_mask.any():
            return u_rl
        u0 = u_rl.reshape(E, 1, M)

        def rep(x):
            return x.reshape(E, 1, -1).expand(E, K, -1).clone()
        s = {
            "pos": rep(drone_state[..., 0:3]), "quat": rep(drone_state[..., 3:7]),
            "vel_b": rep(self.drone.vel_b), "throttle": rep(self.drone.throttle),
            "rpm": rep(self.drone.rotor_params["rpm"]),
            "v_prev": rep(self.drone.prev_body_vels), "acc_prev": rep(self.drone.prev_body_acc),
        }
        d_k = rep(d_hat) if d_hat is not None else None                       # [E,K,6]
        W = half_width.reshape(E, 1)                                          # [E,1]
        # 采样中心。默认在 u_rl 上（作为**滤波器**时正确：围绕策略提议做局部安全修正）。
        # MPC-only 基线必须换掉它 —— 否则采样被策略热启动、w_track=0 时 u_rl 仍通过
        # 采样中心影响结果，那就不是"不用 RL 的 MPC"，K1 的对照名不副实。
        c0 = u0 if center_action is None else center_action.reshape(E, 1, M)
        a_seq = (c0.unsqueeze(2) + self.sigma * torch.randn(E, K, N, M, device=self.device)).clamp(-1, 1)

        cost = torch.zeros(E, K, device=self.device)
        for t in range(N):
            s = self.nd.step(s, a_seq[:, :, t], d_hat=d_k)
            off = (s["pos"][..., axis] - center).abs()                        # [E,K]
            pen = (off + vehicle_radius - W).clamp_min(0.0)                   # 预测侵入深度
            cost = cost + self.w_coll * pen * pen
            if ref_traj is not None and w_ref > 0.0:
                # 参考跟踪项：仅 MPC-only 基线(λ≡1)需要。作为**滤波器**时不该有它 ——
                # 跟踪是策略的职责，滤波器只做最小介入的安全修正。
                # ref_traj [E,N,3]，取第 t 步的参考点。
                e = (s["pos"] - ref_traj[:, t].reshape(E, 1, 3)).norm(dim=-1)  # [E,K]
                cost = cost + w_ref * e * e
        cost = cost + self.w_track * ((a_seq - u0.unsqueeze(2)) ** 2).sum((-1, -2))

        w = torch.softmax(-(cost - cost.min(1, keepdim=True).values) / self.lam, dim=1)
        u_safe = (w.unsqueeze(-1) * a_seq[:, :, 0]).sum(1)                    # [E,M]

        if blend is not None:
            lam_b = blend.reshape(E, 1).clamp(0.0, 1.0)
            return (u0.reshape(E, M) + lam_b * (u_safe - u0.reshape(E, M))).view_as(u_rl)
        if active_mask is not None:
            return torch.where(active_mask.reshape(E, 1), u_safe, u0.reshape(E, M)).view_as(u_rl)
        return u_safe.view_as(u_rl)

    def filter(self, drone_state, obstacle_pos, u_rl, active_mask=None, obstacle_vel=None, blend=None):
        """active_mask [E,1] bool (optional): predictive-risk gate from RiskMonitor (component A).
        Replaces the distance gate; if no env is triggered, MPPI is skipped entirely (zero cost).
        obstacle_vel [E,3] (optional): moving obstacle → constant-velocity extrapolation in the rollout.
        blend [E,1] float in [0,1] (optional): risk-proportional soft takeover
        u = u_rl + λ (u_safe − u_rl) instead of the hard switch (pass active_mask = blend > 0)."""
        E, K, N, M = drone_state.shape[0], self.K, self.N, self.M
        if active_mask is not None and not active_mask.any():
            return u_rl
        u0 = u_rl.reshape(E, 1, M)
        p_o = obstacle_pos.reshape(E, -1, 3)                            # [E,N_obs,3]
        v_o = obstacle_vel.reshape(E, -1, 3) if obstacle_vel is not None else None

        def rep(x):  # [E,1,D] -> [E,K,D]
            return x.reshape(E, 1, -1).expand(E, K, -1).clone()
        s = {
            "pos": rep(drone_state[..., 0:3]), "quat": rep(drone_state[..., 3:7]),
            "vel_b": rep(self.drone.vel_b), "throttle": rep(self.drone.throttle),
            "rpm": rep(self.drone.rotor_params["rpm"]),
            "v_prev": rep(self.drone.prev_body_vels), "acc_prev": rep(self.drone.prev_body_acc),
        }
        a_seq = (u0.unsqueeze(2) + self.sigma * torch.randn(E, K, N, M, device=self.device)).clamp(-1, 1)

        cost = torch.zeros(E, K, device=self.device)
        for t in range(N):
            s = self.nd.step(s, a_seq[:, :, t])
            if v_o is not None:
                p_o = p_o + v_o * self.nd.dt                                  # moving obstacle: extrapolate
            # multi-obstacle: min distance to ANY obstacle
            d_all = (s["pos"].unsqueeze(-2) - p_o.unsqueeze(1)).norm(dim=-1)  # [E,K,N_obs]
            pen = (self.r - d_all.min(dim=-1).values).clamp_min(0.0)          # [E,K]
            cost = cost + self.w_coll * pen * pen
        cost = cost + self.w_track * ((a_seq - u0.unsqueeze(2)) ** 2).sum((-1, -2))

        w = torch.softmax(-(cost - cost.min(1, keepdim=True).values) / self.lam, dim=1)
        u_safe = (w.unsqueeze(-1) * a_seq[:, :, 0]).sum(1)                    # [E,M]

        if blend is not None:
            lam = blend.reshape(E, 1).clamp(0.0, 1.0)
            return (u0.reshape(E, M) + lam * (u_safe - u0.reshape(E, M))).view_as(u_rl)
        if blend is not None:
            # risk-proportional soft takeover: u = u_rl + λ (u_safe − u_rl), λ∈[0,1] from component A
            lam_b = blend.reshape(E, 1).clamp(0.0, 1.0)
            return (u0.reshape(E, M) + lam_b * (u_safe - u0.reshape(E, M))).view_as(u_rl)
        if active_mask is not None:
            active = active_mask.reshape(E, 1)                                # A's predictive-risk gate
        else:
            # fallback: distance gate near the CLOSEST obstacle
            p0 = drone_state[..., 0:3].reshape(E, 1, 3)
            dist = (p0 - p_o).norm(dim=-1).min(dim=-1, keepdim=True).values   # [E,1]
            active = dist < (self.r + self.margin)
        return torch.where(active, u_safe, u0.reshape(E, M)).view_as(u_rl)
