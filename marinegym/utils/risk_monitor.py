"""
Component A of the Predict→Protect→Internalize pipeline: predictive risk monitor.

Rolls the policy's OWN proposed action forward H steps through the validated (<2% one-step
error) NominalDynamics model — a SINGLE rollout per env (vs MPPI's K samples), so it is
cheap enough to run every control step.  Outputs:

  min_clear_pred [E,1] : predicted minimum clearance to the keep-out surface over the horizon
  risk           [E,1] : bounded risk scalar in [0,1] for the policy observation
                         (risk = clip((risk_norm - min_clear_pred)/risk_norm, 0, 1))

Used two ways:
  1. gate component B (MPPI filter) — trigger only when min_clear_pred < threshold, replacing
     the crude distance gate (better timing, and B is skipped entirely on safe batches);
  2. feed `risk` into the policy observation (one step delayed), making the policy risk-aware.
"""

import torch


class RiskMonitor:
    def __init__(self, drone, nominal, keepout_radius, vehicle_radius,
                 horizon: int = 15, threshold: float = 0.3, risk_norm: float = 2.0):
        self.drone = drone
        self.nd = nominal
        self.r = float(keepout_radius) + float(vehicle_radius)
        self.H = int(horizon)
        self.threshold = float(threshold)
        self.risk_norm = float(risk_norm)

    @torch.no_grad()
    def assess(self, drone_state: torch.Tensor, obstacle_pos: torch.Tensor, u_rl: torch.Tensor,
               obstacle_vel: torch.Tensor = None):
        """
        drone_state [E,1,S], obstacle_pos [E,3] (closest sphere), u_rl [E,1,M],
        obstacle_vel [E,3] (optional, moving obstacle → constant-velocity extrapolation)
        returns (min_clear_pred [E,1], risk [E,1], trigger [E,1] bool)
        """
        p_o = obstacle_pos.reshape(-1, 1, 3)
        v_o = obstacle_vel.reshape(-1, 1, 3) if obstacle_vel is not None else None
        s = {
            "pos": drone_state[..., 0:3].clone(), "quat": drone_state[..., 3:7].clone(),
            "vel_b": self.drone.vel_b.clone(), "throttle": self.drone.throttle.clone(),
            "rpm": self.drone.rotor_params["rpm"].clone(),
            "v_prev": self.drone.prev_body_vels.clone(), "acc_prev": self.drone.prev_body_acc.clone(),
        }
        min_clear = torch.full(drone_state.shape[:2], float("inf"), device=drone_state.device)  # [E,1]
        for _ in range(self.H):
            s = self.nd.step(s, u_rl)                                # action held over the horizon
            if v_o is not None:
                p_o = p_o + v_o * self.nd.dt                         # moving obstacle: extrapolate
            clear = (s["pos"] - p_o).norm(dim=-1) - self.r           # [E,1]
            min_clear = torch.minimum(min_clear, clear)
        risk = ((self.risk_norm - min_clear) / self.risk_norm).clamp(0.0, 1.0)
        trigger = min_clear < self.threshold
        return min_clear, risk, trigger

    @torch.no_grad()
    def assess_corridor(self, drone_state, half_width, u_rl, axis: int = 1,
                        center: float = 0.0, vehicle_radius: float = 0.3, d_hat=None):
        """论文②：keep-in 走廊版本。与 assess() 共用同一条前滚，只换 clearance 的定义。

        clearance_k = W − |p_k,axis − c| − r_v      （keep-out 是 ‖p−p_o‖ − r，符号相反）

        d_hat [E,1,6]：在线估计的体系等效外力残差，零阶保持注入前滚 —— 这是本篇相对
        论文①的关键差异：名义模型对未建模洋流是错的，不注入 d̂ 的话 H 步预测本身不可信。
        传 None 即退化成论文①的完美模型假设（消融用）。

        returns (min_clear_pred [E,1], risk [E,1], trigger [E,1] bool)
        """
        s = {
            "pos": drone_state[..., 0:3].clone(), "quat": drone_state[..., 3:7].clone(),
            "vel_b": self.drone.vel_b.clone(), "throttle": self.drone.throttle.clone(),
            "rpm": self.drone.rotor_params["rpm"].clone(),
            "v_prev": self.drone.prev_body_vels.clone(), "acc_prev": self.drone.prev_body_acc.clone(),
        }
        W = half_width.reshape(-1, 1)                                    # [E,1]
        min_clear = torch.full(drone_state.shape[:2], float("inf"), device=drone_state.device)
        for _ in range(self.H):
            s = self.nd.step(s, u_rl, d_hat=d_hat)                       # 动作零阶保持，d̂ 也零阶保持
            off = (s["pos"][..., axis] - center).abs()                   # [E,1]
            min_clear = torch.minimum(min_clear, W - off - vehicle_radius)
        risk = ((self.risk_norm - min_clear) / self.risk_norm).clamp(0.0, 1.0)
        trigger = min_clear < self.threshold
        return min_clear, risk, trigger
