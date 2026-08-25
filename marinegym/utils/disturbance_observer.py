"""论文② 的一号技术组件：在线扰动观测器。

论文①假定名义模型是准的（一步误差 <2%），风险监视器可以直接信它。本篇的前提相反 ——
洋流是**未建模**的水动力，名义模型在有流时会系统性偏，预测门控因此不可信。观测器的
作用就是把这个偏差在线估出来，重新让 H 步前滚可信。

方法：残差广义力（不去反解流速 —— 反解要过阻尼模型的非线性，是病态的）

    d_raw,t = M_rb ⊙ (v_b,t − v̂_b,t) / dt

其中 v̂_b,t 是名义模型从 (x_{t−1}, u_{t−1}) 预测的体系速度。展开即
`M_rb·(v_t − v_{t−1})/dt − wrench_nom`，也就是名义模型没解释掉的那部分广义力。

    d̂_t = (1−α)·d̂_{t−1} + α·d_raw,t          一阶低通

前滚时对 d̂ 零阶保持（offset-free MPC 的标准做法）。适用条件：扰动的变化尺度长于
预测窗 H·dt（本篇 H=15 → 0.24 s，阵风上升沿 0.2–0.5 s，边界情形，用 oracle-d̂
消融量化这个近似的损失）。
"""

import torch


class DisturbanceObserver:
    def __init__(self, nominal, alpha: float = 0.1, clip: float = 200.0):
        """alpha：低通系数，时间常数 ≈ dt/alpha（dt=0.016, alpha=0.1 → ≈0.16 s）。
        clip：单步残差的绝对值上限，防止碰撞/reset 那一步的速度跳变把估计带飞。"""
        self.nd = nominal
        self.alpha = float(alpha)
        self.clip = float(clip)
        self.d_hat = None

    def reset(self, env_ids=None):
        if self.d_hat is None:
            return
        if env_ids is None:
            self.d_hat.zero_()
        else:
            self.d_hat[env_ids] = 0.

    @torch.no_grad()
    def update(self, s_prev, action_prev, vel_b_now):
        """s_prev: nd.step 所需的上一步状态 dict；action_prev: 上一步实际执行的动作；
        vel_b_now: 本步仿真真值体系速度。返回 d̂ [...,6]。"""
        v_pred = self.nd.step(s_prev, action_prev)["vel_b"]
        d_raw = self.nd.M_rb * (vel_b_now - v_pred) / self.nd.dt
        d_raw = d_raw.clamp(-self.clip, self.clip)
        if self.d_hat is None or self.d_hat.shape != d_raw.shape:
            self.d_hat = torch.zeros_like(d_raw)
        self.d_hat.lerp_(d_raw, self.alpha)
        return self.d_hat
