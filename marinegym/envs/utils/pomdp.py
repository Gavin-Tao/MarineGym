"""部分可观测化 (POMDP) + 观测历史窗口 —— 论文③。

设计要点见 paper3_mamba_design.md：

* **退化在观测出口施加，不动物理仿真**。所有 arm（mlp / stack / gru / transformer /
  mamba）看到的是**同一个**退化后的观测流，差别只在编码器，这样"增益来自历史信息"
  才是干净的因果结论。
* **屏蔽（置零）而不是删除维度**，让 MDP 列与 POMDP 列的 obs_dim 完全一致 ——
  否则各 arm 在两列之间参数量会变，无法直接对比。信息上二者等价（常数输入不携带信息）。
* **历史在环境层维护**，输出 [E, 1, L, D] 的定长窗口 → 策略是无状态的，
  PPO 的 minibatch 打乱采样 (ppo.py make_batch) 一行都不用改。
"""

import torch


class ObsDegrader:
    """把全观测的 obs 退化成 AUV 真实传感条件下的部分观测。

    slices 由 track.py 按观测拼装顺序算好传进来（见 _set_specs）。
    """

    def __init__(self, cfg, num_envs: int, obs_dim: int, slices: dict, device):
        self.device = device
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.slices = slices

        g = (lambda k, d: cfg.get(k, d)) if cfg is not None else (lambda k, d: d)
        self.drop_linear_vel = bool(g("drop_linear_vel", False))   # DVL 失效
        self.drop_angular_vel = bool(g("drop_angular_vel", False)) # 陀螺失效
        self.obs_noise_std = float(g("obs_noise_std", 0.0))        # 传感器噪声
        self.obs_delay = int(g("obs_delay", 0))                    # 声学链路延迟 (步)
        self.sparse_pos_period = int(g("sparse_pos_period", 1))    # USBL 稀疏定位周期
        self.obs_dropout = float(g("obs_dropout", 0.0))            # 丢包概率
        # ---- 切断前馈通道 ----
        # 观测里原本给了未来 4 个参考点 + 时间编码，等于把"参考轨迹接下来去哪"
        # 直接告诉策略 ⇒ 任务是前馈主导的，单帧就够，任何 POMDP 机制都测不出差异
        # (实测 drift / 延迟 / 去速度 / 稀疏定位 5 种机制全部无效)。
        # 屏蔽之后，单帧观测无法判断参考点沿 8 字轨迹的行进方向
        # (同一相对位置对应两个方向) ⇒ 必须从历史推断。
        self.drop_future_ref = bool(g("drop_future_ref", False))
        self.drop_time_enc = bool(g("drop_time_encoding", False))

        self.enabled = (
            self.drop_linear_vel or self.drop_angular_vel
            or self.obs_noise_std > 0 or self.obs_delay > 0
            or self.sparse_pos_period > 1 or self.obs_dropout > 0
            or self.drop_future_ref or self.drop_time_enc
        )

        # 需要跨步状态的机制才分配 buffer
        if self.obs_delay > 0:
            # 环形队列：存最近 obs_delay+1 帧
            self._delay_buf = torch.zeros(
                num_envs, self.obs_delay + 1, obs_dim, device=device)
            self._delay_ptr = 0
        if self.sparse_pos_period > 1:
            self._held_pos = torch.zeros(num_envs, slices["rpos"][1], device=device)
        if self.obs_dropout > 0:
            self._last_obs = torch.zeros(num_envs, obs_dim, device=device)

    # ------------------------------------------------------------ 静态屏蔽
    def static_mask(self, obs: torch.Tensor) -> torch.Tensor:
        """只施加"传感器缺失"这类无跨步状态的退化。reset 与 __call__ 共用，
        保证 buffer 里存的和实际输出的是同一种量。"""
        if self.drop_linear_vel:
            lo, hi = self.slices["lin_vel"]
            obs[:, lo:hi] = 0.0
        if self.drop_angular_vel:
            lo, hi = self.slices["ang_vel"]
            obs[:, lo:hi] = 0.0
        if self.drop_future_ref and "rpos_future" in self.slices:
            lo, hi = self.slices["rpos_future"]
            obs[:, lo:hi] = 0.0
        if self.drop_time_enc and "time_enc" in self.slices:
            lo, hi = self.slices["time_enc"]
            obs[:, lo:hi] = 0.0
        return obs

    # ---------------------------------------------------------------- reset
    def reset(self, env_ids, obs_flat: torch.Tensor):
        """episode 重置：用当前帧填满所有跨步 buffer，避免零填充引入分布偏移。"""
        if not self.enabled:
            return
        cur = self.static_mask(obs_flat[env_ids].clone())
        if self.obs_delay > 0:
            self._delay_buf[env_ids] = cur.unsqueeze(1).expand(-1, self.obs_delay + 1, -1)
        if self.sparse_pos_period > 1:
            lo, hi = self.slices["rpos"]
            self._held_pos[env_ids] = cur[:, lo:hi]
        if self.obs_dropout > 0:
            self._last_obs[env_ids] = cur

    # ------------------------------------------------------------- __call__
    def __call__(self, obs_flat: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        """obs_flat: [E, D] → [E, D]。progress: [E] 当前 episode 步数（稀疏定位用）。"""
        if not self.enabled:
            return obs_flat
        # --- 1) 传感器缺失：置零对应分量 ---
        obs = self.static_mask(obs_flat.clone())

        # --- 2) USBL 稀疏定位：位置类观测每 k 步才刷新，其余步保持 ---
        if self.sparse_pos_period > 1:
            lo, hi = self.slices["rpos"]
            fresh = (progress.long() % self.sparse_pos_period) == 0   # [E]
            self._held_pos = torch.where(
                fresh.unsqueeze(-1), obs[:, lo:hi], self._held_pos)
            obs[:, lo:hi] = self._held_pos

        # --- 3) 观测噪声 ---
        if self.obs_noise_std > 0:
            obs = obs + torch.randn_like(obs) * self.obs_noise_std

        # --- 4) 丢包：以概率 p 沿用上一帧 ---
        if self.obs_dropout > 0:
            drop = torch.rand(obs.shape[0], device=obs.device) < self.obs_dropout
            obs = torch.where(drop.unsqueeze(-1), self._last_obs, obs)
            self._last_obs = obs.clone()

        # --- 5) 观测延迟：输出 k 步前的帧 ---
        if self.obs_delay > 0:
            n = self.obs_delay + 1
            self._delay_ptr = (self._delay_ptr + 1) % n
            self._delay_buf[:, self._delay_ptr] = obs
            out_ptr = (self._delay_ptr + 1) % n      # 最老的一帧 = 延迟 obs_delay 步
            obs = self._delay_buf[:, out_ptr].clone()

        return obs


class ObsHistory:
    """定长观测历史窗口，GPU 上的环形缓冲。push 后返回 [E, L, D]（时间升序，末位=当前）。

    `stride` —— 窗口按步长采样（帧跳）。这一项在本任务里是必需的：

      控制频率 62.5 Hz（dt=0.016），位置量级 ~2 m，而每步位移仅
      1.6 m/s × 0.016 s ≈ 2.6 cm —— **相邻帧的相对差异只有约 1%**。
      stride=1 的 16 帧窗口只跨越 256 ms，序列模型要从几乎相同的 16 帧里
      分辨 1% 的差异，信噪比极差（实测各序列臂都学不动）。
      stride=k 让窗口跨越 (L-1)*k+1 步，帧间差异放大 k 倍，信号才提得出来。

    环形缓冲容量 = (L-1)*stride + 1，仍然 O(L*stride) 显存，但只有窗口本身进观测。
    """

    def __init__(self, num_envs: int, ctx_len: int, obs_dim: int, device, stride: int = 1):
        self.L = int(ctx_len)
        self.stride = max(1, int(stride))
        assert self.L >= 1
        self.cap = (self.L - 1) * self.stride + 1
        self.buf = torch.zeros(num_envs, self.cap, obs_dim, device=device)
        self.ptr = 0        # 指向最近写入的位置
        # 采样偏移：升序时间 → 距当前 (L-1)*stride, ..., stride, 0 步
        self._back = torch.arange(self.L - 1, -1, -1, device=device) * self.stride

    def reset(self, env_ids, obs_flat: torch.Tensor):
        """重置的 env 用当前帧填满整个缓冲（而不是补零）。"""
        self.buf[env_ids] = obs_flat[env_ids].unsqueeze(1).expand(-1, self.cap, -1)

    def push(self, obs_flat: torch.Tensor) -> torch.Tensor:
        if self.cap == 1:
            return obs_flat.unsqueeze(1)
        self.ptr = (self.ptr + 1) % self.cap
        self.buf[:, self.ptr] = obs_flat
        idx = (self.ptr - self._back) % self.cap      # 升序：最老 → 最新
        return self.buf[:, idx]


class ActuatorDrift:
    """逐 episode 随机、且**不进观测**的执行器增益（论文③ 兜底 POMDP 源）。

    u_applied = gain * u_policy，gain ~ U(lo, hi) 逐 env 逐 episode 重采样。

    物理依据：生物附着、电机磨损、电压下降、推力标定漂移 —— AUV 上都很常见。
    为什么必须要历史：单帧观测里没有 gain，也无法从任何单帧量推出来；
    只有把"下了多少油门"与"实际产生了多少运动"在时间上对照才能辨识出增益。
    这是教科书式的在线系统辨识 POMDP。

    每个推进器/舵面独立采样（per_channel=True）时更难，因为还耦合了方向偏置。
    """

    def __init__(self, cfg, num_envs: int, action_dim: int, device):
        g = (lambda k, d: cfg.get(k, d)) if cfg is not None else (lambda k, d: d)
        rng = g("thrust_gain_range", None)
        self.enabled = rng is not None and list(rng) != [1.0, 1.0]
        self.per_channel = bool(g("thrust_gain_per_channel", True))
        if self.enabled:
            self.lo, self.hi = float(rng[0]), float(rng[1])
            n = action_dim if self.per_channel else 1
            self.gain = torch.ones(num_envs, n, device=device)
            self.reset(torch.arange(num_envs, device=device))

    def reset(self, env_ids):
        if not self.enabled:
            return
        shape = (len(env_ids), self.gain.shape[-1])
        self.gain[env_ids] = torch.rand(shape, device=self.gain.device) \
            * (self.hi - self.lo) + self.lo

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        """actions: [E, 1, A] (或 [E, A]) → 同形状"""
        if not self.enabled:
            return actions
        g = self.gain.view(self.gain.shape[0], *([1] * (actions.dim() - 2)),
                           self.gain.shape[-1]) if actions.dim() > 2 else self.gain
        return (actions * g).clamp(-1.0, 1.0)


class WaveDisturbance:
    """未观测的周期性扰动（涌浪诱导力）—— 论文③ 的"理论上保证需要记忆"的 POMDP。

        u_applied = clip(u_policy + A · sin(ω t + φ), -1, 1)

    A / ω / φ 逐 env 逐 episode 随机采样，**不进入观测**。

    为什么这个一定需要历史（与之前 6 种失效机制的本质区别）：

      * 之前的机制（增益漂移/延迟/去速度/稀疏定位/切断前馈）都只是让策略
        "看得少一点"，但"追当前目标点"这个无记忆策略依然近似最优 ——
        目标持续可见，比例控制就够。
      * 周期扰动不同：它是一个**时变的、有确定相位的外力**。反应式控制只能
        在误差出现后再纠正，永远滞后半个周期；要抵消它必须**预测下一时刻的
        扰动值**，而这需要从历史估计 (A, ω, φ)。单帧观测里根本不含相位信息
        —— 同一个瞬时状态可以对应扰动的上升沿或下降沿。

    物理依据：近水面 AUV 受涌浪诱导的振荡力，是水下作业中最常见的周期扰动。
    """

    def __init__(self, cfg, num_envs: int, action_dim: int, device):
        g = (lambda k, d: cfg.get(k, d)) if cfg is not None else (lambda k, d: d)
        self.amp = float(g("wave_amp", 0.0))                  # 幅值(动作单位, [-1,1] 量程)
        w = g("wave_freq_range", [2.0, 8.0])                  # 角频率 rad/s
        self.enabled = self.amp > 0.0
        self.device = device
        if self.enabled:
            self.wlo, self.whi = float(w[0]), float(w[1])
            self.per_channel = bool(g("wave_per_channel", True))
            n = action_dim if self.per_channel else 1
            self.n = n
            self.omega = torch.zeros(num_envs, n, device=device)
            self.phase = torch.zeros(num_envs, n, device=device)
            self.amp_e = torch.zeros(num_envs, n, device=device)
            self.reset(torch.arange(num_envs, device=device))

    def reset(self, env_ids):
        if not self.enabled:
            return
        shape = (len(env_ids), self.n)
        dev = self.device
        self.omega[env_ids] = torch.rand(shape, device=dev) * (self.whi - self.wlo) + self.wlo
        self.phase[env_ids] = torch.rand(shape, device=dev) * (2 * torch.pi)
        # 幅值也随机（0.5A ~ A），避免策略把幅值当常数背下来
        self.amp_e[env_ids] = (0.5 + 0.5 * torch.rand(shape, device=dev)) * self.amp

    def __call__(self, actions: torch.Tensor, t_sec: torch.Tensor) -> torch.Tensor:
        """actions: [E, 1, A] 或 [E, A]；t_sec: [E] 当前 episode 已过秒数。"""
        if not self.enabled:
            return actions
        t = t_sec.reshape(-1, 1)                               # [E,1]
        d = self.amp_e * torch.sin(self.omega * t + self.phase)  # [E,n]
        if actions.dim() > 2:
            d = d.view(d.shape[0], *([1] * (actions.dim() - 2)), d.shape[-1])
        return (actions + d).clamp(-1.0, 1.0)
