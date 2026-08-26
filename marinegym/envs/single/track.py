
import marinegym.utils.kit as kit_utils
from marinegym.utils.torch import euler_to_quaternion, quat_rotate
import omni.isaac.core.utils.prims as prim_utils
import torch
import torch.distributions as D
from torch.func import vmap

from marinegym.views import ArticulationView, RigidPrimView
from marinegym.envs.isaac_env import AgentSpec, IsaacEnv, _NoopDrawInterface
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
try:
    from omni.isaac.debug_draw import _debug_draw
except ModuleNotFoundError:
    _debug_draw = None
from marinegym.robots.drone import UnderwaterVehicle

from ..utils import lemniscate, scale_time, circle, helical, attach_payload

class Track(IsaacEnv):

    def __init__(self, cfg, headless):
        self.reset_thres = cfg.task.reset_thres
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding
        self.future_traj_steps = int(cfg.task.future_traj_steps)
        assert self.future_traj_steps > 0
        self.intrinsics = cfg.task.intrinsics
        self.wind = cfg.task.wind
        self.mode = cfg.mode
        self.disturbances = cfg.task.get("disturbances", {})
        self.enable_payload = self.disturbances[self.mode]['payload']['enable_payload']
        self.enable_flow = self.disturbances[self.mode]['flow']['enable_flow']
        self.max_flow_velocity = self.disturbances[self.mode]['flow']['max_flow_velocity']
        self.flow_velocity_gaussian_noise = self.disturbances[self.mode]['flow']['flow_velocity_gaussian_noise']

        # keep-out obstacle (new-paper safe-RL step 1: math-only, no USD prim; headless-safe)
        keepout_cfg = cfg.task.get("keepout", None)
        self.keepout_enable = bool(keepout_cfg is not None and keepout_cfg.get("enable", False))
        if self.keepout_enable:
            self.n_obstacles = int(keepout_cfg.get("n_obstacles", 1))             # number of keep-out spheres (≥1)
            self.keepout_radius = float(keepout_cfg.get("radius", 0.5))           # r_o (single value → all same)
            self.vehicle_radius = float(keepout_cfg.get("vehicle_radius", 0.3))   # r_v
            self.keepout_terminate = bool(keepout_cfg.get("terminate_on_collision", False))
            self.spawn_clearance = float(keepout_cfg.get("spawn_clearance", 0.7))  # min spawn clearance to the obstacle
            self.obstacle_in_obs = bool(keepout_cfg.get("obstacle_in_obs", True))  # False → policy obstacle-blind (safety is the filter's job)
            self.reward_obstacle_weight = float(keepout_cfg.get("reward_weight", 2.0))  # avoidance reward → real avoidance task
            self.reward_obstacle_margin = float(keepout_cfg.get("reward_margin", 0.5))   # penalty band around the keep-out surface (m)
            self.shield_cfg = keepout_cfg.get("shield", None)
            self.shield_enable = bool(self.shield_cfg is not None and self.shield_cfg.get("enable", False))
            self.mppi_cfg = keepout_cfg.get("mppi", None)
            self.mppi_enable = bool(self.mppi_cfg is not None and self.mppi_cfg.get("enable", False))
            self.validate_dynamics = bool(keepout_cfg.get("validate", False))
            self.validate_full = bool(keepout_cfg.get("validate_full", False))
            # OOD placement: lateral offset std (m), 0 = on reference trajectory
            self.lateral_offset_std = float(keepout_cfg.get("lateral_offset_std", 0.0))
            # dynamic (moving, intercepting) obstacle
            dyn = keepout_cfg.get("dynamic", None)
            self.dyn_enable = bool(dyn is not None and dyn.get("enable", False))
            if self.dyn_enable:
                self.dyn_speed = tuple(float(x) for x in dyn.get("speed", (0.4, 0.9)))
                self.dyn_intercept = tuple(int(x) for x in dyn.get("intercept_steps", (150, 350)))
                self.dyn_reaim = int(dyn.get("re_aim_period", 200))
            # component A: predictive risk monitor;  component C: safety internalization
            self.risk_cfg = keepout_cfg.get("risk", None)
            self.risk_enable = bool(self.risk_cfg is not None and self.risk_cfg.get("enable", False))
            self.risk_in_obs = bool(self.risk_enable and self.risk_cfg.get("in_obs", True))
            # 解耦：A 是否「跑并喂观测」(risk.enable) 与 A 是否「充当门控」(risk.gate)。
            # gate=false → A 仍进观测，但门控退回几何(当前间隙)，使几何消融与本方法观测一致。
            self.risk_gate = bool(self.risk_enable and self.risk_cfg.get("gate", True))
            self.internalize_weight = float(keepout_cfg.get("internalize_weight", 0.0))
            # adaptive C: w_C = gain * EMA(mean filter engagement), self-extinguishing
            self.internalize_adaptive = bool(keepout_cfg.get("internalize_adaptive", False))
            self.internalize_gain = float(keepout_cfg.get("internalize_gain", 5.0))
            self.internalize_ema = float(keepout_cfg.get("internalize_ema", 0.995))
            # soft takeover: λ blend instead of the binary trigger.
            # λ from A's PREDICTED clearance when risk monitor is on; else from CURRENT distance (pure-B-soft ablation)
            self.mppi_soft = bool(self.mppi_enable and self.mppi_cfg.get("soft_blend", False))
            self.mppi_soft_lo = float(self.mppi_cfg.get("soft_lo", 0.0)) if self.mppi_cfg else 0.0
            self.mppi_soft_hi = float(self.mppi_cfg.get("soft_hi", 0.6)) if self.mppi_cfg else 0.6
            # 整套实验只允许存在一个门控边界；threshold 与 soft_hi 是同一量的阶跃/斜坡参数化。
            self.gate_boundary = float(self.mppi_soft_hi)
            _thr = float(self.risk_cfg.get("threshold", 0.3)) if self.risk_cfg else self.gate_boundary
            if self.risk_enable and abs(_thr - self.gate_boundary) > 1e-9:
                import logging
                logging.warning("keepout: risk.threshold=%.3g != mppi.soft_hi=%.3g; "
                                "使用 soft_hi 作为唯一门控边界。", _thr, self.gate_boundary)
            # pure-A-soft ablation: A's λ + cheap reactive REPULSION action (no MPPI) — "field" actor
            fld = keepout_cfg.get("field", None)
            self.field_enable = bool(fld is not None and fld.get("enable", False))
            self.field_gain = float(fld.get("gain", 2.0)) if fld else 2.0
            # relax tracking-error termination so the vehicle may legally divert around the obstacle
            self.reset_thres = float(keepout_cfg.get("reset_thres", max(self.reset_thres, 1.5)))
        else:
            self.shield_enable = False
            self.mppi_enable = False
            self.validate_dynamics = False
            self.validate_full = False
            self.n_obstacles = 1
            self.lateral_offset_std = 0.0
            self.risk_enable = False
            self.risk_in_obs = False
            self.internalize_weight = 0.0
            self.internalize_adaptive = False
            self.mppi_soft = False
            self.risk_gate = False
            self.gate_boundary = 0.6
            self.field_enable = False
            self.dyn_enable = False

        # ── 论文② (flow-safety)：keep-in 走廊 + 阵风。与 keepout 完全独立，默认全关 ──
        # 走廊是**世界系**的（海沟侧壁），不是以参考点为中心的管道：
        # 若以参考点为中心，clearance = R − ‖p − p_ref‖ 就是跟踪误差的单调函数，
        # 门控信号与奖励主项是同一个量（正是"拿 tracking error 当 risk"被否决的理由）。
        # 世界系走廊让"贴壁裕度"与"跟踪误差"成为两个不同的量。
        cor_cfg = cfg.task.get("corridor", None)
        self.corridor_enable = bool(cor_cfg is not None and cor_cfg.get("enable", False))
        if self.corridor_enable:
            self.cor_axis = int(cor_cfg.get("axis", 1))              # 0=x 1=y 2=z，受限轴
            self.cor_margin = float(cor_cfg.get("margin", 0.35))     # 半宽 = max|ref_axis| + margin
            _hw = cor_cfg.get("half_width", None)
            self.cor_half_width_abs = float(_hw) if _hw is not None else None
            self.cor_vehicle_radius = float(cor_cfg.get("vehicle_radius", 0.3))
            self.cor_reward_weight = float(cor_cfg.get("reward_weight", 2.0))
            self.cor_reward_margin = float(cor_cfg.get("reward_margin", 0.3))
            self.cor_in_obs = bool(cor_cfg.get("in_obs", True))
            self.cor_terminate = bool(cor_cfg.get("terminate_on_violation", False))
            self.cor_freeze_yaw = bool(cor_cfg.get("freeze_traj_yaw", True))
            # 违约不终止 → 各方法曝光量相同、违约率可比（沿用论文①禁区非碰撞体的口径）。
            # 因此必须放宽跟踪误差终止，否则一飘出去就 reset，曝光不等。
            self.reset_thres = float(cor_cfg.get("reset_thres", max(self.reset_thres, 3.0)))
        else:
            self.cor_in_obs = False
            self.cor_freeze_yaw = False

        gust_cfg = cfg.task.get("gust", None)
        self.gust_enable = bool(gust_cfg is not None and gust_cfg.get("enable", False))
        if self.gust_enable:
            self.gust_speed = list(gust_cfg.get("speed", [0.6, 1.0]))
            self.gust_ramp = list(gust_cfg.get("ramp", [0.2, 0.5]))     # 上升时间 (s)
            self.gust_hold = list(gust_cfg.get("hold", [1.0, 2.0]))     # 保持时间 (s)
            self.gust_period = int(gust_cfg.get("period", 200))          # 重发周期 (步)
            self.gust_axis = int(gust_cfg.get("axis", 1))
            self.gust_aim = str(gust_cfg.get("aim", "curvature"))
            self.gust_lookahead = list(gust_cfg.get("lookahead", [60, 200]))

        # 安全栈 A→λ→B（走廊版）。只有 corridor 开着才有意义。
        saf = cfg.task.get("safety", None)
        _s = (saf if saf is not None else {})
        _dob = _s.get("dobs", {}) or {}
        _rsk = _s.get("risk", {}) or {}
        _mpp = _s.get("mppi", {}) or {}
        self.dobs_enable = bool(self.corridor_enable and _dob.get("enable", True))
        self.dobs_alpha = float(_dob.get("alpha", 0.1))
        self.dobs_clip = float(_dob.get("clip", 200.0))
        self.dobs_oracle = bool(_dob.get("oracle", False))
        self.dobs_zero = bool(_dob.get("zero", False))
        self.cor_risk_enable = bool(self.corridor_enable and _rsk.get("enable", False))
        self.cor_risk_H = int(_rsk.get("horizon", 15))
        self.cor_risk_thr = float(_rsk.get("threshold", 0.6))
        self.cor_risk_gate = bool(_rsk.get("gate", True))
        self.cor_risk_norm = float(_rsk.get("risk_norm", 2.0))
        self.cor_risk_in_obs = bool(self.cor_risk_enable and _rsk.get("in_obs", True))
        self.cor_mppi_enable = bool(self.corridor_enable and _mpp.get("enable", False))
        self.cor_mppi_soft = bool(_mpp.get("soft", True))
        self.cor_mppi_lo = float(_mpp.get("soft_lo", 0.0))
        self.cor_gate_boundary = float(_mpp.get("soft_hi", 0.6))
        self.cor_mppi_w_ref = float(_mpp.get("w_ref", 0.0))
        self.cor_mppi_center = str(_mpp.get("center", "rl"))   # rl | prev | zero，见 force_lambda 分支
        _fl = _mpp.get("force_lambda", None)
        self.cor_force_lambda = float(_fl) if _fl is not None else None   # MPC-only 基线设 1.0
        self.cor_mppi_cfg = _mpp
        self.k4_enable = bool(_s.get("k4", False))   # K4 诊断：同时算 d̂=0 / est / oracle 三种预测
        # 流速观测必须**独立于 gust.enable**：否则 calm 场景(关阵风)的观测维度比训练时少 1，
        # 同一条 checkpoint 加载直接 shape mismatch，"一条策略跑遍所有场景"的设计就废了。
        # 无阵风时这一维恒为 0，不影响策略。
        self.flow_in_obs = bool(self.corridor_enable and _s.get("flow_in_obs", True))
        self.flow_obs_axis = self.gust_axis if self.gust_enable else int(getattr(self, "cor_axis", 1))

        super().__init__(cfg, headless)

        self.drone.initialize()
        if self.enable_payload:
            payload_cfg = self.disturbances[self.mode]['payload']
            self.payload_z_dist = D.Uniform(
                torch.tensor([payload_cfg["z"][0]], device=self.device),
                torch.tensor([payload_cfg["z"][1]], device=self.device)
            )
            self.payload_mass_dist = D.Uniform(
                torch.tensor([payload_cfg["mass"][0]], device=self.device),
                torch.tensor([payload_cfg["mass"][1]], device=self.device)
            )
            self.payload = RigidPrimView(
                f"/World/envs/env_*/{self.drone.name}_*/payload",
                reset_xform_properties=False,
                shape=(-1, self.drone.n)
            )
            self.payload.initialize()

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )
        # 默认 yaw 在 [0,2π) 随机 → 轨迹整体绕 z 轴随机转向。世界系走廊(侧壁)要求轨迹
        # 朝向固定，否则"侧壁"相对轨迹的位置每个 episode 都不同，难度不可控。
        _yaw_hi = 0. if self.cor_freeze_yaw else 2.
        self.traj_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., _yaw_hi], device=self.device) * torch.pi
        )
        self.traj_c_dist = D.Uniform(
            torch.tensor(-0.6, device=self.device),
            torch.tensor(0.6, device=self.device)
        )
        _tsm = float(self.cfg.task.get("traj_scale_mult", 1.0))  # enlarge workspace so obstacle avoidance is feasible
        self.traj_scale_dist = D.Uniform(
            torch.tensor([1.4, 1.4, 0.8], device=self.device) * _tsm,
            torch.tensor([2.6, 2.6, 1.2], device=self.device) * _tsm
        )
        # 轨迹角速度：参考点线速度 ∝ traj_scale × traj_w。
        # 单独放大 traj_scale_mult 会同比例放大参考速度，可能超出载具能力使任务不可跟踪；
        # 需要"大工作空间 + 可跟踪速度"时，按比例调小本区间即可。
        _tw = cfg.task.get("traj_w_range", [0.7, 0.9])
        self.traj_w_dist = D.Uniform(
            torch.tensor(float(_tw[0]), device=self.device),
            torch.tensor(float(_tw[1]), device=self.device)
        )
        self.origin = torch.tensor([0., 0., 2.], device=self.device)

        self.traj_t0 = torch.pi / 2
        self.traj_c = torch.zeros(self.num_envs, device=self.device)
        self.traj_scale = torch.zeros(self.num_envs, 3, device=self.device)
        self.traj_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.traj_w = torch.ones(self.num_envs, device=self.device)

        self.target_pos = torch.zeros(self.num_envs, self.future_traj_steps, 3, device=self.device)
        if self.keepout_enable:
            self.obstacle_pos = torch.zeros(self.num_envs, self.n_obstacles, 3, device=self.device)
            self.obstacle_vel = torch.zeros(self.num_envs, self.n_obstacles, 3, device=self.device)
            self._collision = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        if self.shield_enable:
            if self.drone.num_rotors == self.drone.action_spec.shape[-1]:
                from marinegym.utils.action_inspector import HOCBFShield
                sh = self.shield_cfg
                self.shield = HOCBFShield(
                    self.drone, self.keepout_radius, self.vehicle_radius,
                    alpha1=float(sh.get("alpha1", 2.0)), alpha2=float(sh.get("alpha2", 2.0)),
                    drift_bound=float(sh.get("drift_bound", 5.0)), thrust_gain=float(sh.get("thrust_gain", 40.0)),
                    activation_margin=float(sh.get("activation_margin", 1.0)),
                )
            else:
                import logging
                logging.warning("action inspector: HOCBF shield needs a pure-thruster (T200) vehicle "
                                f"(num_rotors==action_dim); got num_rotors={self.drone.num_rotors}, "
                                f"action_dim={self.drone.action_spec.shape[-1]}. Disabling shield.")
                self.shield_enable = False
        if self.mppi_enable:
            if self.drone.num_rotors == self.drone.action_spec.shape[-1]:
                mp = self.mppi_cfg
                substeps = int(self.cfg.sim.get("substeps", 1))
                if bool(mp.get("exact", False)):
                    # accurate MPPI: rolls out the validated <2% NominalDynamics model
                    from marinegym.utils.nominal_dynamics import NominalDynamics
                    from marinegym.utils.mppi_filter import MPPIExactShield
                    nd = self._nd = NominalDynamics(self.drone, self.dt * substeps)
                    self.mppi = MPPIExactShield(
                        self.drone, nd, self.keepout_radius, self.vehicle_radius,
                        horizon=int(mp.get("horizon", 20)), num_samples=int(mp.get("num_samples", 128)),
                        noise_sigma=float(mp.get("noise_sigma", 0.4)), temperature=float(mp.get("temperature", 0.05)),
                        w_coll=float(mp.get("w_coll", 5.0)), w_track=float(mp.get("w_track", 1.0)),
                        activation_margin=float(mp.get("activation_margin", 1.5)),
                    )
                else:
                    from marinegym.utils.mppi_filter import MPPIShield
                    self.mppi = MPPIShield(
                        self.drone, self.keepout_radius, self.vehicle_radius, dt=self.dt * substeps,
                        horizon=int(mp.get("horizon", 20)), num_samples=int(mp.get("num_samples", 128)),
                        noise_sigma=float(mp.get("noise_sigma", 0.4)), temperature=float(mp.get("temperature", 0.05)),
                        w_coll=float(mp.get("w_coll", 5.0)), w_track=float(mp.get("w_track", 1.0)),
                        drag=float(mp.get("drag", 1.0)), thrust_gain=float(mp.get("thrust_gain", 40.0)),
                        activation_margin=float(mp.get("activation_margin", 1.5)),
                    )
            else:
                import logging
                logging.warning("MPPI filter needs a pure-thruster (T200) vehicle; disabling.")
                self.mppi_enable = False
        if self.risk_enable:
            if self.drone.num_rotors == self.drone.action_spec.shape[-1]:
                from marinegym.utils.risk_monitor import RiskMonitor
                if not hasattr(self, "_nd"):
                    from marinegym.utils.nominal_dynamics import NominalDynamics
                    self._nd = NominalDynamics(self.drone, self.dt * int(self.cfg.sim.get("substeps", 1)))
                rc = self.risk_cfg
                self.risk_monitor = RiskMonitor(
                    self.drone, self._nd, self.keepout_radius, self.vehicle_radius,
                    horizon=int(rc.get("horizon", 15)), threshold=float(rc.get("threshold", 0.3)),
                    risk_norm=float(rc.get("risk_norm", 2.0)),
                )
            else:
                import logging
                logging.warning("risk monitor needs a pure-thruster (T200) vehicle; disabling.")
                self.risk_enable = False
                self.risk_in_obs = False
                self.risk_gate = False
        if self.field_enable:
            # field actor needs thruster axes; reuse (or build) NominalDynamics for its Blin extraction
            if not hasattr(self, "_nd"):
                from marinegym.utils.nominal_dynamics import NominalDynamics
                self._nd = NominalDynamics(self.drone, self.dt * int(self.cfg.sim.get("substeps", 1)))
        if self.keepout_enable:
            # component A/C buffers (defined regardless so obs/reward code is branch-free)
            self._risk = torch.zeros(self.num_envs, 1, 1, device=self.device)          # delayed risk obs
            self._risk_trigger = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
            self._correction = torch.zeros(self.num_envs, 1, device=self.device)       # ‖u_safe−u_rl‖²
            # efficiency metrics: flown vs reference path length (detour), clearance beyond the safe margin
            self._flown = torch.zeros(self.num_envs, 1, device=self.device)
            self._ref_len = torch.zeros(self.num_envs, 1, device=self.device)
            self._last_p = torch.zeros(self.num_envs, 3, device=self.device)
            self._last_ref = torch.zeros(self.num_envs, 3, device=self.device)
            self.safe_margin = float(keepout_cfg.get("safe_margin", 0.5)) if keepout_cfg else 0.5
            self._filter_engage = torch.zeros(self.num_envs, 1, device=self.device)    # λ (soft) or 0/1 trigger
            self._engage_ema = torch.zeros((), device=self.device)                     # adaptive C: mean engagement EMA
            self._wC = torch.zeros((), device=self.device)                             # adaptive C: current weight
        if self.corridor_enable:
            E = self.num_envs
            self._cor_hw = torch.zeros(E, 1, device=self.device)          # 该 episode 的走廊半宽
            self._cor_clear = torch.zeros(E, 1, device=self.device)       # 到最近侧壁的裕度
            self._cor_violate = torch.zeros(E, 1, dtype=torch.bool, device=self.device)
        if self.gust_enable:
            E = self.num_envs
            self._gust_t0 = torch.zeros(E, 1, device=self.device)         # 起风步
            self._gust_amp = torch.zeros(E, 1, device=self.device)        # 幅值 (m/s)，带符号
            self._gust_ramp = torch.ones(E, 1, device=self.device)        # 上升步数
            self._gust_hold = torch.ones(E, 1, device=self.device)        # 保持步数
            self._gust_next = torch.zeros(E, 1, device=self.device)       # 下次重发的步（独立计时）
            # 阵风用**独立的 generator**，不走全局 RNG。否则 MPPI 的 torch.randn 会
            # 消耗全局流，含滤波器的格子与 PPO 格子遇到的阵风序列就不同 —— 50 个
            # episode 下几个百分点的违约率差可能纯粹来自扰动采样差异。独立发生器让
            # 各格面对完全相同的扰动序列，消融变成配对比较。
            self._gust_gen = torch.Generator(device=self.device)
            self._gust_gen.manual_seed(int(self.cfg.get("seed", 0)) * 100003 + 7919)
        if self.corridor_enable and (self.dobs_enable or self.cor_risk_enable or self.cor_mppi_enable):
            from marinegym.utils.nominal_dynamics import NominalDynamics
            _dt = self.dt * int(self.cfg.sim.get("substeps", 1))
            self._cnd = NominalDynamics(self.drone, _dt)
            E = self.num_envs
            self._d_hat = torch.zeros(E, 1, 6, device=self.device)
            self._cor_risk = torch.zeros(E, 1, 1, device=self.device)      # 进观测的 risk 标量（延迟一步）
            self._cor_lam = torch.zeros(E, 1, device=self.device)          # λ 接管系数
            self._cor_u_rl = None
            self._cor_corr = torch.zeros(E, 1, device=self.device)         # ‖u_safe − u_rl‖²
            self._cor_prev = None                                          # 上一步(物理步前)的状态
            self._cor_prev_a = None
            if self.dobs_enable:
                from marinegym.utils.disturbance_observer import DisturbanceObserver
                self._dobs = DisturbanceObserver(self._cnd, alpha=self.dobs_alpha, clip=self.dobs_clip)
            if self.cor_risk_enable:
                from marinegym.utils.risk_monitor import RiskMonitor
                self._cor_rm = RiskMonitor(self.drone, self._cnd, 0.0, 0.0, horizon=self.cor_risk_H,
                                           threshold=self.cor_risk_thr, risk_norm=self.cor_risk_norm)
            if self.cor_mppi_enable:
                from marinegym.utils.mppi_filter import MPPIExactShield
                m = self.cor_mppi_cfg
                self._cor_mppi = MPPIExactShield(
                    self.drone, self._cnd, 0.0, 0.0,
                    horizon=int(m.get("horizon", 20)), num_samples=int(m.get("num_samples", 128)),
                    noise_sigma=float(m.get("noise_sigma", 0.4)), temperature=float(m.get("temperature", 0.05)),
                    w_coll=float(m.get("w_wall", 5.0)), w_track=float(m.get("w_track", 1.0)))
        if self.validate_dynamics:
            from marinegym.utils.nominal_dynamics import NominalDynamics
            self.nominal = NominalDynamics(self.drone, self.dt * int(self.cfg.sim.get("substeps", 1)))
            # C3: 给名义模型注入参数误差（仅影响 f_nom，不动仿真器真值）
            _ms = float(keepout_cfg.get("nom_scale_mass", 1.0))
            _ds = float(keepout_cfg.get("nom_scale_drag", 1.0))
            _ts = float(keepout_cfg.get("nom_scale_thrust", 1.0))
            for _nd_ in {id(x): x for x in (self.nominal, getattr(self, "_nd", None)) if x is not None}.values():
                if _ms != 1.0: _nd_.M_rb = _nd_.M_rb * _ms
                if _ds != 1.0: _nd_.Dl = _nd_.Dl * _ds; _nd_.Dq = _nd_.Dq * _ds
                if _ts != 1.0: _nd_.fc_scale = _nd_.fc_scale * _ts
            self._val_count = 0

        self.alpha = 0.8

        self.draw = (
            _debug_draw.acquire_debug_draw_interface()
            if _debug_draw is not None
            else _NoopDrawInterface()
        )

    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = UnderwaterVehicle.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )
        from marinegym.robots.robot import ASSET_PATH
        import omni.isaac.core.utils.stage as stage_utils
        stage_utils.add_reference_to_stage(usd_path= ASSET_PATH + "/usd/worlds/EmptyMarine.usd",prim_path="/World/defaultGroundPlane")


        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]
        if self.enable_payload:
            attach_payload(drone_prim.GetPath().pathString)        
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        obs_dim = drone_state_dim + 3 * (self.future_traj_steps-1)
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_dim += self.time_encoding_dim
        if self.intrinsics:
            obs_dim += sum(spec.shape[-1] for name, spec in self.drone.info_spec.items())
        if self.keepout_enable and self.obstacle_in_obs:
            obs_dim += 3  # obstacle-relative position (p_o - p) — always 3 dims (closest obstacle)
            if self.dyn_enable:
                obs_dim += 3  # obstacle velocity (fair: the policy gets full information too)
        if self.risk_in_obs:
            obs_dim += 1  # component A: predicted-risk scalar (one step delayed)
        if self.cor_in_obs:
            obs_dim += 2  # 论文②: 到两侧壁的有符号裕度（左/右各一维，策略才知道自己偏向哪边）
        if self.flow_in_obs:
            obs_dim += 1  # 论文②: 当前流速在受限轴上的分量（公平性：策略拿得到扰动信息）
                          # 与 gust.enable 解耦，见 flow_in_obs 的说明
        if self.cor_risk_in_obs:
            obs_dim += 1  # 论文②: A 的预测风险标量（延迟一步）

        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": UnboundedContinuousTensorSpec((1, obs_dim))
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": {
                "action": self.drone.action_spec.unsqueeze(0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": {
                "reward": UnboundedContinuousTensorSpec((1, 1))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )
        stats_spec_dict = {
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "tracking_error": UnboundedContinuousTensorSpec(1),
            "tracking_error_ema": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
        }
        if self.keepout_enable:
            stats_spec_dict["collision"] = UnboundedContinuousTensorSpec(1)       # per-episode 0/1 (violation rate)
            stats_spec_dict["min_obstacle_dist"] = UnboundedContinuousTensorSpec(1)  # min surface clearance (m)
            stats_spec_dict["filter_activation"] = UnboundedContinuousTensorSpec(1)  # fraction of steps B engaged
            stats_spec_dict["correction"] = UnboundedContinuousTensorSpec(1)         # mean ‖u_safe−u_rl‖²
            stats_spec_dict["detour_ratio"] = UnboundedContinuousTensorSpec(1)       # flown / reference path length (≥1, →1 best)
            stats_spec_dict["over_clearance"] = UnboundedContinuousTensorSpec(1)     # min_dist − safe_margin during encounters (small>0 best)
            stats_spec_dict["internalize_w"] = UnboundedContinuousTensorSpec(1)      # adaptive C weight w_C(t)
        if self.corridor_enable:
            # headline: wall_violation = 该 episode 是否触壁(0/1)；违约不终止 → 各方法曝光相同
            stats_spec_dict["wall_violation"] = UnboundedContinuousTensorSpec(1)
            stats_spec_dict["wall_viol_frac"] = UnboundedContinuousTensorSpec(1)   # 触壁步数占比（曝光归一化）
            stats_spec_dict["wall_depth"] = UnboundedContinuousTensorSpec(1)       # 累计侵入深度 Σmax(0,−d)·dt
            stats_spec_dict["min_wall_dist"] = UnboundedContinuousTensorSpec(1)    # 全 episode 最小侧壁裕度 (m)
            stats_spec_dict["filter_lambda"] = UnboundedContinuousTensorSpec(1)    # 平均接管系数 λ（占空比故事）
            stats_spec_dict["wall_correction"] = UnboundedContinuousTensorSpec(1)  # 平均 ‖u_safe−u_rl‖²
            stats_spec_dict["d_hat_norm"] = UnboundedContinuousTensorSpec(1)       # ‖d̂‖ 观测器输出量级
        if self.gust_enable:
            stats_spec_dict["gust_speed"] = UnboundedContinuousTensorSpec(1)       # 当前阵风幅值 |v_c| EMA
        stats_spec = CompositeSpec(stats_spec_dict).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _aim_obstacle(self, env_ids):
        """Dynamic obstacle: launch from the side so it INTERCEPTS the vehicle's reference position
        k steps from now (arrives exactly when the vehicle does — reactive avoidance is not enough).
        Returns (start [n,3], vel [n,3]), env-local frame."""
        n = len(env_ids)
        k_lo, k_hi = self.dyn_intercept
        traj = self._compute_traj(k_hi + 2, env_ids, step_size=1.)              # [n,K,3]
        k = torch.randint(k_lo, k_hi, (n,), device=self.device)
        ar = torch.arange(n, device=self.device)
        p_hit = traj[ar, k]                                                     # future reference point
        tang = traj[ar, k + 1] - traj[ar, k]
        tang = tang / (tang.norm(dim=-1, keepdim=True) + 1e-6)
        up = torch.zeros(n, 3, device=self.device); up[:, 2] = 1.0
        side = torch.cross(tang, up, dim=-1)
        side = side / (side.norm(dim=-1, keepdim=True) + 1e-6)
        side = side * (torch.randint(0, 2, (n, 1), device=self.device) * 2 - 1)  # random left/right
        v = self.dyn_speed[0] + (self.dyn_speed[1] - self.dyn_speed[0]) * torch.rand(n, 1, device=self.device)
        T = (k.float() * self.dt).unsqueeze(-1)                                  # seconds to impact
        return p_hit + side * (v * T), -side * v

    def _place_one_obstacle(self, env_ids, spawn_local):
        """Place a single obstacle on the reference trajectory, spawn-safe, with optional lateral offset."""
        n = len(env_ids)
        K = 40
        traj = self._compute_traj(K, env_ids, step_size=2.)        # [n,K,3]
        dfs = (traj - spawn_local.unsqueeze(1)).norm(dim=-1)        # [n,K]
        target = (self.keepout_radius + self.vehicle_radius) + self.spawn_clearance
        reached = dfs >= target                                    # [n,K]
        rand = torch.rand(n, K, device=self.device)
        rand[~reached] = -1.0
        idx = torch.where(
            reached.any(1), rand.argmax(1),
            torch.full((n,), K - 1, device=self.device, dtype=torch.long)
        )
        obstacle = traj[torch.arange(n, device=self.device), idx]  # [n,3]
        # lateral offset: random perpendicular push (B3 OOD)
        if self.lateral_offset_std > 0:
            # generate random unit vectors in the horizontal plane
            angle = 2 * torch.pi * torch.rand(n, device=self.device)
            dirs = torch.stack([torch.cos(angle), torch.sin(angle), torch.zeros(n, device=self.device)], dim=-1)
            offset = self.lateral_offset_std * torch.randn(n, 1, device=self.device).abs() * dirs
            obstacle = obstacle + offset
        return obstacle

    # ── 论文②：keep-in 走廊 ────────────────────────────────────────────────
    def _ref_axis_extent(self, env_ids):
        """该 episode 参考轨迹在受限轴上的最大 |偏移|（相对 origin）。

        用解析的整周期采样，而不是 `_compute_traj`：后者依赖 progress_buf，而 reset
        时 progress_buf 的清零时机不由本函数保证。lemniscate 在 t 上以 2π 为周期，
        直接在 [0,2π] 上密采即可，与 traj_w / traj_t0 无关。
        """
        n, S = len(env_ids), 361
        tt = torch.linspace(0., 2 * torch.pi, S, device=self.device).unsqueeze(0).expand(n, S)
        pl = vmap(lemniscate)(tt, self.traj_c[env_ids])                             # [n,S,3]
        rot = self.traj_rot[env_ids].unsqueeze(1).expand(-1, S, 4)
        pl = vmap(quat_rotate)(rot, pl) * self.traj_scale[env_ids].unsqueeze(1)     # [n,S,3]
        return pl[..., self.cor_axis].abs().amax(dim=1, keepdim=True)               # [n,1]

    def _oracle_d_hat(self):
        """上界消融：用仿真真值流速直接算出流引起的额外水动力，而不是在线估。

        仿真里洋流是通过**相对速度**进入阻尼项的（underwaterVehicle.apply_hydrodynamic_forces
        中 body_vels -= flow_vels_b），所以流引起的等效外力就是
            hydro(v_b − v_flow,b) − hydro(v_b)
        这给出观测器能达到的性能上界，用来量化在线估计的损失。
        """
        from marinegym.utils.torch import quat_rotate_inverse, quaternion_to_euler
        vb = self.drone.vel_b
        q = self.drone.rot
        fw = self.drone.flow_vels
        fb = torch.cat([
            quat_rotate_inverse(q.reshape(-1, 4), fw[..., :3].reshape(-1, 3)).reshape_as(vb[..., :3]),
            quat_rotate_inverse(q.reshape(-1, 4), fw[..., 3:].reshape(-1, 3)).reshape_as(vb[..., 3:]),
        ], dim=-1)
        rpy = quaternion_to_euler(q)
        vp, ap = self.drone.prev_body_vels, self.drone.prev_body_acc
        h_flow, _, _ = self._cnd.hydro_wrench(vb - fb, vp, ap, rpy)
        h_still, _, _ = self._cnd.hydro_wrench(vb, vp, ap, rpy)
        return h_flow - h_still

    def _reset_corridor(self, env_ids):
        if self.cor_half_width_abs is not None:
            self._cor_hw[env_ids] = self.cor_half_width_abs
        else:
            # W = max|ref_axis| + r_v + margin
            # 于是在参考轨迹最贴壁处，**完美跟踪**的载具恰好保留 margin 的裕度：
            #     clearance = W − |y| − r_v = max|ref| + margin − |y|  →  |y|=max|ref| 时 = margin
            # margin 因此是可直接解释的量，也是标定门控边界的基准：
            # 必须 margin > soft_hi，否则完美跟踪时门控就已经常开，λ 永远 >0。
            self._cor_hw[env_ids] = (self._ref_axis_extent(env_ids)
                                     + self.cor_vehicle_radius + self.cor_margin)
        self.stats["min_wall_dist"][env_ids] = 100.0   # 取 min 的累计量，须初始化为大值
        # reset 那一步速度会跳变，残差会是垃圾；清掉 d̂ 与上一步状态，避免污染下一 episode
        if getattr(self, "_dobs", None) is not None:
            self._dobs.reset(env_ids)
        if getattr(self, "_cor_prev", None) is not None:
            self._cor_prev = None

    # ── 论文②：阵风 ───────────────────────────────────────────────────────
    def _launch_gust(self, env_ids, mask):
        """给 env_ids[mask] 发一阵新风。

        瞄准策略 curvature：阵风的**峰值时刻**对准参考轨迹在受限轴上的极值点 ——
        载具在那里正换向（横向推力裕度最低）且最贴近侧壁。这是本篇"必须预测"的场景
        支点，与论文①"瞄准未来参考位置"同构。

        方向按**未来参考点**的所在侧，不是按载具当前位置：后者会形成正反馈跑飞
        （一旦被推偏，之后每阵风都朝同方向推，载具单调飘走）。用参考点则与载具的
        跟踪误差无关，扰动是外生的，各方法面对的是同一个扰动序列。
        """
        idx = env_ids[mask] if mask.dtype == torch.bool else env_ids
        if idx.numel() == 0:
            return
        n = idx.numel()
        lo, hi = self.gust_speed
        amp = torch.rand(n, 1, device=self.device, generator=self._gust_gen) * (hi - lo) + lo
        r_lo, r_hi = self.gust_ramp
        h_lo, h_hi = self.gust_hold
        ramp = ((torch.rand(n, 1, device=self.device, generator=self._gust_gen) * (r_hi - r_lo) + r_lo) / self.dt).round().clamp_min(1)
        self._gust_ramp[idx] = ramp
        self._gust_hold[idx] = ((torch.rand(n, 1, device=self.device, generator=self._gust_gen) * (h_hi - h_lo) + h_lo) / self.dt).round().clamp_min(1)

        # 在 lookahead 窗口内找参考轨迹在受限轴上 |偏移| 最大的那一步
        k_lo, k_hi = self.gust_lookahead
        traj = self._compute_traj(int(k_hi) + 1, idx, step_size=1.)          # [n,K,3]
        off = traj[..., self.gust_axis] - self.origin[self.gust_axis]        # [n,K]
        off[:, :int(k_lo)] = 0.                                              # 窗口下界之前不选
        k_peak = off.abs().argmax(dim=1)                                     # [n]
        ar = torch.arange(n, device=self.device)
        side = torch.sign(off[ar, k_peak]).unsqueeze(-1)
        side = torch.where(side == 0, torch.ones_like(side), side)
        self._gust_amp[idx] = amp * side
        # 让上升沿恰好在 k_peak 处走完 → 峰值与参考极值点对齐
        now = self.progress_buf[idx].unsqueeze(-1).float()
        self._gust_t0[idx] = now + (k_peak.unsqueeze(-1).float() - ramp).clamp_min(0.)
        self._gust_next[idx] = now + self.gust_period      # 下一次重发的时刻（从本次发射算起）

    def _update_gust(self):
        """每步把阵风写进 drone.flow_vels（世界系）。

        原实现只在 _reset_idx 写一次 flow_vels（每 episode 恒定 + 逐步噪声），
        做不了时变阵风；这里改成逐步写入。梯形波：ramp 上升 → hold 保持 → ramp 下降。
        """
        p = self.progress_buf.unsqueeze(-1).float()                      # [E,1]
        k = (p - self._gust_t0).clamp_min(0.)                            # 起风以来的步数
        up = (k / self._gust_ramp).clamp(0., 1.)                         # 上升沿
        dn = 1. - ((k - self._gust_ramp - self._gust_hold) / self._gust_ramp).clamp(0., 1.)   # 下降沿
        prof = (up * dn).clamp_min(0.)                                   # 梯形 ∈[0,1]
        v = self._gust_amp * prof                                        # [E,1] 带符号幅值
        fv = self.drone.flow_vels
        fv.zero_()
        fv.reshape(self.num_envs, -1)[:, self.gust_axis] = v.squeeze(-1)
        # 周期性重发，形成持续压力。用独立的 _gust_next 计时：_gust_t0 因为要对齐参考
        # 极值点会被推到未来，拿它算周期会错。
        due = p.squeeze(-1) >= self._gust_next.squeeze(-1)
        if bool(due.any()):
            self._launch_gust(torch.arange(self.num_envs, device=self.device), due)

    def _reset_idx(self, env_ids: torch.Tensor):
        if self.enable_flow:
            self.drone.set_flow_velocities(env_ids, self.max_flow_velocity, self.flow_velocity_gaussian_noise)
        self.drone._reset_idx(env_ids)
        self.traj_c[env_ids] = self.traj_c_dist.sample(env_ids.shape)
        self.traj_rot[env_ids] = euler_to_quaternion(self.traj_rpy_dist.sample(env_ids.shape))
        self.traj_scale[env_ids] = self.traj_scale_dist.sample(env_ids.shape)
        traj_w = self.traj_w_dist.sample(env_ids.shape)
        self.traj_w[env_ids] = torch.randn_like(traj_w).sign() * traj_w

        t0 = torch.zeros(len(env_ids), device=self.device)
        pos = lemniscate(t0 + self.traj_t0, self.traj_c[env_ids]) + self.origin
        rot = euler_to_quaternion(self.init_rpy_dist.sample(env_ids.shape))
        vel = torch.zeros(len(env_ids), 1, 6, device=self.device)
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids], rot, env_ids
        )
        self.drone.set_velocities(vel, env_ids)

        if self.enable_payload:
            payload_z = self.payload_z_dist.sample(env_ids.shape)
            joint_indices = torch.tensor([self.drone._view._dof_indices["PrismaticJoint"]], device=self.device)
            self.drone._view.set_joint_positions(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_position_targets(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_velocities(
                torch.zeros(len(env_ids), 1, device=self.device),
                env_indices=env_ids, joint_indices=joint_indices)

            payload_mass = self.payload_mass_dist.sample(env_ids.shape+(1,)) * self.drone.masses[env_ids]
            self.payload.set_masses(payload_mass, env_indices=env_ids)

        self.stats[env_ids] = 0.

        if self.corridor_enable:
            self._reset_corridor(env_ids)
        if self.gust_enable:
            self._launch_gust(env_ids, torch.ones(len(env_ids), dtype=torch.bool, device=self.device))

        if self.keepout_enable:
            # place N_OBSTACLES on the tracked reference (so a tracking policy must avoid them)
            n = len(env_ids)
            spawn_local = pos  # [n,3]
            for ob in range(self.n_obstacles):
                if self.dyn_enable:
                    start, vel = self._aim_obstacle(env_ids)
                    self.obstacle_pos[env_ids, ob] = start
                    self.obstacle_vel[env_ids, ob] = vel
                else:
                    self.obstacle_pos[env_ids, ob] = self._place_one_obstacle(env_ids, spawn_local)
            self.stats["min_obstacle_dist"][env_ids] = 100.0
            # efficiency metrics: reset path integrators
            self._flown[env_ids] = 0.0
            self._ref_len[env_ids] = 0.0
            self._last_p[env_ids] = pos
            self._last_ref[env_ids] = self._compute_traj(1, env_ids, step_size=5.)[:, 0]

        if self._should_render(0) and (env_ids == self.central_env_idx).any():
            self.draw.clear_lines()

            traj_vis = self._compute_traj(self.max_episode_length, self.central_env_idx.unsqueeze(0))[0]
            traj_vis = traj_vis + self.envs_positions[self.central_env_idx]
            point_list_0 = traj_vis[:-1].tolist()
            point_list_1 = traj_vis[1:].tolist()
            colors = [(1.0, 1.0, 1.0, 1.0) for _ in range(len(point_list_0))]
            sizes = [5 for _ in range(len(point_list_0))]
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        if self.gust_enable:
            self._update_gust()          # 必须在物理步之前写 flow_vels

        # ── 论文②：扰动观测器 + A→λ→B（走廊版）──────────────────────────────
        if self.corridor_enable and hasattr(self, "_cnd") and hasattr(self, "drone_state"):
            u_rl = actions
            # d̂：注意用的是**上一步**的状态/动作与本步真值速度的残差 —— 上一步的
            # _cor_prev 是在上一次 _pre_sim_step 末尾存的（物理步之前），本步进来时
            # drone 的 vel_b 已经是上一物理步之后的真值。
            if self.dobs_zero:
                self._d_hat.zero_()
            elif self.dobs_enable and getattr(self, "_cor_prev", None) is not None:
                self._dobs.update(self._cor_prev, self._cor_prev_a, self.drone.vel_b)
                self._d_hat = self._dobs.d_hat
            elif self.dobs_oracle:
                # 上界消融：用仿真真值流速直接算等效阻尼力（不是在线估计）
                self._d_hat = self._oracle_d_hat()

            gate_clear = None
            if self.cor_risk_enable:
                # A 恒定运行：不管谁做门控，risk 标量都进观测 —— 这样几何门控消融
                # 不会改变观测维度（论文①的做法，保证各格网络结构一致）
                min_clear, risk, _ = self._cor_rm.assess_corridor(
                    self.drone_state, self._cor_hw, u_rl, axis=self.cor_axis,
                    center=float(self.origin[self.cor_axis]),
                    vehicle_radius=self.cor_vehicle_radius, d_hat=self._d_hat)
                self._cor_risk = risk.unsqueeze(-1)
                if self.cor_risk_gate:
                    gate_clear = min_clear                      # 预测门控
                self._pred_clear = min_clear
                if self.k4_enable:
                    # K4：同一条前滚，换三种 d̂，比较预测最小裕度。离线再与"未来 H 步的
                    # 实际最小裕度"对比 —— 直接测门控**真正用到的那个量**，而不是泛泛的
                    # 位置预测误差。
                    z = torch.zeros_like(self._d_hat)
                    self._pred_clear_zero = self._cor_rm.assess_corridor(
                        self.drone_state, self._cor_hw, u_rl, axis=self.cor_axis,
                        center=float(self.origin[self.cor_axis]),
                        vehicle_radius=self.cor_vehicle_radius, d_hat=z)[0]
                    self._pred_clear_oracle = self._cor_rm.assess_corridor(
                        self.drone_state, self._cor_hw, u_rl, axis=self.cor_axis,
                        center=float(self.origin[self.cor_axis]),
                        vehicle_radius=self.cor_vehicle_radius, d_hat=self._oracle_d_hat())[0]
            if gate_clear is None and self.cor_mppi_enable:
                gate_clear = self._cor_clear                    # 几何门控：当前侧壁裕度

            if self.cor_mppi_enable and self.cor_force_lambda is not None:
                # MPC-only 基线：λ≡常数(通常 1.0)，全程由 MPPI 接管，带参考跟踪代价。
                # 这是"为什么不直接用 MPC"的对照组 —— 预期它 nominal 跟踪差、占空比 100%。
                N = self._cor_mppi.N
                ref = self._compute_traj(N, step_size=1.)                # [E,N,3]
                lam = torch.full((self.num_envs, 1), self.cor_force_lambda, device=self.device)
                # 采样中心：'rl'=策略动作(默认，滤波器语义)；'prev'=上一步实际执行的动作
                # (标准 MPPI 的一步热启动，与策略无关)；'zero'=完全不借助策略。
                if self.cor_mppi_center == "prev":
                    ca = self._cor_prev_a if self._cor_prev_a is not None else torch.zeros_like(u_rl)
                elif self.cor_mppi_center == "zero":
                    ca = torch.zeros_like(u_rl)
                else:
                    ca = None
                actions = self._cor_mppi.filter_corridor(
                    self.drone_state, self._cor_hw, u_rl, axis=self.cor_axis,
                    center=float(self.origin[self.cor_axis]),
                    vehicle_radius=self.cor_vehicle_radius, blend=lam, d_hat=self._d_hat,
                    ref_traj=ref, w_ref=self.cor_mppi_w_ref, center_action=ca)
                self._cor_lam = lam
                self._cor_corr = ((actions - u_rl) ** 2).sum(-1).reshape(self.num_envs, 1)
                self._u_rl = u_rl.detach().clone()
                self._u_applied = actions.detach().clone()
                tensordict.set(("agents", "action"), actions)
            elif self.cor_mppi_enable and gate_clear is not None:
                if self.cor_mppi_soft:
                    lam = ((self.cor_gate_boundary - gate_clear)
                           / (self.cor_gate_boundary - self.cor_mppi_lo)).clamp(0.0, 1.0)
                    # 同时传 active_mask：整批都 λ=0 时 filter_corridor 直接早退，
                    # 完全跳过 K×N 次前滚。软接管路径漏了这个早退会让无风险时段
                    # 也付全额 MPPI 代价（实测 2.4 s/步，评测矩阵根本跑不完）。
                    actions = self._cor_mppi.filter_corridor(
                        self.drone_state, self._cor_hw, u_rl, axis=self.cor_axis,
                        center=float(self.origin[self.cor_axis]),
                        vehicle_radius=self.cor_vehicle_radius, blend=lam,
                        active_mask=(lam > 0), d_hat=self._d_hat)
                else:
                    trig = gate_clear < self.cor_gate_boundary
                    lam = trig.float()
                    actions = self._cor_mppi.filter_corridor(
                        self.drone_state, self._cor_hw, u_rl, axis=self.cor_axis,
                        center=float(self.origin[self.cor_axis]),
                        vehicle_radius=self.cor_vehicle_radius, active_mask=trig, d_hat=self._d_hat)
                self._cor_lam = lam.reshape(self.num_envs, 1)
                self._cor_corr = ((actions - u_rl) ** 2).sum(-1).reshape(self.num_envs, 1)
                self._u_rl = u_rl.detach().clone()             # 供轨迹采集：滤波前/后动作对
                self._u_applied = actions.detach().clone()
                tensordict.set(("agents", "action"), actions)
            # 存本步（物理步之前）的状态与动作，供下一步算 d̂ 残差
            self._cor_prev = {
                "pos": self.drone_state[..., 0:3].clone(), "quat": self.drone.rot.clone(),
                "vel_b": self.drone.vel_b.clone(), "throttle": self.drone.throttle.clone(),
                "rpm": self.drone.rotor_params["rpm"].clone(),
                "v_prev": self.drone.prev_body_vels.clone(), "acc_prev": self.drone.prev_body_acc.clone(),
            }
            self._cor_prev_a = actions.clone()
        if self.keepout_enable and self.dyn_enable:
            # obstacle kinematics: constant-velocity motion + periodic re-aim (sustained pressure)
            self.obstacle_pos += self.obstacle_vel * self.dt
            reaim = (self.progress_buf > 0) & (self.progress_buf % self.dyn_reaim == 0)
            if reaim.any():
                ids = reaim.nonzero(as_tuple=True)[0]
                for ob in range(self.n_obstacles):
                    start, vel = self._aim_obstacle(ids)
                    self.obstacle_pos[ids, ob] = start
                    self.obstacle_vel[ids, ob] = vel
        if hasattr(self, "drone_state") and (self.shield_enable or self.mppi_enable or self.risk_enable or self.field_enable):
            u_rl = actions
            # nearest keep-out sphere to the vehicle → [E,3] (filters handle one obstacle at a time)
            drone_p = self.drone_state[..., :3]                              # [E,1,3]
            nearest_idx = ((self.obstacle_pos - drone_p) ** 2).sum(-1).argmin(-1)  # [E]
            ar = torch.arange(self.num_envs, device=self.device)
            closest = self.obstacle_pos[ar, nearest_idx]                     # [E,3]
            closest_vel = self.obstacle_vel[ar, nearest_idx] if self.dyn_enable else None  # [E,3]
            # A: predictive risk (single exact-model rollout of the policy's own action)
            trigger = None
            blend = None
            min_clear = None
            # --- A 恒定运行：只要 risk.enable，就前滚预测并把 risk 标量喂进观测，
            #     与「谁来做门控」无关（这样几何门控消融不会改变观测维度）。---
            if self.risk_enable:
                min_clear, risk, _ = self.risk_monitor.assess(
                    self.drone_state, closest, u_rl, obstacle_vel=closest_vel)
                self._risk = risk.unsqueeze(-1)                              # [E,1,1] → next obs (1-step delay)
            # --- 门控信号：预测间隙(A 门控) 或 当前间隙(几何门控) ---
            gate_clear = None
            if self.risk_enable and self.risk_gate:
                gate_clear = min_clear                                       # 预测式门控
            elif self.mppi_enable or self.field_enable:
                gate_clear = ((self.drone_state[..., :3].reshape(-1, 3) - closest).norm(dim=-1, keepdim=True)
                              - (self.keepout_radius + self.vehicle_radius))  # 几何门控
            # --- 接管形状：斜坡(soft) 或 阶跃(binary)，共用同一边界 gate_boundary ---
            if gate_clear is not None:
                if self.mppi_soft or self.field_enable:
                    blend = ((self.gate_boundary - gate_clear)
                             / (self.gate_boundary - self.mppi_soft_lo)).clamp(0.0, 1.0)
                    trigger = blend > 0
                else:
                    trigger = gate_clear < self.gate_boundary
                self._risk_trigger = trigger
            # protection chain — policy → (shield) → field OR MPPI
            if self.shield_enable:
                actions = self.shield.filter(self.drone_state, closest, actions)
            if self.field_enable and blend is not None:
                # pure-A-soft ablation: A's λ + reactive repulsion (thruster-axis allocation of "away" direction)
                u_rl_f = actions.reshape(self.num_envs, -1)
                p = self.drone_state[..., :3].reshape(-1, 3)
                away = p - closest
                away = away / (away.norm(dim=-1, keepdim=True) + 1e-6)
                from marinegym.utils.torch import quat_rotate_inverse
                away_b = quat_rotate_inverse(self.drone_state[..., 3:7].reshape(-1, 4), away)
                u_rep = (self.field_gain * torch.einsum('rd,ed->er', self._nd.Blin, away_b)).clamp(-1.0, 1.0)
                actions = (u_rl_f + blend.reshape(-1, 1) * (u_rep - u_rl_f)).view_as(actions)
            elif self.mppi_enable:
                actions = self.mppi.filter(self.drone_state, closest, actions, active_mask=trigger,
                                           obstacle_vel=closest_vel, blend=blend)
            # engagement signal for stats + adaptive C: λ when soft, else the 0/1 trigger
            if blend is not None:
                self._filter_engage = blend
            elif trigger is not None:
                self._filter_engage = trigger.float()
            # C: record the correction the filter applied (internalization penalty + stats)
            self._correction = ((actions - u_rl) ** 2).sum(-1)               # [E,1]
            # 供轨迹采集：滤波前(策略提议)与滤波后(实际执行)的动作对
            self._u_rl = u_rl.detach().clone()
            self._u_applied = actions.detach().clone()
            tensordict.set(("agents", "action"), actions)  # downstream logging/metrics see u_safe
        if self.validate_dynamics or self.validate_full:  # record pre-step state + applied action
            self._val_vb = self.drone.vel_b.clone()
            self._val_q = self.drone.rot.clone()
            self._val_a = actions.clone()
            self._val_thr = self.drone.throttle.clone()
            self._val_rpm = self.drone.rotor_params["rpm"].clone()
            self._val_vprev = self.drone.prev_body_vels.clone()
            self._val_aprev = self.drone.prev_body_acc.clone()
        self.effort = torch.abs(self.drone.apply_action(actions))

        if self.wind:
            t = (self.progress_buf * self.dt).reshape(-1, 1, 1)
            self.wind_force = self.wind_i * torch.sin(t * self.wind_w).sum(-1)
            wind_forces = self.drone.MASS_0 * self.wind_force
            wind_forces = wind_forces.unsqueeze(1).expand(*self.drone.shape, 3)
            self.drone.base_link.apply_forces(wind_forces, is_global=True)

    def _nom_onestep(self):
        """N1: 名义模型单步机体速度相对误差 —— 完整模型 + 两项消融。"""
        import copy, json, statistics
        nd = self.nominal
        if not hasattr(self, "_nom_var"):
            a = copy.copy(nd); a.tau = 1.0; a.time_const = nd.time_const * 0 + 1e-9  # 去 T200 迟滞
            b = copy.copy(nd); b.M_rb = nd.M_rb + nd.M_A.abs()                       # 附加质量并入 M
            self._nom_var = {"full": nd, "no_t200_lag": a, "added_mass_in_M": b}
            self._nom_err = {k: [] for k in self._nom_var}
        st = dict(pos=self.drone_state[..., :3], quat=self._val_q, vel_b=self._val_vb,
                  v_prev=self._val_vprev, acc_prev=self._val_aprev,
                  throttle=self._val_thr, rpm=self._val_rpm)
        va = self.drone.vel_b
        for name, m in self._nom_var.items():
            pred = m.step({k: v.clone() for k, v in st.items()}, self._val_a)["vel_b"]
            self._nom_err[name].append(
                ((pred - va).norm(dim=-1) / (va.norm(dim=-1) + 1e-3)).median().item())
        if len(self._nom_err["full"]) % 200 == 0:
            out = {k: dict(mean=statistics.mean(v), median=statistics.median(v), n=len(v))
                   for k, v in self._nom_err.items()}
            with open("/home/jovyan/MarineGym/scripts/outputs_aei/data/nominal_onestep.json", "w") as f:
                json.dump(out, f, indent=2)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()

        if self.validate_full and hasattr(self, "_val_vb"):
            self._nom_onestep()
        if self.validate_dynamics and hasattr(self, "_val_vb") and self._val_count < 8:
            from marinegym.utils.torch import quat_rotate_inverse, quaternion_to_euler
            nd = self.nominal
            va = self.drone.vel_b
            q, vb = self._val_q, self._val_vb
            g_w = torch.zeros_like(vb[..., :3]); g_w[..., 2] = -nd.weight
            g_b = quat_rotate_inverse(q.reshape(-1, 4), g_w.reshape(-1, 3)).reshape_as(g_w)
            # sim thrust wrench (actual thrust incl lag) vs my T200 reimpl
            ts = self.drone.thrusts[..., 0]
            Fts = torch.einsum('...r,rd->...d', ts, nd.Blin); Tts = torch.einsum('...r,rd->...d', ts, nd.tau_cross)
            _, _, tm = nd.t200_step(self._val_thr, self._val_rpm, self._val_a)
            Ftm = torch.einsum('...r,rd->...d', tm, nd.Blin); Ttm = torch.einsum('...r,rd->...d', tm, nd.tau_cross)
            # sim hydro vs my hydro reimpl (prev vel/acc from sim are already in the flipped frame)
            myh, _, _ = nd.hydro_wrench(vb, self._val_vprev, self._val_aprev, quaternion_to_euler(q))
            Fhs = self.drone.forces.reshape(*vb.shape[:-1], 3); Ths = self.drone.torques.reshape(*vb.shape[:-1], 3)
            vbC = vb + nd.dt * torch.cat([Fts + myh[..., 0:3] + g_b, Tts + myh[..., 3:6]], -1) / nd.M_rb   # sim thrust + MY hydro
            vbD = vb + nd.dt * torch.cat([Ftm + Fhs + g_b, Ttm + Ths], -1) / nd.M_rb                        # MY thrust + sim hydro
            relC = ((vbC - va).norm(dim=-1) / (va.norm(dim=-1) + 1e-3)).median().item()
            relD = ((vbD - va).norm(dim=-1) / (va.norm(dim=-1) + 1e-3)).median().item()
            print(f"[ISO {self._val_count}] C(simThrust+MYhydro)={relC:.3f}  D(MYthrust+simHydro)={relD:.3f}  "
                  f"(low ⇒ that half is OK; other half is my bug)", flush=True)
            self._val_count += 1

        self.target_pos[:] = self._compute_traj(self.future_traj_steps, step_size=5)

        self.rpos = self.target_pos - self.drone_state[..., :3]

        if self.corridor_enable:
            # keep-in 走廊（世界系侧壁）：clearance = 半宽 − |受限轴坐标| − 载具等效半径。
            # 与跟踪误差是**两个不同的量** —— 参考线本身在 8 字两端就贴近侧壁，载具可以
            # 跟得很准却仍贴壁，也可以跟得一般却远离侧壁。这是门控信号与奖励主项解耦的关键。
            # 必须在拼 obs 之前算（obs 里要用到）。
            self._cor_off = self.drone_state[..., self.cor_axis] - self.origin[self.cor_axis]  # [E,1] 有符号
            self._cor_clear = self._cor_hw - self._cor_off.abs() - self.cor_vehicle_radius
            self._cor_violate = self._cor_clear < 0

        obs = [
            self.rpos.flatten(1).unsqueeze(1),
            self.drone_state[..., 3:],
        ]
        if self.keepout_enable and self.obstacle_in_obs:
            # For multi-obstacle: provide the CLOSEST obstacle's relative position to the policy
            # (keeps obs_dim constant — the policy was trained with 3 obstacle dims)
            # obstacle_pos [E,N,3]  minus  drone_p [E,1,3] (from drone_state[...,:3]) → [E,N,3]
            drone_p = self.drone_state[..., :3]  # [E,1,3] (3D)
            obst_dists = torch.norm(self.obstacle_pos - drone_p, dim=-1)  # [E,N]
            closest_idx = obst_dists.argmin(dim=-1)  # [E]
            # Gather closest: obstacle_pos[E,N,3][arange, idx] → [E,3]; reshape to [E,1,3]
            batched = torch.arange(obst_dists.shape[0], device=self.device)
            closest_obs = self.obstacle_pos[batched, closest_idx]  # [E,3]
            closest_obs = closest_obs.reshape_as(drone_p)  # [E,1,3]
            obs.append(closest_obs - drone_p)  # [E,1,3] − [E,1,3] = [E,1,3]
            if self.dyn_enable:
                obs.append(self.obstacle_vel[batched, closest_idx].reshape_as(drone_p))  # [E,1,3]
        if self.risk_in_obs:
            obs.append(self._risk)  # [E,1,1] component A risk scalar (1-step delayed)
        if self.cor_in_obs:
            # 到两侧壁的裕度分开给（而不是给 min），策略才知道自己偏向哪一侧
            d_pos = (self._cor_hw - self._cor_off - self.cor_vehicle_radius).unsqueeze(-1)
            d_neg = (self._cor_hw + self._cor_off - self.cor_vehicle_radius).unsqueeze(-1)
            obs.append(torch.cat([d_pos, d_neg], dim=-1))            # [E,1,2]
        if self.flow_in_obs:
            # 公平性：策略拿得到扰动信息（真实 AUV 上 DVL 能测对水速度），
            # 这样"策略知道有流仍然失败、带预测的滤波器成功"的对比才落在**预测**上，
            # 而不是信息不对称。
            # 用 flow_in_obs 而非 gust_enable：见其定义处 —— 观测维度不能随场景变。
            _a = self.flow_obs_axis
            obs.append(self.drone.flow_vels.reshape(self.num_envs, 1, -1)[..., _a:_a + 1])
        if self.cor_risk_in_obs:
            obs.append(self._cor_risk)   # [E,1,1]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        if self.intrinsics:
            obs.append(self.drone.get_info())

        obs = torch.cat(obs, dim=-1)

        if self.keepout_enable:
            # update safety metrics for ALL obstacles
            # [E,N,3] minus [E,1,3] → [E,N,3] (broadcasting-safe, NO unsqueeze!)
            drone_p = self.drone_state[..., :3]  # [E,1,3]
            all_dists = torch.norm(self.obstacle_pos - drone_p, dim=-1)  # [E,N]
            all_clearance = all_dists - (self.keepout_radius + self.vehicle_radius)
            min_clearance = all_clearance.min(dim=-1, keepdim=True).values  # [E,1]
            self._clearance = min_clearance
            self._collision = (min_clearance < 0)
            self.stats["collision"] = torch.maximum(self.stats["collision"], self._collision.float())
            self.stats["min_obstacle_dist"] = torch.minimum(self.stats["min_obstacle_dist"], min_clearance)

        if self.corridor_enable:
            v = self._cor_violate.float()
            self.stats["wall_violation"] = torch.maximum(self.stats["wall_violation"], v)
            # 用 EMA 而不是"累计步数 + episode 末尾除以 ep_len"：
            # EpisodeStats 收集的是逐步值再做平均，拿不到"末尾那一次除法"之后的结果
            # （同样的原因让 stats.tracking_error 报出 -545 这种未归一化的累计值）。
            # EMA 与 action_smoothness / tracking_error_ema 同一约定，可直接当占比读。
            self.stats["wall_viol_frac"].lerp_(v, (1 - self.alpha))
            self.stats["wall_depth"] += (-self._cor_clear).clamp_min(0.) * self.dt
            self.stats["min_wall_dist"] = torch.minimum(self.stats["min_wall_dist"], self._cor_clear)
            if hasattr(self, "_cor_lam"):
                self.stats["filter_lambda"].lerp_(self._cor_lam, (1 - self.alpha))
                self.stats["wall_correction"].lerp_(self._cor_corr, (1 - self.alpha))
                self.stats["d_hat_norm"].lerp_(
                    self._d_hat[..., :3].norm(dim=-1).reshape(self.num_envs, 1), (1 - self.alpha))
        if self.gust_enable:
            self.stats["gust_speed"].lerp_(
                self.drone.flow_vels.reshape(self.num_envs, -1)[:, self.gust_axis].abs().unsqueeze(-1),
                (1 - self.alpha))

        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))
        if self.keepout_enable:
            # EMA over the episode, same convention as action_smoothness
            self.stats["filter_activation"].lerp_(self._filter_engage, (1 - self.alpha))
            self.stats["correction"].lerp_(self._correction, (1 - self.alpha))
            # efficiency: flown vs reference path length (detour_ratio ≥ 1, →1 best),
            # and clearance beyond the safe margin (over_clearance small-positive best)
            p_now = self.drone_state[..., :3].reshape(-1, 3)
            ref_now = self.target_pos[:, 0]
            self._flown += (p_now - self._last_p).norm(dim=-1, keepdim=True)
            self._ref_len += (ref_now - self._last_ref).norm(dim=-1, keepdim=True)
            self._last_p = p_now.clone()
            self._last_ref = ref_now.clone()
            self.stats["detour_ratio"] = self._flown / self._ref_len.clamp_min(1e-3)
            self.stats["over_clearance"] = (self.stats["min_obstacle_dist"] - self.safe_margin).clamp_min(0.0)
            self.stats["internalize_w"][:] = self._wC

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        distance = torch.norm(self.rpos[:, [0]], dim=-1)
        self.stats["tracking_error"].add_(-distance)
        self.stats["tracking_error_ema"].lerp_(distance, (1-self.alpha))
        
        reward_pose = 0.5*torch.exp(-self.reward_distance_scale * distance)
        
        tiltage = torch.abs(1 - self.drone.up[..., 2])
        reward_up = 0.5 / (1.0 + torch.square(tiltage))

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)

        spin = torch.square(self.drone.vel[..., -1])
        reward_spin = 0.5 / (1.0 + torch.square(spin))

        reward = (
            reward_pose
            + reward_pose * (reward_up + reward_spin)
            + reward_effort
            + reward_action_smoothness
        )

        if self.keepout_enable:
            # per-obstacle avoidance penalty: quadratic once entering margin band
            # _clearance is [E,1] = min across all obstacles
            pen = (self.reward_obstacle_margin - self._clearance).clamp_min(0.0)
            reward = reward - self.reward_obstacle_weight * pen * pen

        if self.corridor_enable:
            # 走廊接近惩罚：进入 margin 带后二次增长。和论文①一样，避壁是**任务的一部分**，
            # 纯 PPO 有动机自己避 —— baseline 不是"根本不知道有墙"的弱 baseline。
            wpen = (self.cor_reward_margin - self._cor_clear).clamp_min(0.0)
            reward = reward - self.cor_reward_weight * wpen * wpen
            if self.internalize_adaptive:
                # adaptive C: internalization pressure tracks recent reliance on the filter —
                # heavy engagement raises the penalty, and it self-extinguishes once the
                # policy stops needing protection (w_C → 0 as engagement → 0)
                self._engage_ema.lerp_(self._filter_engage.mean(), 1.0 - self.internalize_ema)
                self._wC = self.internalize_gain * self._engage_ema
                reward = reward - self._wC * self._correction
            elif self.internalize_weight > 0:
                # component C: internalize the filter — penalize the correction B had to apply,
                # so the policy learns to propose safe actions and B's activation decays to ~0
                reward = reward - self.internalize_weight * self._correction

        misbehave = (
            (self.drone.pos[..., 2] < 0.1)
            | (distance > self.reset_thres)
        )
        hasnan = torch.isnan(self.drone_state).any(-1)

        terminated = misbehave | hasnan
        if self.keepout_enable and self.keepout_terminate:
            terminated = terminated | self._collision  # self._collision computed in _compute_state_and_obs
        if self.corridor_enable and self.cor_terminate:
            terminated = terminated | self._cor_violate
        truncated = (self.progress_buf >= self.max_episode_length - 1).unsqueeze(-1)

        ep_len = self.progress_buf.unsqueeze(-1)
        self.stats["tracking_error"].div_(
            torch.where(terminated | truncated, ep_len, torch.ones_like(ep_len))
        )
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )

    def _compute_traj(self, steps: int, env_ids=None, step_size: float=1.):
        if env_ids is None:
            env_ids = ...
        t = self.progress_buf[env_ids].unsqueeze(1) + step_size * torch.arange(steps, device=self.device)
        t = self.traj_t0 + scale_time(self.traj_w[env_ids].unsqueeze(1) * t * self.dt)
        traj_rot = self.traj_rot[env_ids].unsqueeze(1).expand(-1, t.shape[1], 4)

        target_pos = vmap(lemniscate)(t, self.traj_c[env_ids])

        target_pos = vmap(quat_rotate)(traj_rot, target_pos) * self.traj_scale[env_ids].unsqueeze(1)

        return self.origin + target_pos
