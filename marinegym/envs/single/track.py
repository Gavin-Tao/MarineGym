
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
            self.n_obstacles = 1
            self.lateral_offset_std = 0.0
            self.risk_enable = False
            self.risk_in_obs = False
            self.internalize_weight = 0.0
            self.internalize_adaptive = False
            self.mppi_soft = False
            self.field_enable = False
            self.dyn_enable = False

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
        self.traj_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
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
        self.traj_w_dist = D.Uniform(
            torch.tensor(0.7, device=self.device),
            torch.tensor(0.9, device=self.device)
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
        if self.validate_dynamics:
            from marinegym.utils.nominal_dynamics import NominalDynamics
            self.nominal = NominalDynamics(self.drone, self.dt * int(self.cfg.sim.get("substeps", 1)))
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
            if self.risk_enable:
                min_clear, risk, trigger = self.risk_monitor.assess(self.drone_state, closest, u_rl, obstacle_vel=closest_vel)
                self._risk = risk.unsqueeze(-1)                              # [E,1,1] → next obs (1-step delay)
                if self.mppi_soft or self.field_enable:
                    # risk-proportional soft takeover: λ ramps 0→1 as PREDICTED clearance falls hi→lo
                    blend = ((self.mppi_soft_hi - min_clear)
                             / (self.mppi_soft_hi - self.mppi_soft_lo)).clamp(0.0, 1.0)
                    trigger = blend > 0
                self._risk_trigger = trigger
            elif self.mppi_soft:
                # pure-B-soft ablation: λ from CURRENT clearance (no prediction)
                clearance_now = ((self.drone_state[..., :3].reshape(-1, 3) - closest).norm(dim=-1, keepdim=True)
                                 - (self.keepout_radius + self.vehicle_radius))
                blend = ((self.mppi_soft_hi - clearance_now)
                         / (self.mppi_soft_hi - self.mppi_soft_lo)).clamp(0.0, 1.0)
                trigger = blend > 0
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
            tensordict.set(("agents", "action"), actions)  # downstream logging/metrics see u_safe
        if self.validate_dynamics:  # record pre-step state + applied action for one-step dynamics validation
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

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()

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
