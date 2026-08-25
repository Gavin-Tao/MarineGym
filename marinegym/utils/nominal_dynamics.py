"""
Nominal 6-DoF underwater-vehicle forward model for model-based safety filtering (arm A / MPPI).

Batched, stateful torch reimplementation of MarineGym's OWN dynamics, matched to the simulator
(validated <2% one-step velocity error via the "reuse sim forces" reference).  The earlier hand
model was ~90% off because it (a) ignored the T200 throttle/rpm lag and (b) integrated with
M_RB+M_A instead of the sim's scheme (M_RB rigid-body integration with added mass applied as a
finite-difference force).  This version replicates the sim exactly:

  actuator (T200, per rotor, stateful):  throttle += tau*(clip(a)-throttle);
      rpm = e^{-dt/T} rpm + (1-e^{-dt/T}) rpm_target(throttle);  thrust = fc/4.4e-7 * 9.81 * poly(rpm)
  hydro (body, with the sim's [1,2,4,5] sign flips):
      damping = (Dl + Dq|v|) v ;  coriolis = C_A(v) v ;  added_mass = M_A * acc_filt ;  buoyancy(rpy)
      acc_filt = 0.7 acc_prev + 0.3 (v - v_prev)/dt      (calculate_acc low-pass)
  rigid body:  M_RB nu_dot = tau_thrust + hydro + gravity ;  nu += dt nu_dot

Carried state per sample: pos_w, quat, vel_b, v_prev, acc_prev, throttle, rpm.  step() advances one
control tick and is batched over an arbitrary leading shape (E for validation, E*K for MPPI).
"""

import torch

from marinegym.utils.torch import quat_rotate, quat_rotate_inverse, quat_axis, quaternion_to_euler

_FLIP = [1, 2, 4, 5]


class NominalDynamics:
    def __init__(self, drone, dt: float):
        self.device = drone.device
        self.dt = float(dt)
        self.M = drone.num_rotors

        m = drone.masses.reshape(-1)[0].to(self.device)
        I = drone.INERTIA_0.reshape(-1)[:3].to(self.device)
        self.M_rb = torch.cat([m.expand(3), I]).to(self.device)                    # (6,) rigid-body diag
        self.M_A = drone.ADDED_MASS_0.to(self.device)                              # (6,)
        self.Dl = drone.LINEAR_DAMPING_0.to(self.device)                           # (6,)
        self.Dq = drone.QUADRATIC_DAMPING_0.to(self.device)                        # (6,)
        self.buoyForce = float(997 * 9.8 * drone.VOLUME_0.reshape(-1)[0].item())
        self.coBM = float(drone.CoBM_0.reshape(-1)[0].item())
        self.weight = float(m.item() * 9.81)

        # T200 actuator params
        self.tau = 0.43
        self.time_const = drone.TIME_CONSTANTS_0.reshape(-1).to(self.device)       # (R,) or scalar
        self.fc_scale = drone.FORCE_CONSTANTS_0.reshape(-1)[0].item() / 4.4e-7

        # body-frame thruster axes / positions
        rp_w, rr_w = drone.rotors_view.get_world_poses()
        rp_w = rp_w.reshape(*drone.shape, self.M, 3)
        rr_w = rr_w.reshape(*drone.shape, self.M, 4)
        axis_w = quat_axis(rr_w.flatten(end_dim=-2), axis=0).unflatten(0, (*drone.shape, self.M))
        brot = drone.rot.reshape(*drone.shape, 1, 4).expand(*drone.shape, self.M, 4)
        bpos = drone.pos.reshape(*drone.shape, 1, 3).expand(*drone.shape, self.M, 3)
        self.Blin = quat_rotate_inverse(brot.reshape(-1, 4), axis_w.reshape(-1, 3)).reshape(*drone.shape, self.M, 3)[0, 0]
        self.r_i = quat_rotate_inverse(brot.reshape(-1, 4), (rp_w - bpos).reshape(-1, 3)).reshape(*drone.shape, self.M, 3)[0, 0]
        self.tau_cross = torch.cross(self.r_i, self.Blin, dim=-1)                   # (M,3)

    def t200_step(self, throttle, rpm, action):
        """stateful T200: returns (throttle', rpm', thrust). shapes [...,R]."""
        throttle = throttle + self.tau * (action.clamp(-1, 1) - throttle)
        target_rpm = torch.where(throttle > 0.075, 3.6599e3 * throttle + 3.4521e2,
                     torch.where(throttle < -0.075, 3.4944e3 * throttle - 4.3350e2, torch.zeros_like(throttle)))
        alpha = torch.exp(-torch.tensor(self.dt, device=self.device) / self.time_const)
        rpm = (alpha * rpm + (1 - alpha) * target_rpm).clamp(-3900, 3900)
        thrust = self.fc_scale * 9.81 * torch.where(
            rpm > 0, 4.7368e-7 * rpm.pow(2) - 1.9275e-4 * rpm + 8.4452e-2,
                     -3.8442e-7 * rpm.pow(2) - 1.6186e-4 * rpm - 3.9139e-2)
        return throttle, rpm, thrust

    def hydro_wrench(self, vel_b, v_prev_flip, acc_prev_flip, rpy):
        """Exact mirror of sim.apply_hydrodynamic_forces (all in the sim's [1,2,4,5]-flipped frame).
        Returns (wrench, v_flip, acc_flip) so the flipped prev-state can be carried through a rollout."""
        v = vel_b.clone(); v[..., _FLIP] *= -1                                  # body_vels flipped
        acc = 0.7 * acc_prev_flip + 0.3 * (v - v_prev_flip) / self.dt           # calculate_acc low-pass (flipped frame)
        damping = (self.Dl + self.Dq * v.abs()) * v
        added_mass = self.M_A * acc
        ab = self.M_A * v
        cor = torch.zeros_like(v)
        cor[..., 0:3] = -torch.cross(ab[..., 0:3], v[..., 3:6], dim=-1)
        cor[..., 3:6] = -(torch.cross(ab[..., 0:3], v[..., 0:3], dim=-1) + torch.cross(ab[..., 3:6], v[..., 3:6], dim=-1))
        hydro = -(added_mass + cor + damping)
        hydro[..., _FLIP] *= -1                                                 # flip back
        r = rpy.clone(); r[..., [1, 2]] *= -1                                   # body_rpy flipped
        buoy = torch.zeros_like(v)
        bf, dis = self.buoyForce, self.coBM
        buoy[..., 0] = bf * torch.sin(r[..., 1])
        buoy[..., 1] = -bf * torch.sin(r[..., 0]) * torch.cos(r[..., 1])
        buoy[..., 2] = -bf * torch.cos(r[..., 0]) * torch.cos(r[..., 1])
        buoy[..., 3] = -dis * bf * torch.cos(r[..., 1]) * torch.sin(r[..., 0])
        buoy[..., 4] = -dis * bf * torch.sin(r[..., 1])
        buoy[..., _FLIP] *= -1                                                  # flip buoyancy back (was missing!)
        return hydro + buoy, v, acc

    @torch.no_grad()
    def step(self, s, action, d_hat=None):
        """s = dict(pos, quat, vel_b, v_prev, acc_prev, throttle, rpm) [v_prev/acc_prev in flipped frame].

        d_hat [...,6] (论文②): 体系等效广义外力残差，加到 wrench 上。用来把在线估到的
        未建模水动力（洋流）带进前滚 —— offset-free MPC 的标准做法，前滚期间零阶保持。
        """
        pos, quat, vel_b = s["pos"], s["quat"], s["vel_b"]
        throttle, rpm, thrust = self.t200_step(s["throttle"], s["rpm"], action)
        F_thr = torch.einsum('...r,rd->...d', thrust, self.Blin)
        T_thr = torch.einsum('...r,rd->...d', thrust, self.tau_cross)
        rpy = quaternion_to_euler(quat)
        hydro, v_flip, acc_flip = self.hydro_wrench(vel_b, s["v_prev"], s["acc_prev"], rpy)
        g_w = torch.zeros_like(pos); g_w[..., 2] = -self.weight
        g_b = quat_rotate_inverse(quat.reshape(-1, 4), g_w.reshape(-1, 3)).reshape_as(pos)
        wrench = torch.cat([F_thr + hydro[..., 0:3] + g_b, T_thr + hydro[..., 3:6]], dim=-1)
        if d_hat is not None:
            wrench = wrench + d_hat
        nu_dot = wrench / self.M_rb
        v_new = vel_b + self.dt * nu_dot
        v_w = quat_rotate(quat.reshape(-1, 4), v_new[..., 0:3].reshape(-1, 3)).reshape_as(pos)
        return {
            "pos": pos + self.dt * v_w, "quat": quat, "vel_b": v_new,
            "v_prev": v_flip, "acc_prev": acc_flip, "throttle": throttle, "rpm": rpm,
        }
