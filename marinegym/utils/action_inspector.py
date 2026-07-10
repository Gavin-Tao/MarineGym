"""
Action inspector (safety shield) for keep-out avoidance — allocation-aware HOCBF filter.

Mid-layer of the safe-RL design (纯 PPO + action inspector + 预测层): the RL policy
proposes u_rl, the inspector returns a certified-safer u_safe.

CBF (inertial / world frame — MarineGym's get_state() exposes world position and
linear velocity, avoiding the body-frame sign conventions of the hydro code):

    h(p)  = ||p - p_o||^2 - (r_o + r_v)^2                          (rel. degree 2)
    hdot  = 2 (p - p_o)^T v
    psi2  = 2||v||^2 + 2 (p - p_o)^T a + (a1+a2) hdot + a1 a2 h >= 0,   a = a_ctrl + a_drift

Controllable world acceleration: a_ctrl = R M_lin^{-1} (Blin^T f), f ≈ thrust_gain*action.
Uncontrolled drift bounded by drift_bound (worst-case robustness margin).  This yields a
per-step linear constraint  A(x).action + b(x) >= 0,  A_i = k (Blin[i] . (g_body/M_lin)).

ALLOCATION-AWARE (torque-neutral) projection — the fix for tumbling:
    A naive per-rotor projection that only enforces the *translational* CBF induces large
    parasitic torques (saturated thrusts tumble the vehicle, so the "away" force is lost).
    Instead we restrict the correction to the null space of the torque allocation,
    Delta_action in null(B_torque):  the shield changes only the NET FORCE (to meet the CBF)
    while leaving the NET TORQUE exactly as the policy commanded.  Blin[i] and the body-frame
    rotor position r_i (hence B_torque[i] = r_i x Blin[i]) are extracted once at init.

    P = I - Bt^T (Bt Bt^T)^{-1} Bt      (projection onto null(Bt), Bt = torque allocation [3, M])
    c   = max(0, -(A.u_rl + b))          (needed increase of the CBF margin)
    Δu  = c * (P A^T) / (A^T P A)        (min-norm torque-neutral correction achieving A.Δu = c)
    u   = clip(u_rl + Δu, -1, 1)         (thruster box), gated to activate only near the obstacle

STATUS (2026-07-09, WIP — does NOT reliably improve safety, default DISABLED):
On a hard-conflict BlueROV test (radius=1.3) vs a shield-off baseline (collision 0.45,
min clearance -0.00), three formulations were tried across ~15 Isaac runs:
    naive per-rotor projection ....... coll 0.92  min -0.37  (tumbles: parasitic torque)
    torque-neutral projection ........ coll 0.97  min -0.41
    tuned (drift_bound 0.5, gate 0.3)  coll 0.67  min -0.21  (best, still > baseline)
The projection direction is locally correct (corr·d̂>0, pushes away), but a single-step
filter cannot brake past the input-constrained point of no return, and a correct
allocation-aware CBF-QP is non-trivial.  This is exactly the design-doc §5.1 thesis:
feasibility needs the PREDICTIVE/BACKUP layer (arm A-MPC / step 3).  Kept as evidence for
the "single-step shield insufficient" ablation; do not enable without the predictive layer.
"""

import torch

from marinegym.utils.torch import quat_rotate_inverse, quat_axis


class HOCBFShield:
    def __init__(
        self,
        drone,
        keepout_radius: float,
        vehicle_radius: float,
        alpha1: float = 2.0,
        alpha2: float = 2.0,
        drift_bound: float = 5.0,
        thrust_gain: float = 40.0,
        activation_margin: float = 1.0,
        num_passes: int = 3,
    ):
        self.device = drone.device
        self.r = float(keepout_radius) + float(vehicle_radius)
        self.a1 = float(alpha1)
        self.a2 = float(alpha2)
        self.drift_bound = float(drift_bound)
        self.k = float(thrust_gain)
        self.activation_margin = float(activation_margin)
        self.num_passes = int(num_passes)
        self.num_rotors = drone.num_rotors
        M = self.num_rotors

        # translational effective mass M_lin = m + added_mass[:3]
        m = drone.masses.reshape(-1)[0].to(self.device)
        added = drone.ADDED_MASS_0[:3].to(self.device)
        self.M_lin = (m + added).view(1, 1, 3)                       # (1,1,3)

        # rigid-body thruster geometry in the BODY frame (constant): thrust axes and positions
        rotor_pos_w, rotor_rot_w = drone.rotors_view.get_world_poses()
        rotor_pos_w = rotor_pos_w.reshape(*drone.shape, M, 3)
        rotor_rot_w = rotor_rot_w.reshape(*drone.shape, M, 4)
        axis_w = quat_axis(rotor_rot_w.flatten(end_dim=-2), axis=0).unflatten(0, (*drone.shape, M))  # thrust dir (local x)
        base_rot = drone.rot.reshape(*drone.shape, 1, 4).expand(*drone.shape, M, 4)
        base_pos = drone.pos.reshape(*drone.shape, 1, 3).expand(*drone.shape, M, 3)
        axis_b = quat_rotate_inverse(base_rot.reshape(-1, 4), axis_w.reshape(-1, 3)).reshape(*drone.shape, M, 3)
        r_b = quat_rotate_inverse(base_rot.reshape(-1, 4), (rotor_pos_w - base_pos).reshape(-1, 3)).reshape(*drone.shape, M, 3)

        self.Blin = axis_b[0, 0].to(self.device)                    # (M, 3) body-frame thrust axes
        r_i = r_b[0, 0].to(self.device)                             # (M, 3) body-frame rotor positions
        Bt = torch.cross(r_i, self.Blin, dim=-1).transpose(0, 1)    # (3, M) torque allocation: Bt @ f = net torque
        eye3 = torch.eye(3, device=self.device)
        BBt_inv = torch.inverse(Bt @ Bt.transpose(0, 1) + 1e-6 * eye3)
        self.P = torch.eye(M, device=self.device) - Bt.transpose(0, 1) @ BBt_inv @ Bt   # (M, M) proj onto null(Bt)

    @torch.no_grad()
    def filter(self, drone_state: torch.Tensor, obstacle_pos: torch.Tensor, u_rl: torch.Tensor) -> torch.Tensor:
        """
        drone_state: [E,1,S] (pos 0:3, rot 3:7, world linear vel 7:10)
        obstacle_pos: [E,3];  u_rl: [E,1,M] action in [-1,1];  returns u_safe [E,1,M]
        """
        p = drone_state[..., 0:3]
        rot = drone_state[..., 3:7]
        v = drone_state[..., 7:10]
        p_o = obstacle_pos.unsqueeze(1)

        d = p - p_o
        dist2 = (d * d).sum(-1, keepdim=True)
        dist = dist2.clamp_min(1e-8).sqrt()
        h = dist2 - self.r ** 2
        hdot = 2.0 * (d * v).sum(-1, keepdim=True)

        # translational CBF sensitivity A_i = k * Blin[i] . (g_body / M_lin), g_body = R^T (2 d)
        g_body = quat_rotate_inverse(rot.reshape(-1, 4), (2.0 * d).reshape(-1, 3)).reshape_as(d)
        w = g_body / self.M_lin
        A = self.k * torch.einsum('rd,eod->eor', self.Blin, w)      # [E,1,M]

        v2 = (v * v).sum(-1, keepdim=True)
        b = (2.0 * v2 + (self.a1 + self.a2) * hdot + self.a1 * self.a2 * h
             - 2.0 * dist * self.drift_bound)                       # [E,1,1]

        # torque-neutral min-norm projection: Δu in null(B_torque) so the correction adds no net torque
        PA = torch.einsum('mn,eon->eom', self.P, A)                 # [E,1,M]  (P A^T)
        denom = (A * PA).sum(-1, keepdim=True).clamp_min(1e-8)      # A^T P A
        u = u_rl
        for _ in range(self.num_passes):
            slack = (A * u).sum(-1, keepdim=True) + b               # CBF margin; <0 ⇒ unsafe
            c = (-slack).clamp_min(0.0)                             # needed increase
            u = u + PA * (c / denom)                                # torque-neutral force correction
            u = u.clamp(-1.0, 1.0)                                  # thruster box

        active = dist < (self.r + self.activation_margin)           # intervene only near the boundary
        return torch.where(active, u, u_rl)
