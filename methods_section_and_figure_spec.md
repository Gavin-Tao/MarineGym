# Methods（论文正文草稿）+ Framework 示意图绘制规格

> 用途：① 第 I–VIII 节是论文 Method 部分的可用草稿（英文正文 + 中文批注）；② 第 IX 节是**给画图的人看的图规格说明**（中文，逐框逐箭头写死）。
> 方法名：**A+B-soft = Risk-Proportional Predictive Safety Filter (RP-PSF)**

---

# 第一部分：Methods 正文

## I. Method Overview

**中文说明：这一段是给读者的"三句话总览"，画图的人也应该先读这一段——图必须一眼看出这三件事。**

We augment a standard PPO tracking policy with a **predictive safety filter whose engagement is continuous and action-conditioned**. At每 control step the pipeline executes four stages:

1. **Propose.** The policy $\pi_\theta$ observes $o_t$ and proposes a thrust command $u^{\mathrm{rl}}_t \in [-1,1]^6$.
2. **Predict (Component A).** A *single* rollout of $u^{\mathrm{rl}}_t$ through a calibrated nominal vehicle model returns the **predicted minimum clearance** $\hat d_t$ to the keep-out set over an $H$-step lookahead. This is a property of *the policy's own action*, not of the instantaneous geometry.
3. **Schedule.** $\hat d_t$ is mapped through a fixed ramp to a **takeover coefficient** $\lambda_t \in [0,1]$.
4. **Protect & blend (Component B).** If $\lambda_t > 0$, a sampling-based MPPI filter using the *same* nominal model returns a safe command $u^{\mathrm{safe}}_t$; the executed command is the convex combination
$$u_t = (1-\lambda_t)\,u^{\mathrm{rl}}_t + \lambda_t\,u^{\mathrm{safe}}_t .$$

The two design choices that distinguish RP-PSF from prior safety filters are therefore **(i) what triggers the filter** — a predicted, action-conditioned risk rather than a distance heuristic — and **(ii) how strongly it acts** — a continuous proportional blend rather than a binary switch. Both are orthogonal to the choice of the underlying filter.

> **画图要点**：图必须同时体现「一条预测轨迹（A，便宜）」和「K 条采样轨迹（B，贵）」的对比，以及 λ 这个**连续旋钮**（不是开关）。

---

## II. Problem Formulation

### A. Vehicle and task

**中文：这段给图里"环境框"的内容。**

The platform is a BlueROV-class electric underwater vehicle with $M=6$ bidirectional T200 thrusters, simulated in MarineGym/Isaac Sim at $\Delta t = 0.016\,\mathrm{s}$ (62.5 Hz), with episodes of 600 steps (9.6 s). The action is the normalized throttle vector $u \in [-1,1]^{6}$. The task is tracking a lemniscate reference trajectory $p^{\mathrm{ref}}(t)$.

Vehicle dynamics follow the Fossen form
$$\dot\eta = J(\eta)\nu, \qquad M_{RB}\dot\nu + C(\nu)\nu + D(\nu)\nu + g(\eta) = B\,\tau(u),$$
where $\tau(u)$ additionally includes the T200 actuator lag (Sec. III).

### B. Keep-out safety constraint

A spherical keep-out region of radius $r_o = 0.8\,\mathrm{m}$ centred at $p_o$, with vehicle equivalent radius $r_v = 0.3\,\mathrm{m}$, defines the **clearance**
$$d(p,p_o) \;=\; \lVert p - p_o \rVert - r,\qquad r = r_o + r_v = 1.1\,\mathrm{m},$$
the safe set $\mathcal C = \{d \ge 0\}$, and a violation as $d < 0$. The keep-out region is a purely geometric constraint (no physical collision body), so that adding the safety layer does not perturb the simulated dynamics.

### C. Dynamic intercepting threat

**中文：这是场景设计，值得单独一张小图（Fig. 2）。核心是"障碍瞄准你的未来位置"，所以反应式避障原理上不够。**

Static obstacles are *not* informative for this study: a plain PPO policy with an avoidance reward drives violations to zero within 60 iterations, leaving no measurable headroom. We therefore adopt an **actively intercepting** obstacle. At each launch, the obstacle is placed laterally and given a constant velocity such that it **arrives at the vehicle's reference position $k$ steps in the future exactly when the vehicle does**:
$$p^{\mathrm{hit}} = p^{\mathrm{ref}}(t + k\Delta t),\quad k \sim \mathcal U[150,350],\quad v \sim \mathcal U[0.4,0.9]\ \mathrm{m/s},$$
$$p_o(t) = p^{\mathrm{hit}} + \hat s\, v\,(k\Delta t), \qquad \dot p_o = -\hat s\, v,$$
where $\hat s$ is a unit vector perpendicular to the reference tangent (randomly left/right). The obstacle is **re-aimed and re-launched every 200 steps** (3.2 s), producing sustained pressure. Because the threat is aimed at the *future* rendezvous point, purely reactive (distance-triggered) avoidance is by construction insufficient.

### D. Observation and reward

**中文：注意"公平性"——策略拿到了障碍位置和速度，所以我们不是在打一个瞎眼 baseline。**

$$o_t = \big[\, \underbrace{p^{\mathrm{ref}}_{t:t+4} - p_t}_{12},\ \underbrace{\text{vehicle state}}_{\text{quat, }\nu,\ \ldots},\ \underbrace{p_o - p_t}_{3},\ \underbrace{\dot p_o}_{3},\ \underbrace{\rho_{t-1}}_{1},\ \underbrace{\text{time enc.}}_{4} \,\big]$$

The policy receives the obstacle's relative position **and velocity**, i.e. it has all the information required to avoid the threat on its own. $\rho_{t-1}$ is the risk scalar of Component A (Sec. IV-B), necessarily delayed by one step because it is conditioned on the action.

The reward is a single scalar (no cost critic, no Lagrange multiplier):
$$r_t = r_{\mathrm{pose}} + r_{\mathrm{pose}}(r_{\mathrm{up}} + r_{\mathrm{spin}}) + w_e e^{-\text{effort}} + w_s e^{-\Delta\text{thr}} \;-\; w_{\mathrm{obs}}\,\big[\max(0,\, m - d_t)\big]^2 ,$$
with $w_{\mathrm{obs}}=2.0$, $m=0.5\,\mathrm{m}$. Avoidance is thus **part of the task**, and the baseline is PPO-with-avoidance-penalty rather than a safety-agnostic policy.

---

## III. Calibrated Nominal Dynamics Model

**中文：这是 A 和 B 共用的"世界模型"。图里应该画成一个被 A、B 两个框共享的底座（或一个被两者共同引用的小图标）。**

Both safety components roll the vehicle forward with the same batched, differentiable-free nominal model $f_{\mathrm{nom}}$, whose **one-step body-velocity error is below 2 %** of the simulator. The model carries the state
$$s = (p_w,\ q,\ \nu_b,\ \nu_{\mathrm{prev}},\ a_{\mathrm{prev}},\ \text{throttle},\ \text{rpm}),$$
and one step comprises three stages:

**(a) T200 actuator (stateful, first-order lag).**
$$\text{thr} \leftarrow \text{thr} + \tau_a(\mathrm{clip}(u) - \text{thr}),\quad \tau_a = 0.43,$$
$$\text{rpm} \leftarrow \alpha\,\text{rpm} + (1-\alpha)\,\text{rpm}^{\mathrm{tgt}}(\text{thr}),\quad \alpha = e^{-\Delta t / T_c},$$
$$f_i = c\cdot \mathrm{poly}^{\pm}(\text{rpm}),$$
with a dead zone at $|\text{thr}| < 0.075$ and **asymmetric forward/reverse** thrust polynomials.

**(b) Hydrodynamics.** Added mass, Coriolis, linear + quadratic damping and buoyancy, evaluated in the simulator's sign convention, with the low-pass acceleration estimate $a_{\mathrm{filt}} = 0.7\,a_{\mathrm{prev}} + 0.3(\nu - \nu_{\mathrm{prev}})/\Delta t$.

**(c) Rigid-body integration.** $\dot\nu = (B_{\mathrm{lin}}f + \text{hydro} + g_b)/M_{RB}$, integrated with $M_{RB}$ while added mass enters as a finite-difference force.

Attitude $q$ is held constant over the (short, $\le 0.32\,\mathrm{s}$) prediction horizons.

> **Reproducibility note (worth stating in the paper).** Omitting any one of (i) the T200 throttle/rpm lag, (ii) the frame convention of the buoyancy term, or (iii) the $M_{RB}$-with-finite-difference-added-mass integration scheme degrades the one-step error from $<2\%$ to $\sim 90\%$, at which point every model-based safety layer becomes useless. This is the enabling engineering result behind the method.

---

## IV. Component A — Action-Conditioned Predictive Risk Monitor

**中文：A 是"廉价前哨"。一条 rollout。图里画成 1 条绿色预测轨迹。**

### A. Predicted minimum clearance

Given the current state $s_t$, the nearest obstacle $(p_o, \dot p_o)$ and the proposed action $u^{\mathrm{rl}}_t$, Component A performs a **single** zero-order-hold rollout:
$$s_{k+1} = f_{\mathrm{nom}}(s_k, u^{\mathrm{rl}}_t),\qquad p_o^{k+1} = p_o^{k} + \dot p_o \Delta t,\qquad k = 0,\dots,H-1,$$
$$\boxed{\ \hat d_t \;=\; \min_{1\le k\le H}\Big( \lVert p_k - p_o^{k}\rVert - r \Big)\ }$$
with $H = 15$ (0.24 s). The obstacle is extrapolated at constant velocity, which is *exact* for this threat model between re-aiming events.

$\hat d_t$ is the pivotal quantity of the method: **it answers "where does the clearance go if I execute what I am about to execute", whereas a distance gate answers only "where am I now"**. Two states with identical instantaneous clearance receive different $\hat d_t$ — and hence different protection — depending on whether the proposed action accelerates toward or decelerates away from the threat.

### B. Risk scalar fed back to the policy

$$\rho_t = \mathrm{clip}\!\left(\frac{\varrho - \hat d_t}{\varrho},\,0,\,1\right), \qquad \varrho = 2.0\,\mathrm{m},$$
appended to the *next* observation, $o_{t+1}$. The one-step delay is causally necessary: $\rho_t$ depends on $u^{\mathrm{rl}}_t$, which is itself produced from $o_t$.

**The two normalisation scales are deliberately different**: the observation uses a wide $\varrho = 2.0\,\mathrm{m}$ so the policy receives a non-zero, informative risk gradient well before any intervention, while the takeover ramp (Sec. VI) uses a narrow $0.6\,\mathrm{m}$ so authority is surrendered only when genuinely required. *Information early, authority late.*

### C. Cost

One rollout per environment per step (versus $K=128$ for the MPPI filter), i.e. the decision of whether to pay for the filter is itself nearly free — and is made on a predictive rather than a geometric criterion.

---

## V. Component B — MPPI Predictive Safety Filter

**中文：B 是"贵但强"的部分。图里画成 K 条灰色扇形采样轨迹，其中几条穿进球体被标红。**

When $\lambda_t > 0$, Component B samples $K = 128$ perturbed action sequences of length $N = 20$ (0.32 s) around the proposal,
$$a^{(k)}_n = \mathrm{clip}\big(u^{\mathrm{rl}}_t + \varepsilon^{(k)}_n,\,-1,\,1\big),\qquad \varepsilon \sim \mathcal N(0,\sigma^2 I),\ \sigma = 0.4,$$
rolls each through the **same** $f_{\mathrm{nom}}$ (batched over environments $\times$ samples), extrapolating the obstacle at constant velocity, and scores it with
$$J_k \;=\; \underbrace{w_{\mathrm{coll}} \sum_{n=1}^{N} \Big[\max\big(0,\ r - \min_j \lVert p^{(k)}_n - p_{o,j}^{\,n}\rVert\big)\Big]^2}_{\text{keep-out penetration}} \;+\; \underbrace{w_{\mathrm{track}} \sum_{n=0}^{N-1} \lVert a^{(k)}_n - u^{\mathrm{rl}}_t \rVert^2}_{\text{deviation from the policy}},$$
with $w_{\mathrm{coll}} = 5.0$, $w_{\mathrm{track}} = 1.0$. The safe command is the importance-weighted first action,
$$w_k = \mathrm{softmax}_k\!\left(-\frac{J_k - \min_j J_j}{\varsigma}\right),\ \varsigma = 0.05, \qquad u^{\mathrm{safe}}_t = \sum_{k} w_k\, a^{(k)}_0 .$$

Two properties matter for the architecture:

1. **It is a filter, not a planner.** $J_k$ contains no trajectory-tracking term; task performance remains entirely the responsibility of the model-free policy, and the model-based layer only enforces safety. This resolves the apparent tension of using a model inside a model-free method: *precision is learned, safety is certified against a coarse but calibrated model.*
2. **It self-effaces when safe.** The penetration term is zero for any rollout that stays outside the sphere, so in safe states $J_k$ reduces to the deviation term and $u^{\mathrm{safe}} \approx u^{\mathrm{rl}}$. The filter and the policy therefore never "fight" in the low-risk regime, which is precisely what makes a *proportional* blend well behaved.

---

## VI. Risk-Proportional Soft Takeover

**中文：这是方法的核心机制，图里必须有一个独立的小插图（λ 斜坡函数）。**

The predicted clearance is mapped to the takeover coefficient by a fixed ramp
$$\boxed{\ \lambda_t \;=\; \mathrm{clip}\!\left( \frac{h_{\mathrm{hi}} - \hat d_t}{h_{\mathrm{hi}} - h_{\mathrm{lo}}},\ 0,\ 1 \right),\qquad h_{\mathrm{lo}} = 0.0\,\mathrm{m},\ \ h_{\mathrm{hi}} = 0.6\,\mathrm{m}\ }$$
and the executed command is
$$u_t = (1-\lambda_t)\,u^{\mathrm{rl}}_t + \lambda_t\, u^{\mathrm{safe}}_t .$$

| regime | $\hat d_t$ | $\lambda_t$ | behaviour |
|---|---|---|---|
| **Autonomous** | $\ge h_{\mathrm{hi}}$ | $0$ | policy executed unchanged; **MPPI is not evaluated at all** |
| **Shared** | $(h_{\mathrm{lo}}, h_{\mathrm{hi}})$ | $(0,1)$ | authority transferred in proportion to predicted risk |
| **Protective** | $\le h_{\mathrm{lo}}$ | $1$ | filter fully in command |

Because $\lambda_t = 0$ implies the MPPI stage is skipped, the expected computational overhead is $\mathbb E[\lambda > 0]$ times the filter cost; the measured engagement rate of the converged policy is $\approx 0.10$.

**Why proportional rather than binary.** We identify three effects:

* **(i) Continuity of the applied command.** A binary shield switches between $u^{\mathrm{rl}}$ and $u^{\mathrm{safe}}$ discontinuously and chatters near the threshold, exciting thrust reversals. $\lambda_t$ is Lipschitz in $\hat d_t$, hence $u_t$ is continuous.
* **(ii) Mildness of the induced off-policy shift (learning-side effect).** During training the *executed* command is the one recorded in the rollout, so the policy gradient is evaluated at $u_t$ while the behaviour log-probability was recorded at $u^{\mathrm{rl}}_t$. The resulting importance ratio deviates from unity by an amount proportional to $\lVert u_t - u^{\mathrm{rl}}_t \rVert = \lambda_t \lVert u^{\mathrm{safe}}_t - u^{\mathrm{rl}}_t\rVert$. A proportional blend keeps this deviation small and graded; a binary switch injects a full-magnitude jump whenever it fires. This predicts that proportional takeover should dominate binary takeover *under every other configuration*, which is what the ablation matrix shows.
* **(iii) Unimodality of the executed action distribution.** The executed action remains a smooth deformation of the policy's own distribution rather than a mixture of two disjoint modes.

---

## VII. Policy Learning

**中文：图的下半部分（训练回路）。重点是"什么都没加"——纯 PPO。**

The policy is trained with **unmodified PPO** on the scalar reward of Sec. II-D: no cost critic, no Lagrange multiplier, no constraint-satisfaction machinery, and — importantly — **no penalty on the filter's correction**. The safety filter is present *during training as well as deployment*, so that (a) exploration is protected from the first iteration and (b) there is no train–deploy distribution shift in the action channel.

We explicitly evaluated the natural alternative of *internalising* safety by penalising the filter's intervention,
$$r_t \leftarrow r_t - w_C \lVert u^{\mathrm{safe}}_t - u^{\mathrm{rl}}_t \rVert^2,$$
in both a fixed-weight and a self-extinguishing adaptive form ($w_C = \kappa\,\mathrm{EMA}[\lambda]$). Both **degrade** safety monotonically with their complexity (Sec. Experiments); we therefore report internalisation as a negative result and exclude it from the final method.

---

## VIII. Degenerate Variants Used in the Ablation

**中文：这张表是消融的骨架，也建议在框架图旁边配一个 2×2 小图（Fig. 3）。**

The architecture exposes two orthogonal design axes — **the gating signal** and **the takeover law** — yielding a $2\times 2$ factorial design in which our method is one cell:

| | **Binary takeover** ($u = \mathbb 1[\cdot]\,?\,u^{\mathrm{safe}} : u^{\mathrm{rl}}$) | **Proportional takeover** ($u = (1{-}\lambda)u^{\mathrm{rl}} + \lambda u^{\mathrm{safe}}$) |
|---|---|---|
| **Geometric gate** (current distance $d_t$) | distance-triggered MPPI | $\lambda$ from $d_t$ |
| **Predictive gate** (predicted $\hat d_t$, Component A) | risk-triggered MPPI | **RP-PSF (ours)** |

Additional reference points: a single-step HOCBF action shield (reactive projection), and PPO with the avoidance reward but no filter.

---

# 第二部分：Framework 示意图绘制规格

> **给画图的人**：下面把图拆成 3 张。**Fig. 1 是主图（必须画好）**，Fig. 2 和 Fig. 3 是可选的小图。
> 所有符号请严格按「符号表」写，不要自己改字母。

## 符号表（图中标注必须与此一致）

| 符号 | 含义 | 维度/取值 |
|---|---|---|
| $o_t$ | 观测 | 向量 |
| $\pi_\theta$ | PPO 策略（MLP） | — |
| $u^{\mathrm{rl}}_t$ | 策略提议动作（归一化推力） | $[-1,1]^6$ |
| $f_{\mathrm{nom}}$ | 名义动力学模型（误差 <2%） | 两个组件共用 |
| $\hat d_t$ | **预测**最小间隙 | 标量，单位 m |
| $\rho_t$ | 风险标量（进观测） | $[0,1]$ |
| $\lambda_t$ | **接管系数** | $[0,1]$ |
| $u^{\mathrm{safe}}_t$ | MPPI 安全动作 | $[-1,1]^6$ |
| $u_t$ | **实际执行动作** | $[-1,1]^6$ |
| $p_o,\ \dot p_o$ | 障碍位置 / 速度 | 3D |
| $r$ | 安全半径 $r_o+r_v = 1.1$ m | — |
| $H=15$ | A 的预测步数（0.24 s） | — |
| $K=128,\ N=20$ | B 的采样数 / 步数（0.32 s） | — |
| $h_{\mathrm{lo}}=0,\ h_{\mathrm{hi}}=0.6$ | λ 斜坡的两个阈值（m） | — |

---

## Fig. 1 — 主框架图（双栏宽，建议 2:1 左右的横向构图）

### 整体构图

**一条从左到右的主回路 + 一个下方的训练回路 + 两个插图。**

```
┌──────────────────────────── Fig.1 版面 ────────────────────────────┐
│                                                                     │
│  [环境 Env]  →  [观测 o_t]  →  [策略 π_θ]  ──u^rl──┬──────────┐    │
│      ▲                ▲                             │          │    │
│      │                │(虚线,z⁻¹)                    ▼          ▼    │
│      │                └────ρ_t────┐        ┌────────────┐  ┌────────┐│
│      │                            │        │ A 风险预测器│  │        ││
│      │                            └────────│  1 条rollout│  │ (直通) ││
│      │                                     │  → d̂_t      │  │        ││
│      │                                     └──────┬─────┘  │        ││
│      │                          ┌─────────────────┴───┐    │        ││
│      │                          │ λ 斜坡 (插图 b)      │    │        ││
│      │                          │ λ=clip((h_hi−d̂)/h_hi)│    │        ││
│      │                          └──────┬──────────────┘    │        ││
│      │                        λ=0 跳过 │ λ>0               │        ││
│      │                     ┌───────────┴─────────┐         │        ││
│      │                     │ B  MPPI 安全滤波器   │         │        ││
│      │                     │ K=128 条 rollout     │         │        ││
│      │                     │ → u^safe             │         │        ││
│      │                     └───────────┬─────────┘         │        ││
│      │                                 ▼        ┌──────────┘        ││
│      │                          ┌──────────────────────┐            ││
│      └──────────u_t─────────────│ ⊗ 比例融合            │            ││
│                                 │ u=(1−λ)u^rl+λu^safe  │            ││
│                                 └──────────────────────┘            ││
│                                                                     │
│  ┌─────────────── 训练回路（灰色虚线，画在下方）───────────────┐    │
│  │  Rollout buffer (o_t, u_t, r_t) → PPO update → π_θ           │    │
│  │  纯 PPO：无 cost critic / 无拉格朗日乘子 / 无修正惩罚         │    │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 逐框内容（框内要写什么字、画什么小图）

**① 环境框 "Environment (MarineGym / Isaac Sim)"**
- 小图：一个 BlueROV 图标 + 一条 8 字形（lemniscate）参考轨迹（细白/灰线）+ 一个半透明红色球体（keep-out，标 $r$）+ 一个带速度箭头的移动障碍。
- 框角标注：`62.5 Hz, Δt = 0.016 s`。

**② 观测框 "Observation $o_t$"**
- 画成横向排列的小色块（chips），依次写：
  `tracking error` | `vehicle state` | `p_o − p` | `ṗ_o` | **`ρ_{t−1}` (risk)** | `time enc.`
- 最后一块 `ρ_{t−1}` 用**橙色**（与安全通路同色），其余用蓝色，表示只有它来自安全层。

**③ 策略框 "$\pi_\theta$ — PPO policy (MLP)"**
- 简单画 3 层小圆点网络即可。
- 输出箭头标 **$u^{\mathrm{rl}}_t \in [-1,1]^6$**，箭头**蓝色加粗**。

**④ 分叉**：$u^{\mathrm{rl}}_t$ 从策略出来后**分成三路**（这是图的关键，务必画清楚）：
   - 路 1 → 进 A（作为被预测的对象）
   - 路 2 → 进 B（作为采样中心）
   - 路 3 → 直接进融合节点（作为 $(1-\lambda)$ 那一项）

**⑤ A 框 "Component A — Predictive Risk Monitor"**
- 副标题小字：`single rollout, H = 15 (0.24 s)`
- 框内小图：一条**绿色实线**预测轨迹从载具出发，旁边一个红色球（障碍，带虚线箭头表示它也在匀速外推），标出轨迹与球面最近处的一小段**双箭头线段**，标注 **$\hat d_t$**。
- 输出两条箭头：
  - **橙色实线** → λ 斜坡框，标 $\hat d_t$
  - **橙色虚线**（向上、绕回观测框）→ 标 **$\rho_t$**，并在虚线中间画一个小方框写 **$z^{-1}$**，旁注小字「one-step delay (causal)」。

**⑥ λ 斜坡框 "Takeover schedule"**
- 框内直接嵌插图 (b)（见下），或写公式 $\lambda_t = \mathrm{clip}\big((h_{\mathrm{hi}}-\hat d_t)/(h_{\mathrm{hi}}-h_{\mathrm{lo}}),0,1\big)$。
- 输出一条橙色箭头标 **$\lambda_t$**，**分两支**：一支进 B 框（作为门控），一支进融合节点（作为混合系数）。
- 在进 B 的那支旁边加一个**小开关图标**并标注：**`λ = 0 → MPPI skipped (no cost)`**。这是"廉价前哨"故事的可视化，别漏。

**⑦ B 框 "Component B — MPPI Predictive Safety Filter"**
- 副标题小字：`K = 128 rollouts, N = 20 (0.32 s)`
- 框内小图：从载具发出**一把灰色细线扇形**（K 条采样轨迹），其中 2–3 条穿进红球内部的部分用**红色高亮**（表示 penetration 代价），另有 1 条**粗橙线**表示加权结果。
- 输出箭头标 **$u^{\mathrm{safe}}_t$**（橙色加粗）。

**⑧ 共享模型底座（重要，容易漏）**
- 在 A 框和 B 框**之间或下方**画一个横跨两者的窄条框："**Calibrated nominal model $f_{\mathrm{nom}}$ — <2 % one-step error (T200 lag + full hydrodynamics)**"，用**两条细线分别连到 A 和 B**，表示两者共用同一个模型。
- 这个底座建议用**浅灰底 + 橙色边框**。

**⑨ 融合节点 "Risk-proportional blending"**
- 画成一个圆形节点，里面是 **⊗** 或调音台推子（fader）图标。
- 旁边写公式：$u_t = (1-\lambda_t)u^{\mathrm{rl}}_t + \lambda_t u^{\mathrm{safe}}_t$
- **两个输入箭头的粗细最好按 $(1-\lambda)$ / $\lambda$ 做视觉暗示**（蓝箭头粗、橙箭头细，示意典型情况 λ 较小）。
- 输出 **$u_t$**（紫色/深色加粗）→ 回到环境框，箭头上可加小字 `thrust allocation → 6 × T200`。

**⑩ 训练回路（下方，全部灰色虚线）**
- `Rollout buffer` → `PPO update (clipped surrogate)` → 虚线回到 $\pi_\theta$。
- buffer 框内小字标注：**stores the executed action $u_t$**（这条如果要在论文里讲学习动力学，就必须画出来）。
- 回路旁边放一个"否定清单"小标签：**`No cost critic · No Lagrange multiplier · No correction penalty`**。

### 配色（建议，可整体替换但要保持分组）

| 通路 | 颜色 | 含义 |
|---|---|---|
| 学习通路（obs → π → $u^{\mathrm{rl}}$） | **蓝色** | model-free，负责性能 |
| 安全通路（A → λ → B → $u^{\mathrm{safe}}$） | **橙色** | model-based，负责安全 |
| 执行动作 $u_t$ | **深紫/黑** | 两者的混合 |
| 训练回路 | **灰色虚线** | 只在训练时存在 |
| 障碍/危险 | **红色**（半透明球体） | — |

> 一句话审美目标：**读者一眼看出"蓝色主干 + 橙色旁路，旁路通过一个连续旋钮 λ 按比例汇入主干"**。不要把 A、B 画成串在主干上的两个"关卡"——那会变成一张普通 shielding 图，丢掉本文的卖点。

### 插图 (b) — λ 斜坡函数（嵌在 Fig.1 右上角或 ⑥ 框内）

- 小坐标系：横轴 $\hat d_t$ (m)，纵轴 $\lambda$，$\lambda \in [0,1]$。
- 曲线：从 $(0,\,1)$ 水平向左延伸（$\hat d \le 0$ 时 $\lambda = 1$），在 $\hat d \in [0, 0.6]$ 线性下降到 0，$\hat d \ge 0.6$ 保持 0。
- 横轴标出 $h_{\mathrm{lo}} = 0$ 与 $h_{\mathrm{hi}} = 0.6$ 两条竖虚线。
- 三个区间用淡背景色 + 文字标注：**`Protective (filter in command)` / `Shared authority` / `Autonomous (policy only, MPPI skipped)`**。
- 可选：用一条**灰色阶跃虚线**叠画二值 shield 作对照，图例写 `binary shield (ablation)`。这一笔非常值钱，直接把"我们的贡献"画进图里。

---

## Fig. 2 — 场景示意图（单栏小图，可选但推荐）

目的：说明"障碍瞄准的是**未来**位置"，因此反应式避障不够。

- 画一条 8 字形参考轨迹。
- 载具在当前位置 $p_t$（蓝色）；在轨迹上标出 **$k$ 步之后的参考点 $p^{\mathrm{hit}}$**（打叉标记），$k\Delta t \in [2.4, 5.6]$ s。
- 从侧向画一条直线箭头（红色）指向 $p^{\mathrm{hit}}$，标 $v \in [0.4,0.9]$ m/s，起点画障碍球。
- 用两条虚线分别标注载具与障碍到 $p^{\mathrm{hit}}$ 的路径，并注明 **"simultaneous arrival"**。
- 角落小字：`re-aimed every 200 steps (3.2 s)`。

---

## Fig. 3 — 为什么"预测门控"优于"距离门控"（单栏双面板小图，可选）

目的：一眼说明 $\hat d_t$ 与 $d_t$ 的区别。

- **左右两个面板，载具与障碍的当前位置完全相同**（即当前距离 $d_t$ 相同，标出来）。
- 左面板：$u^{\mathrm{rl}}$ 指向障碍 → 绿色预测轨迹撞进球体 → $\hat d < 0$ → **$\lambda \approx 1$**。
- 右面板：$u^{\mathrm{rl}}$ 已在制动/偏离 → 预测轨迹绕开 → $\hat d$ 大 → **$\lambda = 0$**。
- 底部一行结论字：**"Same distance, different risk — the gate must be conditioned on the action."**

---

## 画图注意事项清单（交付前自查）

- [ ] $u^{\mathrm{rl}}$ 的**三路分叉**画出来了（进 A、进 B、进融合节点）——最常被漏。
- [ ] A 是 **1 条**轨迹、B 是 **K 条**轨迹，视觉上必须有明显疏密对比。
- [ ] A 与 B **共享同一个 $f_{\mathrm{nom}}$** 有被画出来。
- [ ] λ 是一个**连续旋钮**（有斜坡插图），不是开关。
- [ ] `λ = 0 → MPPI skipped` 的跳过路径画出来了。
- [ ] $\rho$ 回到观测的虚线上标了 $z^{-1}$（一步延迟）。
- [ ] 训练回路是**灰色虚线**且标了"训练时存在"，不会被误读为部署时也要跑 PPO。
- [ ] 图中所有字母与「符号表」一致，没有自创符号。
- [ ] 矢量格式（PDF/SVG），字号 ≥ 7 pt（IEEE 双栏缩放后仍可读），黑白打印可区分（靠线型区分，不只靠颜色）。
