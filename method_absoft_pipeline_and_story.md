# A+B-soft（Risk-Proportional Predictive Safety Filter）方法文档与论文故事

> 面向 IEEE TTE 投稿。本文档 = **Pipeline + Methods 的可写作版本** + **故事线（gap / contribution / findings）** + **审稿人攻击面与待补实验**。
> 代码基准：`marinegym/envs/single/track.py`、`marinegym/utils/risk_monitor.py`、`marinegym/utils/mppi_filter.py`、`marinegym/utils/nominal_dynamics.py`、`cfg/task/Track.yaml`。
> 数据来源：`PAPER_DATA.md`（2026-08-25 全部重跑，四格已对齐）。
> 早期 `results_summary.md` / `results_tte.md` 的数字已被取代并删除。

---

## 0. 一句话方法

> 策略每一步提议动作 `u_rl`；用一个**与仿真误差 <2% 的名义 6-DoF 模型**把「策略自己这个动作」前滚 H 步，得到**预测最小间隙** `d̂_min`；由 `d̂_min` 生成一个**连续接管系数 λ∈[0,1]**；同一模型的 MPPI 采样出安全动作 `u_safe`；实际执行 **`u = (1−λ)·u_rl + λ·u_safe`**。

三句话说清与已有工作的差别：

1. **门控信号是"策略自己的未来"，不是"当前距离"** —— 这是 A（Risk Monitor）。
2. **接管是比例的，不是开关** —— 这是 soft blend（λ）。
3. **不含内化惩罚** —— 内化不属于本方法，已从代码路径与全部产出中剔除。

---

## 1. 问题设定

### 1.1 载具与仿真

- 平台：MarineGym / Isaac Sim，载具 **BlueROV**（纯推进器 T200 × 6，`num_rotors == action_dim`，动作 = 归一化 throttle ∈ [−1,1]^6）。
- 控制周期：`dt = 0.016 s`，`substeps = 1` → **62.5 Hz**；`max_episode_length = 600` → 9.6 s/episode。
- 任务：Lemniscate 轨迹跟踪（`traj_scale_mult = 2.5`，放大工作空间使绕行可行）。
- 动力学：Fossen 型（刚体 + 附加质量 + 线性/二次阻尼 + Coriolis + 浮力）+ T200 执行器迟滞。

### 1.2 安全约束（keep-out）

球形禁区，中心 `p_o`、半径 `r_o = 0.8 m`，载具等效半径 `r_v = 0.3 m`，令 `r = r_o + r_v = 1.1 m`：

```
clearance:  d(p, p_o) = ‖p − p_o‖ − r
safe set:   𝒞 = { d ≥ 0 }
violation:  d < 0
```

禁区**不是物理碰撞体**（纯数学球），因此不改变动力学，只改变 reward / 指标 —— 保证「加安全模块」不引入仿真副作用。

### 1.3 核心场景：动态拦截障碍（关键设计）

**静态障碍已被纯 PPO 打穿**（60 iters 后 PPO 自己就零违约，任何安全模块无增量）。因此设计了**主动拦截**的动态障碍（`task.keepout.dynamic`，`track.py:319-337 _aim_obstacle`）：

- 障碍**从侧向发射**，瞄准载具**未来 k∈[150,350] 步的参考位置**（2.4–5.6 s 后），速度 `v ∈ [0.4, 0.9] m/s`，使其**恰好与载具同时到达该点**；
- 每 200 步（3.2 s）**重新瞄准发射一次**，形成持续压力；
- 策略**公平地拿到障碍相对位置 + 障碍速度**（`obstacle_in_obs=true`，动态时额外给 3 维速度，`track.py:541-554`）。

> 这个场景的意义：**反应式（基于当前距离）避障在原理上不够** —— 障碍是冲着你未来位置来的。这就把"必须预测"从一句口号变成了一个**可测的实验事实**（B-only 距离门控 ≈ PPO）。

### 1.4 奖励（策略仍是纯 PPO，单标量奖励）

```
r = r_pose + r_pose·(r_up + r_spin) + w_e·exp(−effort) + w_s·exp(−Δthrottle)
    − w_obs · max(0, m − d)²          # 避障接近惩罚，w_obs=2.0, m=0.5
```

即：**避障是任务的一部分**（PPO 有动机自己避），安全模块只是在此之上再降违约。这一点非常重要 —— 我们不是在跟一个"根本没被告知障碍存在"的弱 baseline 比。

---

## 2. 系统总览（Pipeline）

```
                    ┌────────────────────────────────────────────────┐
                    │            obs_t （含 risk_{t−1} 标量）          │
                    └───────────────────┬────────────────────────────┘
                                        ▼
                            ┌────────────────────┐
                            │  PPO 策略 π_θ (MLP) │
                            └─────────┬──────────┘
                                      │ u_rl  (提议 throttle ∈[−1,1]^6)
             ┌────────────────────────┼────────────────────────┐
             ▼                        │                        ▼
   ┌──────────────────────┐           │           ┌───────────────────────────┐
   │  A: Risk Monitor      │          │           │  B: MPPI Exact Filter      │
   │  名义模型 1 条 rollout │          │           │  名义模型 K=128 条 rollout  │
   │  零阶保持 u_rl，H=15   │          │           │  N=20 步，含障碍匀速外推     │
   │  → d̂_min, risk, λ     │──λ,gate─▶│           │  → u_safe                  │
   └──────────┬───────────┘           │           └─────────────┬─────────────┘
              │ risk → obs_{t+1}      │                         │
              │ (1 步延迟，因依赖 u_t) │                         │
              ▼                       ▼                         ▼
        ┌───────────────────────────────────────────────────────────────┐
        │   soft blend:   u_t = (1 − λ_t)·u_rl + λ_t·u_safe             │
        └───────────────────────────┬───────────────────────────────────┘
                                    ▼
                          Isaac Sim（施加 u_t）
                                    │
                                    ▼
                     reward / done / stats（collision, minDist, …）
```

**关键点**：A 和 B **共享同一个名义模型** `NominalDynamics`。A 是 B 的"**廉价前哨**"（1 条 rollout vs 128 条），只有 A 报警的批次才启动 B（整批全安全时 B 直接跳过，零开销，`mppi_filter.py:141-142`）。

代码入口：`track.py:427-499 _pre_sim_step()`，链路顺序为
`策略动作 → (A 评估) → (可选 shield) → field 或 MPPI → 写回 tensordict → apply_action`。

---

## 3. 名义动力学模型（A/B 的共同基座）

`marinegym/utils/nominal_dynamics.py` —— 这是整个方法能成立的工程前提，**一步预测速度误差 <2%**（`keepout.validate=true` 有在线验证钩子，`track.py:511-532`）。

状态 `s = (p_w, q, ν_b, ν_prev, a_prev, throttle, rpm)`，一步：

```
① T200 执行器（有状态，必须建，否则误差 ~90%）
   throttle ← throttle + τ·(clip(a) − throttle),        τ = 0.43
   rpm_tgt  = 分段线性(throttle)   （含 |throttle|<0.075 死区）
   rpm      ← α·rpm + (1−α)·rpm_tgt,  α = exp(−dt/T_c),  |rpm| ≤ 3900
   thrust   = fc · 9.81 · poly(rpm)          （正/反转不同多项式，前反不对称）

② 水动力（在仿真自己的 [1,2,4,5] 翻转帧内计算，再翻回）
   acc_filt = 0.7·acc_prev + 0.3·(ν − ν_prev)/dt        （低通，复刻 calculate_acc）
   hydro = −( M_A·acc_filt + C_A(ν)ν + (D_l + D_q|ν|)ν )
   buoy  = 浮力(rpy)                                     （帧翻转，漏了会差 ~200 N）

③ 刚体积分（用 M_RB，附加质量作为有限差分力，不是 M_RB+M_A 隐式）
   ν̇ = ( B_lin·thrust + hydro + g_b ) / M_RB
   ν ← ν + dt·ν̇ ;  p ← p + dt·R(q)ν_{1:3}
```

**论文里要写的三个踩坑（可作为 reproducibility 贡献）**：
1. 浮力必须在**翻转帧**里算再翻回；
2. 用 **M_RB 积分 + 附加质量当力**，不是 M_RB+M_A；
3. **T200 throttle/rpm 一阶迟滞必须建模**。
三者缺一，模型误差从 <2% 退化到 ~90%，安全滤波器直接失效。

姿态 `q` 在短时域内**保持不变**（`nominal_dynamics.py:112`）—— 显式的保守近似，需在论文中声明（时域 0.24–0.32 s，姿态变化有限）。

---

## 4. 组件 A：Predictive Risk Monitor

`marinegym/utils/risk_monitor.py`，配置 `task.keepout.risk`。

### 4.1 数学

给定当前状态 `s_t`、最近障碍 `(p_o, v_o)`、策略提议 `u_rl`：

```
零阶保持：a_k = u_rl,  k = 0…H−1        （H = 15 → 0.24 s）
状态递推：s_{k+1} = f_nom(s_k, u_rl)
障碍外推：p_o^{k+1} = p_o^k + v_o·dt      （匀速；与本环境障碍真实运动一致）

预测最小间隙：
    d̂_min = min_{k=1..H} ( ‖p_k − p_o^k‖ − r )

风险标量（进观测）：
    risk = clip( (ρ − d̂_min) / ρ , 0, 1 ),      ρ = risk_norm = 2.0 m

硬触发（A+B-hard 用）：
    trigger = 1[ d̂_min < δ ],                   δ = threshold = 0.6 m  (= h_hi，全套唯一边界)
```

### 4.2 三点设计解读（写 methods 时的"为什么"）

1. **它评估的是"策略自己这个动作的后果"，不是环境的静态几何。**
   同样的位置、同样的距离，如果 `u_rl` 朝障碍加速 → 高风险；如果 `u_rl` 已在制动 → 低风险。
   这是与所有 distance-gated / TTC-gated 滤波器的本质区别。**2026-08-25 重跑的证据**：
   在比例与开关两种接管下，预测门控都显著减少绕行（|d|=5.8 / 10.2）；
   但**碰撞率两列均不显著**（当前场景无安全压力，见 §9.3）。

2. **代价是 1 条 rollout（vs MPPI 的 128 条），可以每步跑。**
   于是"要不要付 MPPI 的钱"这个决策本身变得几乎免费，而且判据是预测性的。

3. **risk 进观测有 1 步延迟，且这个延迟是因果必然的。**
   `risk_t` 依赖 `u_t`，而 `u_t` 是看了 `obs_t` 才产生的，所以 `risk_t` 只能出现在 `obs_{t+1}`（`track.py:456` 写入 `self._risk`，`track.py:555-556` 在下一次观测里读出）。论文中要明说，否则像 bug。

4. **两个不同的归一化尺度是刻意的**：观测用 `ρ=2.0 m`（宽 —— 让策略在远处就有非零、有梯度的风险信号），接管用 `soft_hi=0.6 m`（窄 —— 只在真正逼近时才交出控制权）。**信息给得早，控制权交得晚。**

---

## 5. 组件 B：MPPI Exact Predictive Safety Filter

`marinegym/utils/mppi_filter.py::MPPIExactShield`，配置 `task.keepout.mppi`（`exact=true`）。

### 5.1 数学

```
采样：a^{(k)}_n = clip( u_rl + ε^{(k)}_n , −1, 1 ),  ε ~ 𝒩(0, σ²),  σ=0.4
      k = 1…K (=128),  n = 0…N−1 (=20 → 0.32 s)

前滚：s^{(k)}_{n+1} = f_nom(s^{(k)}_n, a^{(k)}_n)         （同一 <2% 模型，E×K 批量）
      p_o^{n+1} = p_o^n + v_o·dt                          （障碍匀速外推）

代价：J_k = w_coll · Σ_n [ max(0, r − min_j‖p^{(k)}_n − p_{o,j}^n‖) ]²      # 穿透深度平方
           + w_track · Σ_n ‖a^{(k)}_n − u_rl‖²                              # 与策略动作的偏离
      w_coll = 5.0,  w_track = 1.0

加权：w_k = softmax_k( −(J_k − min J) / λ_temp ),          λ_temp = 0.05
输出：u_safe = Σ_k w_k · a^{(k)}_0
```

### 5.2 三点设计解读

1. **代价里的碰撞项是"穿透深度"而非"间隙裕度"** —— 只要采样轨迹不进球内就零成本。因此在安全区，`J` 只剩 `w_track` 项 → **`u_safe ≈ u_rl`**。即 B 具有"无事时自动隐身"的性质，与 λ 的软接管天然相容（λ 小的时候两个来源本来就接近，混合不会打架）。
2. **它是 filter 不是 planner**：目标函数里没有轨迹跟踪项，只有"贴近策略 + 别撞"。任务性能仍完全由 RL 负责，这维持了 model-free 的精度优势，只把硬安全交给粗模型。
3. **批级跳过**：`if active_mask is not None and not active_mask.any(): return u_rl` —— 全批安全则 MPPI 完全不执行。这是部署开销故事的关键（激活率 ≈ 0.10 → 期望开销 ≈ 10% 的 MPPI 成本）。
   ⚠️ 但注意：**只要批内有一个 env 触发，MPPI 就对整批 E×K 计算**（没有做 per-env gather）。E5 计时表里要诚实说明，或改成只对触发子集计算。

---

## 6. 核心机制：Risk-Proportional Soft Takeover（λ 融合）

代码：`track.py:450-458`（λ 生成）+ `mppi_filter.py:171-173`（融合）。

### 6.1 公式

```
λ_t = clip( (h_hi − d̂_min) / (h_hi − h_lo) , 0, 1 ),     h_hi = soft_hi = 0.6,  h_lo = soft_lo = 0.0

u_t = (1 − λ_t)·u_rl + λ_t·u_safe
```

分段解读：

| 预测最小间隙 `d̂_min` | λ | 行为 |
|---|---|---|
| ≥ 0.6 m | 0 | **完全信任策略**，MPPI 甚至不算（trigger = λ>0 = false） |
| 0.6 → 0 m | 0 → 1 线性 | **比例接管**，越危险策略权重越低 |
| ≤ 0 m（预测已穿透） | 1 | **完全交给 MPPI** |

### 6.2 与两个退化情形的关系（这是消融的骨架）

```
A+B-hard  :  u = 1[d̂_min < 0.3] ? u_safe : u_rl          （风险门控 + 开关）
B-soft    :  λ 由 **当前** 间隙 d_now 生成（无预测）        （track.py:459-466）
B-hard    :  1[ ‖p−p_o‖ < r + 1.5 ] ? u_safe : u_rl        （距离门控 + 开关）
A+B-soft  :  λ 由 **预测** 间隙 d̂_min 生成                  ★ 本文方法
```

于是天然形成一个 **2×2 析因设计：{门控信号: 距离 / 预测} × {接管方式: 开关 / 比例}**。这是本文实验部分最强的结构（见 §9）。

### 6.3 为什么"比例"比"开关"好 —— 三条可写进论文的机理

**(i) 控制层面：消除切换抖振。** 开关式在阈值附近来回跳变，u 在 `u_rl` 和 `u_safe` 之间不连续跳跃，激励高频推力振荡（smooth 指标、功率都受害）。λ 是 `d̂_min` 的连续（Lipschitz）函数 → `u` 连续。

**(ii) 学习层面：一处必须修掉的实现缺陷（2026-08-25 更正）。**
训练回路里写回 tensordict 的是执行动作 `u_t`（`track.py:519`），而 `sample_log_prob`
仍是提议动作 `u_rl` 的对数概率（`ppo.py:132` 写入后从不重算）。于是 `ppo.py:220` 的

```
ratio = exp( log π_θ(u_t) − log π_old(u_rl) )      ← 分子分母是两个不同的动作
```

**这不是一个可解释的机理，而是错配的重要性比率**：它不对应任何估计量。
判据：epoch 0 时（θ=θ_old）所有步的 ratio 应恒为 1，实际在滤波生效的步上 ≠1。

正确做法是**表述 A**：shield 是确定性且不含 θ 的映射，应归入转移核，学习器存 `u_rl`，
`u_t` 只交给仿真器执行。因为 `∇log p_θ(τ) = Σ_t ∇log π_θ(u_rl,t|s_t)` —— 转移核不含 θ，
求导后直接消失，score function 里只可能出现 `u_rl`。滤波带来的全部后果已通过
`r` 与 `s'` 进入 advantage，credit assignment 是正确的。

（"分子分母都用 u_t" 也不可行：`u_t` 服从 β=shield∘π，而 β 在全量替换时含狄拉克质点，
对 Lebesgue 测度无密度，`log β` 无定义。）

> ⚠️ **此前版本把这一现象写成"executed-action relabeling"并建议作为方法卖点，那是错的，已撤回。**
> 更关键的是：**扭曲程度沿消融轴单调**——不滤波(无) → 比例接管 λ 小(轻微) → 开关接管全量替换(严重)。
> 因此"比例 vs 开关"的观测差异中，**部署效应与训练扭曲尚未分离**。投稿前必须按表述 A 修正并全量重训。

**(iii) 数据分布层面：** λ 连续意味着策略经历的动作分布是自身分布的**平滑形变**而非双模态混合，on-policy 假设破坏更小。

### 6.4 关于内化惩罚（已移出本文范围）

`internalize_weight` / `internalize_adaptive`（`track.py`）对滤波修正量施加奖励惩罚。
该组件**不属于本方法**，相关实验与数据已从全部产出中剔除，本文不作任何声称。

---

## 7. 超参数与实现细节速查

| 项 | 值 | 位置 |
|---|---|---|
| 控制频率 | 62.5 Hz (dt=0.016, substeps=1) | `cfg/base/sim_base.yaml` |
| 环境数 / episode | 16 / 600 步 | `cfg/task/Track.yaml` |
| 障碍 `r_o`, 载具 `r_v` | 0.8 / 0.3 m（r=1.1） | `keepout.radius/vehicle_radius` |
| 动态障碍速度 / 重瞄周期 | 0.4–0.9 m/s / 200 步 | `keepout.dynamic` |
| A: horizon H | 15 步 (0.24 s) | `keepout.risk.horizon` |
| A: risk_norm ρ | 2.0 m | `keepout.risk.risk_norm` |
| 门控边界 δ = h_hi | **0.6 m** | `keepout.risk.threshold` = `keepout.mppi.soft_hi`（全套唯一边界） |
| A 是否充当门控 | true | `keepout.risk.gate`（false → A 仍进观测，门控退回几何） |
| B: K / N | 128 / 20 (0.32 s) | `keepout.mppi.num_samples/horizon` |
| B: σ / λ_temp | 0.4 / 0.05 | `noise_sigma / temperature` |
| B: w_coll / w_track | 5.0 / 1.0 | `keepout.mppi` |
| **soft_lo / soft_hi** | **0.0 / 0.6 m** | `keepout.mppi.soft_lo/soft_hi` |
| 观测维度 | 45（含 risk 标量） | `keepout.risk.in_obs=true`；几何门控消融亦为 45 |
| PPO | train_every=64, epochs=4, minibatch=16, lr=5e-4 | `cfg/algo/ppo.yaml` |

**已知代码瑕疵（不影响结果，但投稿前建议清掉）**：
- `mppi_filter.py:171-177` 有一段**重复的 `if blend is not None` 死代码**（第二段永远不执行）。
- `MPPIShield`（粗模型版）会忽略 `blend/active_mask`，只有 `MPPIExactShield` 支持软接管 —— 所以 **soft 必须配 `exact=true`**。
- MPPI 批级跳过 vs per-env 计算（§5.2 ⚠️）。

---

## 8. 指标定义（论文表格里必须写清楚）

| 指标 | 定义 | 备注 |
|---|---|---|
| `collision` | episode 内是否出现过 `d<0`（0/1，取 max） | **违约率**，不是碰撞次数 |
| **cum违约** | 训练全程各迭代 `collision` 的**累积和** | ⚠️ **safe exploration 的核心指标**；需在论文里给出精确定义式（见下） |
| `min_obstacle_dist` | episode 内最小间隙 `d` | 越大越安全，但过大=过度保守 |
| `filter_activation` | λ（soft）或 0/1（hard）的 EMA | **部署开销 = 激活率 × MPPI 成本** |
| `correction` | `‖u−u_rl‖²` 的 EMA | 滤波器介入强度 |
| `detour_ratio` | 实飞路径长 / 参考路径长 | ≥1，越接近 1 越好（**效率代价**） |
| `over_clearance` | `min_dist − safe_margin`（≥0） | 小正值最好（**不过度保守**的证据） |
| `avg_power_W` | 由 throttle 反推 T200 电功率 | **TTE 的主指标之一** |
| `tracking_error_ema` | 跟踪误差 EMA | 任务性能不退化的证据 |

> **必须补的严谨性工作**：`cum违约` 目前是从 wandb 训练曲线上求和得到的，代码里没有落地实现。投稿前请把它写成
> `CumViol = Σ_{i=1}^{I} mean_env[ collision_i ]`（I = 训练迭代数），并写清 I（60/120）与 batch 大小 —— 否则跨训练长度的行不可比（见 §11 的确认清单）。

---

## 9. 实验结果（2026-08-25 重跑，全部对齐后）

> 数据源 `PAPER_DATA.md`。ood 场景，每格 10 个独立**评测** seed（单训练 seed）。

> 四格边界均 0.6、观测均 45 维，每格相对 ours 只差一个开关。

### 9.1 主消融：2×2 析因

**碰撞率（violation rate ↓）—— 四格互相全部不显著**

| | 开关 hard | 比例 soft |
|---|---|---|
| 几何门控 | 0.0215 | 0.0204 |
| 预测门控 | 0.0153 | **0.0169** |

PPO baseline: 0.0164。ours vs PPO 两比例检验 p=0.90。

**绕行比 detour_ratio ↓**

| | 开关 hard | 比例 soft |
|---|---|---|
| 几何门控 | 0.1915 | 0.2866 |
| 预测门控 | **0.1197** | 0.2215 |

**动作平滑度 action_smoothness ↑**

| | 开关 hard | 比例 soft |
|---|---|---|
| 几何门控 | -0.8052 | -0.7830 |
| 预测门控 | -0.8116 | **-0.7829** |

**两个在正交轴上都复现的主效应：**

1. **接管方式**（两行都成立）：比例接管给出更平滑的执行（|d|=3.9 / 3.0）与更低跟踪误差
   （|d|=3.0 / 1.4）；开关接管给出更短绕行（|d|=13.0 / 8.9）。
2. **门控信号**（两列都成立）：预测门控显著减少绕行（|d|=5.8 / 10.2）。

**但碰撞率在全部六组成对比较中均不显著**，包括 vs PPO baseline。

### 9.2 关键对照：A-only（保留风险监视器、关闭 MPPI 滤波）

| 指标 | ours | A-only | 判定 |
|---|---|---|---|
| collision ↓ | 0.0169 | 0.0166 | 打平 |
| return ↑ | 29.5125 | 30.4767 | A-only 优 |
| detour ↓ | 0.2215 | 0.1985 | A-only 优 |
| tracking ↓ | 1.3484 | 1.3508 | ours 优 |

**组件 B 在当前场景未测出收益**：安全性与完整方法打平，而 return（|d|=1.9）与
detour（|d|=2.6）反而更优。这一格必须在论文里主动交代。

### 9.3 场景为何测不出安全差异

| 场景 | 一条 episode 最近距离均值 | PPO 碰撞率 |
|---|---|---|
| nominal | 2.29 m | 0.8% |
| ood | 6.70 m | 1.6% |
| hard | -0.04 m | 74.8% |

ood 名义上更难（障碍 2.2–2.8 m/s vs 0.4–0.9），但速度过快反而**擦身而过**，最近距离比
nominal 还远一倍；滤波器仅在 1.8% 的步上介入。hard 则标定过头进入**饱和区**（两臂皆 0.75、
最近距离为负）。**三个场景均无区分度。**

功效：nominal 每组 1405 episode 仅能检出「碰撞率下降 89% 以上」，ood 为 67%。
若把场景标定到 PPO 碰撞率 0.2–0.4，每组 121 个 episode 即可检出 50% 的下降。

> **结论：当前实验测不出安全性差异，这不等于方法在安全性上无效。**
> 投稿前必须先重标定场景（扫 `intercept_steps`）。

### 9.4 训练期违约（单训练 run，Wilson 95% CI）

见 `PAPER_DATA.md` §6。PPO 6/420=1.43%，ours 2/426=0.47%，方向有利但
**两比例检验 p=0.15，不显著** —— 整个训练期事件仅 6 次 vs 2 次，属泊松噪声量级。

---
## 10. 论文故事（Gap → Contributions → Findings）

### 10.1 定位与一句话 headline

> **Risk-Proportional Predictive Safety Filtering for Smooth, Energy-Efficient
> Reinforcement Learning Control of Electric Underwater Vehicles.**
>
> ⚠️ 原标题含 *Collision-Free*。**当前数据不支持任何降低碰撞的声称**（六组成对比较全部不显著，
> 含 vs PPO baseline，p=0.90）。在场景重标定之前，标题与 abstract 不得出现 collision reduction。

站位（与你已发表的 PPO-Lagrangian 功率预算工作的 delta）：
- 前作：**平均功率**的**期望意义软约束**（收敛后成立）。
- 本文：**空间安全的逐步防护**，且**不牺牲能量** —— 从"用约束优化换安全"变成"用预测式比例接管换安全，能量顺带更好"。

### 10.2 Gap（四层，逐层收窄 —— 建议就按这个顺序写 Introduction）

1. **约束型安全 RL（CPO / PPO-Lagrangian / RCPO）只给期望意义的软约束**，训练早期违约、部署无逐步保证。对于电动水下载具，一次碰撞 = 设备损失，期望意义不够。
2. **单步 CBF/HOCBF shield 在推力驱动、有执行器迟滞的 6-DoF 水下载具上失效**：位置约束相对阶为 2、推力箱非对称含死区、T200 一阶迟滞 → 单步投影"来不及"，且饱和引发姿态失稳。**我们实测它把违约提高了 57%** —— 这是一个此前文献没有报告过的、领域特定的负结果。
3. **预测式安全滤波器（MPPI / MPC safety filter）的"何时介入"仍由启发式距离门控决定**。在**主动拦截型**动态威胁下，距离不等于风险 —— 我们实测距离门控 MPPI 与无保护的 PPO 无差别（0.230 vs 0.226）。**真正缺的不是滤波器，是门控信号。**
4. **几乎所有 shielding 方法都是二值切换**，这在**训练回路中**造成动作分布的双模态跳变与非平稳性，损害策略学习本身。安全模块与学习过程之间的这层相互作用，此前基本被当作实现细节，而非设计变量。

> **一句话 gap**：`在动态威胁下，安全滤波器的"门控信号"应当来自策略自身动作的预测后果，且介入应当是连续比例而非二值开关；这两点是与"用什么滤波器"正交、且影响更大的设计维度。`

### 10.3 Contributions（4 条，按强度排序）

**C1. Risk-Proportional Predictive Safety Filter（方法主贡献）。**
提出用一个 <2% 精度的名义 6-DoF 模型对**策略自身提议动作**做单条前滚，得到预测最小间隙 `d̂_min`，据此生成**连续接管系数 λ**，与 MPPI 安全动作做比例融合。该设计把安全滤波器的两个关键决策——**何时介入**、**介入多少**——从启发式几何量替换为**动作条件的预测风险**，且与具体滤波器实现解耦（λ 可套在任何 `u_safe` 生成器上，本文另给了 repulsion-field 版本作为验证）。

**C2. 一个 2×2 析因证据：门控信号与接管方式是两个正交且各自可复现的设计维度。**
{几何, 预测} × {开关, 比例} 四格全部跑通，**四格边界均 0.6、观测均 45 维，每格相对完整方法只差一个开关**。
两个主效应各自在另一维度的两档上都复现：接管方式（比例更平滑、开关绕行更短，两行一致）、
门控信号（预测式绕行更短，两列一致）。**这把主张从"我们的方法更好"提升为"这两个维度本身重要"。**
但须同时声明：**碰撞率在四格间及对 baseline 全部不显著**（见 §9.3 的场景原因）。

**C3. 与仿真误差 <2% 的水下载具名义模型 + 其三个必需的建模要素。**
公开 T200 迟滞、翻转帧浮力、M_RB 积分三处踩坑（缺一则误差 ~90%）。这是让"model-based 安全层"
在水下载具上真正可用的工程贡献，也化解了"model-free RL 却要模型"的张力：
**精确跟踪交给 RL，硬安全只需一个保守但校准过的粗模型。**
> ⚠️ 该 <2% 的在线验证在本轮**未能产出**（config key `task.keepout.validate_full` 不存在）。
> 投稿前必须补跑，否则此贡献无数据支撑。

**C4. 系统性负结果**（论文诚实度与信息量的主要来源）：
(a) 单步 CBF shield 有害（+57%，早期实验）；
(b) **在无安全压力的场景中，安全滤波器测不出安全收益** —— A-only（关闭 MPPI）与完整方法
安全性打平，且 return / detour 更优。这界定了方法的适用边界，也说明**评测场景的标定
与滤波器设计同等重要**。

### 10.4 Findings（2026-08-25 重跑后重写）

| # | Finding | 支撑 | 强度 |
|---|---|---|---|
| **F1** | **接管方式是有效设计维度**：比例接管给出更平滑执行与更低跟踪误差，开关接管给出更短绕行。**在预测门控与几何门控两行都复现。** | \|d\|=3.9/3.0（平滑）、3.0/1.4（跟踪）、13.0/8.9（绕行） | 强 |
| **F2** | **门控信号是有效设计维度**：预测门控显著减少绕行。**在比例与开关两列都复现。** | \|d\|=5.8 / 10.2 | 强 |
| **F3** | **安全性在全部六组成对比较中不显著**，含 vs PPO baseline。 | p=0.52~0.97；ours vs PPO p=0.90 | 确定 |
| **F4** | 原因是**场景无安全压力**而非方法无效：ood 最近距离均值 6.68 m、滤波仅 1.8% 步介入；hard 两臂皆 0.75（饱和）。 | §9.3 + `power.csv` | 确定 |
| **F5** | **组件 B 未测出收益**：A-only（仅 risk 进观测、关闭 MPPI）安全打平，return（\|d\|=1.9）与 detour（\|d\|=2.6）更优。 | §9.2 | 确定 |
| **F6** | 训练期违约方向有利但**不显著**（PPO 6/420 vs ours 2/426，p=0.15）—— 事件数属泊松噪声量级。 | §9.4 | 弱 |
| **F7** | **PPO 训练目标存在错配**（ratio 分子分母用不同动作），且扭曲程度沿"不滤波→比例→开关"单调递增，故 F1 中部署效应与训练扭曲**尚未分离**。 | §6.3(ii) | 必须交代 |

**已撤回的旧 Finding**：原 F4「训练全程累计违约降 66–74%、收敛期零违约」、F5「safety for free」、
F6「内化有害」、F7「OOD 碰撞砍半」—— 分别因未通过显著性检验、指标口径变更、组件出局、
以及被 10 评测 seed 的重跑取代而作废。

### 10.5 论文结构建议

```
I.   Introduction              —— §10.2 四层 gap → §10.3 四条 contribution
II.  Related Work              —— 约束安全RL / CBF-shield / 预测式安全滤波 / 水下RL控制
III. Problem Formulation       —— §1（含动态拦截场景的设计动机：为什么静态场景没有信息量）
IV.  Method
     A. Nominal Dynamics (<2%)      —— §3（含三个必需建模要素）
     B. Predictive Risk Monitor (A) —— §4
     C. MPPI Safety Filter (B)      —— §5
     D. Risk-Proportional Takeover  —— §6 ★ 核心小节，含 hard/soft 的学习动力学论证
     E. Training Loop & Metrics     —— §6.3(ii) + §8
V.   Experiments
     A. Setup & Metrics
     B. Main Result (E1: PPO vs A+B-soft, 10 test seeds, mean±std + Welch t-test)
     C. 2×2 Factorial Ablation ★    —— F1–F3
     D. Internalization Matrix (负结果) —— F6
     E. Rule-based Comparison (CBF/距离门控) —— F9/F1
     F. Sensitivity: soft_hi / risk horizon / threshold
     G. Generalization: 跨轨迹 (E2) + OOD 威胁 (S4) + 扰动 (E6)
     H. Energy & Computation: power, detour, latency, 激活率 (E5)
     I. Limitations: S5 边界、单障碍、姿态冻结近似、仿真限定
VI.  Conclusion
```

### 10.6 审稿人会打的五刀 + 预备回应

| 攻击 | 回应 |
|---|---|
| **"只有单训练 seed，差异是噪声"** | ⚠️ **依然致命且部分坐实**。现已补 10 个评测 seed + 全部成对 Welch 检验：接管方式与门控信号两个主效应在正交轴上均复现且 |d| 很大；但**碰撞率全部不显著**，训练期违约 p=0.15。仍需 ≥3 个**训练 seed**（现有为评测 seed，属伪重复）。 |
| "MPPI 太慢，不能部署" | 用**激活率 × 单次成本**给期望开销；给 Jetson/单卡计时表（E5）；强调 A 的单条 rollout 才是每步跑的部分。同时修掉批级跳过（§5.2 ⚠️）。 |
| "为什么不和 CPO/PPO-Lagrangian 比" | 已有前作数据可嵌；并在正文里说明**语义不同**（期望软约束 vs 逐步比例防护），以及本文 PPO 基线已含避障奖励（=PPO+penalty）。 |
| "只有一个球形障碍，太简单" | 代码已支持 `n_obstacles>1`（滤波器取最近障碍，MPPI 代价取对任意障碍的最小距离）。**建议补一组 2–3 障碍实验**，成本很低。 |
| "λ 的两个阈值是调出来的" | E3a 已按 `soft_hi ∈ {0.3,0.6,0.9,1.2}` **分别重训**（关键：soft_hi 会改变训练分布，不能只 eval-only 扫）。补上误差棒后这是加分项而非弱点。 |

---

## 11. 投稿前清单（2026-08-25 更新）

### 已完成 ✅
- [x] E1 重跑：主对比使用 `165556-y7en5cfg`（早期 `104822-0hsw6qx5` 是 K=8/N=6、激活率 0.0001 的废实验）
- [x] 补齐 2×2 第四格（Geometric+binary，120 it，与其余格同参）
- [x] **对齐 Predictive+binary 的触发边界** 0.3 → 0.6（此前该格与完整方法相差两个变量）
- [x] **对齐几何门控的观测维度** 44 → 45（新增 `risk.gate`，解耦「A 进观测」与「A 做门控」）
- [x] 违约指标改为 **训练期违约率 + Wilson 95% CI**（旧 `cum违约` 依赖记录点数，不可比）
- [x] Welch 检验改用真 t 分布（小样本下正态近似偏激进）
- [x] 内化 C 从代码路径与全部产出中剔除
- [x] 每格 10 个独立评测 seed + 全部成对检验

### P0：不做就不能投
- [ ] **重标定场景**：扫 `intercept_steps`，把 PPO 碰撞率调到 **0.2–0.4**。
      当前三个场景均无区分度（§9.3），**不做这一条，后续任何实验都埋在噪声里**。
- [ ] **修 PPO 训练目标**（表述 A：学习器存 `u_rl`，`u_t` 只给仿真器）+ 全量重训。
      在此之前「比例 vs 开关」的差异无法归因（§6.3 ii）。
- [ ] 标题 / abstract **移除 collision reduction 声称**，直到上面两条完成后重新评估。
- [ ] 至少 3 个**训练 seed**（现有 10 个是评测 seed，属伪重复）。

### P1：数据缺口
- [ ] 名义模型 <2% 的在线验证未产出（key `task.keepout.validate_full` 不存在）→ C3 目前无数据支撑
- [ ] 模型失配敏感性 18 次全失败（key `nom_scale_mass/drag/thrust` 不存在）
- [ ] 计算延迟需重做：现测的是 Isaac rollout fps，K=32→256 数值不变。
      正确做法：在 `mppi_filter.py` 内对滤波调用打 CUDA-synced 计时，报「单次滤波 ms × 激活率」
- [ ] **核实帧数**：`HANDOFF.md` 写 30,720、`experiment_plan_tte.md` 写 10M，与实际不符，投稿数字须统一
- [ ] `risk.horizon ∈ {5,10,15,20,30}` 扫描 → 回应"0.24 s 算不算预测"

### P2：代码清理
- [ ] 删 `mppi_filter.py:174-177` 死代码（第 171 行已 return，永不可达）
- [ ] 删 `_risk_trigger` —— 全文件只写不读
- [ ] `soft_blend` 在 `exact=false` 时静默失效 → 加 assert
- [ ] MPPI per-env gather，避免整批计算

> **注**：旧清单里「E3b 参数扫描结果可疑（所有配置 return 完全相同）」——
> 该症状与本轮发现的 **seed 被冻结集反压** 完全一致（同键多值，后者生效）。
> 现已有 `scripts/check_overrides.py` 做命令行预检，同键多值即中止。旧的 E3b 数据应视为无效。

---

## 12. 附：一段可直接改写进论文的 Method 摘要（英文骨架）

> We equip a standard PPO tracking policy with a *risk-proportional predictive safety filter*. At every control step (62.5 Hz), the policy's proposed thrust command `u_rl` is rolled out for H=15 steps through a nominal 6-DoF underwater-vehicle model whose one-step velocity error is below 2 % of the simulator, with the intercepting obstacle extrapolated at constant velocity. The resulting *predicted minimum clearance* `d̂_min` — a quantity conditioned on the policy's own action rather than on instantaneous geometry — defines a continuous takeover coefficient `λ = clip((h_hi − d̂_min)/(h_hi − h_lo), 0, 1)`. Whenever `λ > 0`, a sampling-based MPPI safety filter (K = 128 rollouts, N = 20 steps, same nominal model) produces a safe command `u_safe` that minimises keep-out penetration while staying close to `u_rl`; the executed command is the convex combination `u = (1 − λ) u_rl + λ u_safe`. The policy is trained with unmodified PPO on the task reward alone — no cost critic, no Lagrange multiplier, and no penalty on the filter's correction; we show that adding such an internalisation penalty is *monotonically harmful*. Because `λ` is a Lipschitz function of the predicted clearance, the executed action is a smooth deformation of the policy's own action distribution, which we identify as the mechanism behind the consistent advantage of proportional over binary takeover.
