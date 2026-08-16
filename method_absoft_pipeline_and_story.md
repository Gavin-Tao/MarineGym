# A+B-soft（Risk-Proportional Predictive Safety Filter）方法文档与论文故事

> 面向 IEEE TTE 投稿。本文档 = **Pipeline + Methods 的可写作版本** + **故事线（gap / contribution / findings）** + **审稿人攻击面与待补实验**。
> 代码基准：`marinegym/envs/single/track.py`、`marinegym/utils/risk_monitor.py`、`marinegym/utils/mppi_filter.py`、`marinegym/utils/nominal_dynamics.py`、`cfg/task/Track.yaml`。
> 数据来源：`results_summary.md`（E0 消融）、`results_tte.md`（T1–T5）。

---

## 0. 一句话方法

> 策略每一步提议动作 `u_rl`；用一个**与仿真误差 <2% 的名义 6-DoF 模型**把「策略自己这个动作」前滚 H 步，得到**预测最小间隙** `d̂_min`；由 `d̂_min` 生成一个**连续接管系数 λ∈[0,1]**；同一模型的 MPPI 采样出安全动作 `u_safe`；实际执行 **`u = (1−λ)·u_rl + λ·u_safe`**。

三句话说清与已有工作的差别：

1. **门控信号是"策略自己的未来"，不是"当前距离"** —— 这是 A（Risk Monitor）。
2. **接管是比例的，不是开关** —— 这是 soft blend（λ）。
3. **不做内化惩罚（无 C）** —— 实测越加越差，写成负结果。

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
    trigger = 1[ d̂_min < δ ],                   δ = threshold = 0.3 m
```

### 4.2 三点设计解读（写 methods 时的"为什么"）

1. **它评估的是"策略自己这个动作的后果"，不是环境的静态几何。**
   同样的位置、同样的距离，如果 `u_rl` 朝障碍加速 → 高风险；如果 `u_rl` 已在制动 → 低风险。
   这是与所有 distance-gated / TTC-gated 滤波器的本质区别，也是 **B-only(0.109/0.230) vs A+B(0.062/0.106)** 差距的机理解释。

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

**(ii) 学习层面（关键，且是本文独有的洞见）：**
训练回路里，**写回 tensordict 的是执行动作** `u_t`（`track.py:490`），而 torchrl 的 collector 存的正是这个被就地修改过的 tensordict（`_shuttle` 同一对象），于是 PPO 的 `_update` 里
`log_probs = dist.log_prob(td[("agents","action")])` 用的是**执行动作 `u_t`**，而 `sample_log_prob` 是**提议动作 `u_rl`** 的对数概率。
后果：
- 策略被朝**实际执行的安全动作**推（一种隐式的 filtered-action relabeling / 安全动作模仿）；
- 重要性比 `ratio = π_θ(u_t|s)/π_old(u_rl|s)` 在第一个 epoch **不等于 1**，其偏离量正比于 `‖u_t − u_rl‖`。

**soft 与 hard 在这里的差别是决定性的**：soft 时 `u_t − u_rl = λ(u_safe − u_rl)`，λ 小则扰动小、ratio 接近 1、梯度低偏差；hard 时一旦触发就是**全量跳变**，产生大 ratio 偏离 → 被 clip 截断或产生高方差/有偏更新。这正好解释了为什么**在内化维度的每一档上，soft 都 ≥ hard**（0.062<0.106、0.118<0.276、0.309≈0.289）。

> ⚠️ 这条既是亮点也是风险点：它是一个**实现细节驱动的机理**。写论文有两种处理：
> **(a) 诚实且主动地把它写成方法的一部分**（"executed-action relabeling"），并加一个消融：把 buffer 里的 action 换回 `u_rl` 再训一版，对比之。这会让机理故事非常硬。
> **(b) 改成标准做法**（存 `u_rl`）再重训。
> **强烈建议走 (a)** —— 它把"soft 为什么赢"从"更平滑"这种软解释升级成一个可验证的学习动力学命题，而且实验成本只有 1–2 次训练。

**(iii) 数据分布层面：** λ 连续意味着策略经历的动作分布是自身分布的**平滑形变**而非双模态混合，on-policy 假设破坏更小。

### 6.4 关于 C（内化惩罚）为什么有害 —— 同一机理的自然推论

C 的做法是 `r ← r − w_C·‖u_safe − u_rl‖²`（`track.py:632-642`，含固定权重与自适应 EMA 两版）。
既然 soft blend 已经通过 **executed-action relabeling** 提供了"向安全动作靠拢"的隐式压力，C 就是在**同一目标上叠加第二个、且是负向的、量纲不匹配的压力**：它惩罚的是「被修正」这件事本身，等价于鼓励策略**远离障碍到根本不触发滤波器**，从而与跟踪任务直接冲突 → 回报下降、行为保守化、反而在动态拦截下更容易被逼进危险区。
实测梯度完全一致：**无C < 固定C < 自适应C**（越复杂越糟）。这在论文里是一个漂亮的**负结果 + 机理解释**，而不只是"我们试了没用"。

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
| A: hard threshold δ | 0.3 m | `keepout.risk.threshold` |
| B: K / N | 128 / 20 (0.32 s) | `keepout.mppi.num_samples/horizon` |
| B: σ / λ_temp | 0.4 / 0.05 | `noise_sigma / temperature` |
| B: w_coll / w_track | 5.0 / 1.0 | `keepout.mppi` |
| **soft_lo / soft_hi** | **0.0 / 0.6 m** | `keepout.mppi.soft_lo/soft_hi` |
| C 权重 | 0（本文方法不开） | `keepout.internalize_weight` |
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

## 9. 实验结果的骨架（重排为论文可用的结构）

### 9.1 主消融：2×2 析因（S2 动态拦截，cum违约↓）

| | **开关（hard）** | **比例（soft）** |
|---|---|---|
| **距离门控** | 0.230（≈PPO 0.226） | 0.109 |
| **预测门控（A）** | 0.106 | **0.062 ★** |

PPO 基线：0.181 / 0.236（120it 双 seed）、0.226（60it）。

**读法（这是论文 findings 的核心）**：
- **主效应 1（门控信号）**：在两种接管方式下，预测门控都优于距离门控（0.230→0.106，0.109→0.062）。
- **主效应 2（接管方式）**：在两种门控下，比例都优于开关（0.230→0.109，0.106→0.062）。
- **两者可叠加**，无交互反号。
- **单独用 B（距离+开关）几乎无用**（0.230 vs PPO 0.226）→ 直接反驳"随便挂个 MPPI 就行"。

> ⚠️ **confound 必修**：距离+开关那行是 **60 iters**，其余是 120 iters。虽然 PPO 在两个长度下 cum 接近（0.226 vs 0.181/0.236），但审稿人会抓。**补跑一次 120it 的 B-hard**，这一格补齐后 2×2 就无懈可击。

### 9.2 内化维度：3×3 全矩阵（负结果）

| 门控\内化 | 无 C | 固定 C (0.5) | 自适应 C |
|---|---|---|---|
| 硬触发 | 0.106 | 0.276 / 0.343 | 0.289 |
| **软接管** | **0.062 ★** | 0.118 | 0.309 |

**两条结论**：(1) soft ≥ hard **在每一档内化条件下都成立** → adaptive 门控系统性成立，不是挑出来的一个点；(2) C 单调有害：无C < 固定C < 自适应C。

### 9.3 反例组（论证"必须预测式"）

| 方法 | cum违约 | vs PPO |
|---|---|---|
| 单步 CBF/HOCBF shield（S1 静态） | 3.30 | **+57%（有害）** |
| B-only 距离门控 | 0.230 | +2%（无效） |
| A+B-soft | 0.062 | −66~74% |

单步 shield 有害的机理（已在 `safe_rl_auv_design.md §10.2` 记录）：**推力对位置的相对阶为 2 + 执行器迟滞 + 推力饱和** → 单步来不及刹车，且饱和产生的力矩把载具翻滚。这是一个**领域特定的、有说服力的失败案例**，一定要写进论文（水下 6-DoF 载具 ≠ 二维质点，CBF 的教科书结论不能直接搬）。

### 9.4 跨场景一致性

| 场景 | 结论 |
|---|---|
| S1 静态障碍（30it） | A+B −55%（但 60it 后 PPO 自己也归零 → 只有过程差异） |
| S2 动态拦截 60it / 120it | A+B-hard **两次独立训练都是 0.106** → 结果稳 |
| S3 动态 HARD (v=1.2–1.8) | A-only −73%（单 seed 波动；**缺 A+B 无C 行，需补**） |
| S4 OOD 部署（未见 2.2–2.8 m/s） | 同一策略挂 B：collision 0.042 → **0.021（砍半）** |
| S5 随机策略 + A+B | **不成立**（0.31 vs 0.19）→ 方法边界 |

### 9.5 代价侧（TTE 最关心的一栏）

| | PPO | A+B-soft |
|---|---|---|
| return | 32.92 | 32.74（−0.5%） |
| tracking_error_ema | 1.359 | 1.352（略优） |
| min clearance | 1.869 | **2.358** |
| **avg power (W)** | 1189.96 / 1288.33 | **1189.05**（E0）；TTE T2: **1168 vs 1322（−12%）** |
| filter 激活率 | — | 0.10 |

**这是本文投 TTE 的关键一栏**：安全提升 66–74%，而**回报、跟踪、功率均不退化甚至更优**。可以直接立一个 claim：**"safety for free"（在能量与任务性能意义上）**。
> 但要小心：功率更低有可能只是"绕得更远、加速更少"的副作用，且 `detour_ratio` 会同时变化。**建议把 power 和 detour_ratio 一起报**，并在正文里给出解释（软接管避免了开关抖振 → 推力反复换向减少 → 电功率下降）。这个解释如果能用 `action_smoothness` / throttle 换向次数佐证，就非常漂亮。

---

## 10. 论文故事（Gap → Contributions → Findings）

### 10.1 定位与一句话 headline

> **Risk-Proportional Predictive Safety Filtering for Energy-Efficient, Collision-Free Reinforcement Learning Control of Electric Underwater Vehicles.**

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

**C2. 一个 2×2 析因证据：门控信号与接管方式是两个独立且可叠加的设计维度。**
{距离, 预测} × {开关, 比例} 四格全跑通，两个主效应方向一致、无反号交互；并进一步在 3×3（× 内化）矩阵上验证 soft ≥ hard 在每一档都成立。**这把"我们的方法更好"升级成"这两个设计维度本身重要"** —— 后者对社区的价值高得多，也更难被"换个 baseline 就翻盘"。

**C3. 与仿真误差 <2% 的水下载具名义模型 + 其三个必需的建模要素。**
公开 T200 迟滞、翻转帧浮力、M_RB 积分三处踩坑（缺一则误差 ~90%）。这是让"model-based 安全层"在水下载具上真正可用的工程贡献，也化解了"model-free RL 却要模型"的张力：**精确跟踪交给 RL，硬安全只需一个保守但校准过的粗模型。**

**C4. 系统性负结果三则**（论文诚实度与信息量的主要来源）：
(a) 单步 CBF shield 有害（+57%）；(b) 距离门控 MPPI 无效（+2%）；(c) 安全内化惩罚（C，固定/自适应）单调有害。三者共同界定了**方法的必要条件**：必须预测、必须软、不要内化。

### 10.4 Findings（结果部分逐条回答的问题）

| # | Finding | 支撑 |
|---|---|---|
| **F1** | 在主动拦截型动态威胁下，**距离不是风险** —— 距离门控滤波器等于没挂。 | 0.230 vs PPO 0.226 |
| **F2** | 把门控信号换成**动作条件的预测间隙**，违约减半。 | 0.230→0.106（hard）、0.109→0.062（soft） |
| **F3** | 把接管从**开关换成比例**，在任何门控/内化条件下都更好。 | 2×2 与 3×3 矩阵全一致 |
| **F4** | 完整方法（A+B-soft）**训练全程累计违约降 66–74%，收敛期零违约**。 | 0.062 vs 0.181–0.236 |
| **F5** | 安全提升**不以任务性能或能量为代价**：return −0.5%，跟踪略优，**功率最低（TTE 版 −12%）**。 | §9.5 |
| **F6** | **安全内化（C）有害且单调**：无C < 固定C < 自适应C。 | 3×3 矩阵 |
| **F7** | 滤波器可**即插即用地迁移到 OOD 威胁**（未见的 2.2–2.8 m/s 拦截），无需重训，碰撞砍半。 | 0.042→0.021 |
| **F8** | **方法边界（诚实报告）**：滤波器辅助的是**称职策略**，救不了随机/劣质策略。 | S5 失败 |
| **F9** | 教科书式单步 CBF 在推力驱动 6-DoF 水下载具上**主动有害**。 | +57% |

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
| **"只有单 seed，0.062 vs 0.106 是噪声"** | ⚠️ **目前最致命**。必须补 E1（10 test seeds + Welch t-test），并至少给 A+B-soft / A+B-hard / PPO 各 3 个 **training seed**。3×3 矩阵内方向一致是有力佐证，但不能替代统计检验。 |
| "MPPI 太慢，不能部署" | 用**激活率 × 单次成本**给期望开销；给 Jetson/单卡计时表（E5）；强调 A 的单条 rollout 才是每步跑的部分。同时修掉批级跳过（§5.2 ⚠️）。 |
| "为什么不和 CPO/PPO-Lagrangian 比" | 已有前作数据可嵌；并在正文里说明**语义不同**（期望软约束 vs 逐步比例防护），以及本文 PPO 基线已含避障奖励（=PPO+penalty）。 |
| "只有一个球形障碍，太简单" | 代码已支持 `n_obstacles>1`（滤波器取最近障碍，MPPI 代价取对任意障碍的最小距离）。**建议补一组 2–3 障碍实验**，成本很低。 |
| "λ 的两个阈值是调出来的" | E3a 已按 `soft_hi ∈ {0.3,0.6,0.9,1.2}` **分别重训**（关键：soft_hi 会改变训练分布，不能只 eval-only 扫）。补上误差棒后这是加分项而非弱点。 |

---

## 11. 投稿前必须核实/补齐的清单

**数据完整性（P0）**
- [ ] **E1 重跑**（HANDOFF 已记录：T2 checkpoint 曾用错，`104822-0hsw6qx5` 是 `K=8,N=6` 的早期实验，激活率 0.0001 ≈ 没开滤波）→ 必须用 `165556-y7en5cfg`。
- [ ] 补 **B-hard @120it**，补齐 2×2 的训练长度 confound（§9.1）。
- [ ] 补 **S3-HARD 的 A+B 无C 行**。
- [ ] 至少 3 个 **training seed**（不只是 eval seed）跑主对比。

**定义与可复现（P0）**
- [ ] 把 `cum违约` 落地成代码/脚本并写出定义式；确认所有行的 I（迭代数）与 batch 一致。
- [ ] **核实帧数**：`frames_per_batch = num_envs(16) × train_every(64) = 1024`，120 iters → **122,880 帧**；但 `HANDOFF.md`/`results_tte.md` 写的是 30,720 帧、`experiment_plan_tte.md` 写 10M 帧。三处不一致，投稿数字必须统一。
- [ ] E3b 参数扫描结果可疑（所有配置 return 完全相同）→ 确认参数是否真的生效。

**机理证据（P1，做了会显著加分）**
- [ ] **executed-action relabeling 消融**：buffer 存 `u_rl` vs 存 `u_t`，各训一版 → 直接验证 §6.3(ii) 的学习动力学解释。
- [ ] 记录并作图：`‖u_t − u_rl‖` 分布、首个 epoch 的 importance ratio 分布（soft vs hard）→ 把 "soft 更好" 的机理可视化。
- [ ] `risk.horizon ∈ {5,10,15,20,30}` 扫描 → 回应"0.24 s 的预测算不算预测"。
- [ ] power 下降与 throttle 换向次数/smoothness 的相关性 → 支撑 §9.5 的能量解释。

**代码清理（P2）**
- [ ] 删 `mppi_filter.py:174-177` 死代码。
- [ ] MPPI per-env gather，避免整批计算。
- [ ] `soft_blend` 在 `exact=false` 时静默失效 → 加一个 assert/warning。

---

## 12. 附：一段可直接改写进论文的 Method 摘要（英文骨架）

> We equip a standard PPO tracking policy with a *risk-proportional predictive safety filter*. At every control step (62.5 Hz), the policy's proposed thrust command `u_rl` is rolled out for H=15 steps through a nominal 6-DoF underwater-vehicle model whose one-step velocity error is below 2 % of the simulator, with the intercepting obstacle extrapolated at constant velocity. The resulting *predicted minimum clearance* `d̂_min` — a quantity conditioned on the policy's own action rather than on instantaneous geometry — defines a continuous takeover coefficient `λ = clip((h_hi − d̂_min)/(h_hi − h_lo), 0, 1)`. Whenever `λ > 0`, a sampling-based MPPI safety filter (K = 128 rollouts, N = 20 steps, same nominal model) produces a safe command `u_safe` that minimises keep-out penetration while staying close to `u_rl`; the executed command is the convex combination `u = (1 − λ) u_rl + λ u_safe`. The policy is trained with unmodified PPO on the task reward alone — no cost critic, no Lagrange multiplier, and no penalty on the filter's correction; we show that adding such an internalisation penalty is *monotonically harmful*. Because `λ` is a Lipschitz function of the predicted clearance, the executed action is a smooth deformation of the policy's own action distribution, which we identify as the mechanism behind the consistent advantage of proportional over binary takeover.
