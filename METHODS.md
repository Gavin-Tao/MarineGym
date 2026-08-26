# 论文③ 方法部分规格（写作用）

本文件把 Methods 需要的**确切数值与实现细节**集中在一处，全部与代码一致，
可直接引用。代码位置写在括号里。

---

## 1. 任务与仿真

**平台**：MarineGym（Isaac Sim 4.10），载具 **iAUV**（`marinegym/robots/drone/iAUV.py`，
继承 `UnderwaterVehicleFin`：推进器 + 舵面）。

**任务**：`Track` —— lemniscate（8 字）轨迹跟踪（`marinegym/envs/single/track.py`）。

| 项 | 值 | 位置 |
|---|---|---|
| 控制频率 | 62.5 Hz（`dt = 0.016 s`, `substeps = 1`） | `cfg/base/sim_base.yaml` |
| episode 上限 | 600 步（9.6 s） | `cfg/task/Track.yaml` |
| 终止条件 | 跟踪误差 > `reset_thres = 0.5 m` 即提前终止 | `track.py: terminated \|= distance > reset_thres` |
| 并行环境数 | **4096** | `cfg/base/env_base.yaml`（MarineGym 基准原始值） |
| 参考轨迹尺度 | `traj_scale ~ U([1.4,1.4,0.8], [2.6,2.6,1.2])` m | `track.py:154` |
| 参考角速度 | `traj_w ~ U(0.7, 0.9)` rad/s → **实测参考点速度 0.543 m/s**（⚠ 该配置下任务能力受限，见 §2.1） | `Track.yaml: traj_w_range` |
| 动作空间 | 连续，`BoundedTensorSpec(-1, 1, 3)`（推进器 + 2 路舵面） | `underwaterVehicleFin.py:53` |

**观测**（37 维，`track.py::_compute_state_and_obs`）：

```
rpos(3×4=12)  未来 4 个参考点相对位置（间隔 5 步）
quat(4) | vel(6, 线+角) | heading(3) | up(3) | throttle(3) | time_encoding(4)
```

> `drone_state = [pos(3), rot(4), vel(6), heading(3), up(3), throttle(n_act)]`
> （`underwaterVehicleFin.py::get_state`），观测里用 `rpos` 取代 `pos`。

---

## 2. 部分可观测条件

主条件 **`drift`**：未观测的执行器增益漂移
（`marinegym/envs/utils/pomdp.py::ActuatorDrift`）。

```
u_applied = gain ⊙ u_policy,  gain ~ U(0.5, 1.0)  逐通道、逐 episode 重采样
gain 不进入观测
```

物理依据：生物附着、电机磨损、电压下降、推力标定漂移。

**对照条件**（用于把"信息缺失"与"任务变难"分开）：

| 条件 | 说明 |
|---|---|
| `drift_oracle` | 相同的 gain 采样与施加，但 gain 归一化后拼进观测（40 维） |
| `drift_accel` | 相同任务，额外把 `(v_t − v_{t−1})/dt`（6 维）拼进观测（43 维） |

> ⚠ **已得到的负结果**：`drift_oracle` 相对 `drift` 对任何 arm 都没有改善
> （MLP 76.12 → 76.43，Mamba 81.85 → 80.96，单位为 episode 长度）。
> ⇒ 任务的困难**不在**增益辨识。当前假说是"优势来自恢复加速度（附加质量项）"，
> 由 `drift_accel` 检验。**这条结论要如实写进论文**（负结果同样有信息量）。

---

## 3. 观测历史窗口

历史在**环境层**维护（`pomdp.py::ObsHistory`），观测形状为 `[n_agents, L, obs_dim]`。
策略因此是**无状态**的，PPO 的 minibatch 打乱采样无需改动。

| 参数 | 值 | 理由 |
|---|---|---|
| 窗口长度 `L` | 16 | |
| 窗口步长 `stride` | **4** | 见下 |
| 窗口跨度 | (16−1)×4+1 = **61 步 ≈ 0.98 s** | 与 episode 长度同量级 |
| reset 行为 | 用重置后第一帧填满整个缓冲（非补零） | 避免分布偏移与跨 episode 污染 |

**为什么必须做帧跳（stride > 1）** —— 这是一条非平凡的实现要求，建议写进论文：

62.5 Hz 下位置量级 ~2 m，而每步位移仅约 2.6 cm（实测参考点速度 0.543 m/s），
**相邻帧相对差异仅约 1%**。用相隔 k 帧的两点估计速度，噪声为 `√2·σ/k`，
即**信噪比正比于帧间隔**。实测（`scripts/p3_probe.py`，探针 B）：

| arm | stride=1 | stride=2 | stride=4 | stride=8 | stride=16 |
|---|---:|---:|---:|---:|---:|
| Frame-stack | 0.058 | 0.036 | 0.027 | 0.024 | 0.023 |
| GRU | 0.107 | 0.034 | 0.021 | 0.013 | 0.007 |
| Transformer | 0.029 | 0.023 | 0.015 | 0.009 | 0.005 |
| **Mamba** | **0.288** | 0.020 | 0.012 | 0.011 | 0.011 |

**Mamba 在 stride=1 时比 stride=2 差 24 倍** —— 因果 conv1d 与选择性扫描正是在
相邻帧之间做差分/门控，"相邻帧近乎重复"是它的最坏情形。
stride 是所有序列臂**共享**的超参，取值相同，故公平。

---

## 4. 编码器（对照臂）

统一契约：输入 `[..., L, D]`，输出 `[..., 256]`，接同一个策略头/值头
（`marinegym/learning/modules/encoders.py`）。

| 臂 | 配置 | 参数量 | 相对基线 |
|---|---|---:|---:|
| **Transformer（基线）** | DTQN 原生默认：d128 × **2 层** × 8 heads × ff4 | 436,992 | 1.00× |
| **Mamba（ours）** | d128 × **1 层**，d_state=16，d_conv=4，expand=2 | 155,392 | **0.36×** |
| MLP（单帧下界） | `[256,256,256]`，只取窗口末位 | 142,848 | 0.33× |
| Frame-stack | flatten(L×D) → `[128,256,256]` | 176,774 | 0.40× |
| GRU | d128 × 1 层 + `LayerNorm(h+input)` | 137,728 | 0.32× |

**容量扫描（ours 变体）**：Mamba d96×1（97,536）、d64×1（~50k）
**反向对照**：`transformer_small` d96×1（143,142）、`mamba_big` d128×4（~503k）、
`mlp_wide` `[384,384,384]`（312,576）

### 4.1 Transformer —— 忠实复刻 `kevslinger/DTQN`

| 项 | DTQN 上游 | 本实现 |
|---|---|---|
| 位置编码 | `--pos learned`（默认），按 context_len 分配 | 同 |
| LayerNorm 位置 | `--identity` 默认 False ⇒ LN 在跳连**之后** | 同 |
| 门控 | `--gate res`（默认）；`gru` 为 GTrXL 变体（`w_z.bias = −2`） | 两者都实现，默认 res |
| 子层输出 | 先过 ReLU 再进门控 | 同 |
| FFN | `Linear(d,4d) → ReLU → Linear(4d,d) → Dropout` | 同 |
| causal mask | `torch.triu(ones(L,L), diagonal=1)` 上三角置 `−inf` | 同（**已缓存**） |
| dropout | 0.0 | 同 |
| 规模 | `--in-embed 128 --heads 8 --layers 2` | 同 |

**有意偏离（一处，需在论文中声明）**：DTQN 在窗口每个时刻都预测 Q 值并一起训练
（`--history`，intermediate Q-values）。那是 Q-learning 专有的样本效率技巧；
本文是 on-policy actor-critic（PPO），损失只定义在当前动作上，故只取末位。
该差异对所有序列臂一视同仁。

### 4.2 Mamba —— 使用官方 `state-spaces/mamba` 组件

不自行实现 SSM 或残差接法：

- `from mamba_ssm import Mamba`（README 中的官方模块）
- `create_block(d_model, d_intermediate=0, ssm_cfg={d_state, d_conv, expand}, rms_norm=False)`
  —— `d_intermediate=0` 即纯 Mamba block，与官方 Mamba-1 一致
- 前向的 `residual` / `norm_f` 处理**逐行照抄** `MixerModel.forward` 的
  `fused_add_norm=False` 分支

版本：`mamba_ssm 2.2.4` + `causal_conv1d 1.5.0.post8`（预编译 CUDA wheel，
`selective_scan_cuda` 与 `causal_conv1d_cuda` 均启用）。

### 4.3 初始化保护（实现要点）

PPO 原本对**所有** `nn.Linear` 施加 `orthogonal_(gain=0.01)` + `bias.fill_(0)`。
这会覆盖 Mamba 的 `dt_proj.bias` / `A_log` / `D`、GRU 的 `uniform(±1/√H)`、
GRUGate 的 `w_z.bias = −2` 等**上游精心设计的初始化**（且 Mamba 的投影层
`bias=False` 会直接抛异常）。

⇒ 各编码器暴露 `protected_modules()`，PPO 只对策略头/值头/输入投影施加初始化
（`marinegym/learning/ppo/ppo.py`）。已单元测试验证上游初始化被保留。

---

## 5. 训练

| 项 | 值 |
|---|---|
| 算法 | PPO（`marinegym/learning/ppo/ppo.py`，继承自 OmniDrones 实现） |
| 并行环境数 | **4096**（基准原始协议） |
| 迭代数 | **76** |
| `train_every` | 64 ⇒ `frames_per_batch = 4096 × 64 = 262,144` |
| 总环境步 | **≈ 20M**（基准原始 `total_frames`） |
| 学习率 | 5e-4（actor/critic 各自 Adam） |
| PPO epochs / minibatches | 4 / **64**（4096 envs 下保持单 minibatch ≈4096 样本） |
| clip / entropy coef | 0.1 / 0.001 |
| GAE | γ=0.99, λ=0.95 |
| 梯度裁剪 | 5.0 |
| 训练 seed | **1 个** |

> ⚠ **不要用 64 envs**。论文①② 的脚本用 64 envs（492k 帧），那对比较**安全滤波器**
> 够用，但对比较**策略架构**不够：所有臂都停在欠训练区（entropy 仅从 4.25 降到 3.8–4.1），
> 差异反映的是"谁学得快"而非"谁能力强"。实测 64 envs 下序列模型比 MLP 高 7.5%，
> 而 4096 envs 下差异完全消失。
>
> 4096 envs 需要两处配合：PPO 的 critic 全批前向要**分块**（整批 262k 样本会 OOM，
> 分块与整批数学等价），以及**不要**削减 PhysX GPU 缓冲（那是按 64 envs 调的）。

---

## 6. 评测协议

**单次训练 × 10 个独立评测 seed**（与论文①② 口径一致，见 `PAPER_DATA.md` §4）。

- 确定性策略（`ExplorationType.MODE`）
- **共同随机数**：所有 arm 使用同一组评测 seed（12345…12354）⇒
  面对完全相同的初始条件与轨迹参数（`traj_c` / `traj_scale` / `traj_w`）序列。
  不这样做的话，轨迹参数带来的方差会淹没 arm 之间几个百分点的差异。
- 每个 seed 聚合约 2000 条完整 episode，总计约 20000 条
- 误差棒 = **跨 10 个评测 seed 的 std**
- 显著性 = n=10 的 Welch t 检验

> ⚠ **必须写进论文的限制**：该口径刻画**评测随机性**，
> **不含训练 run 间方差**（RL 中后者通常更大）。
> 因此"显著"只能说"这两个训练出来的策略表现不同"，
> **不能**推断"该方法平均更好"。若需方法层面的结论，须补多个训练 seed。

---

## 7. 指标

**主指标**（按重要性；顺序已于 2026-08-26 更正，理由见下）：

| 指标 | 定义 | 方向 |
|---|---|---|
| `episode_len` | 失控前的存活步数 | ↑ |
| `return` | 逐步累加的回报（`track.py:826`） | ↑ |
| `tracking_err_mean_m` | `−tracking_error / episode_len`，平均跟踪误差（米） | ↓ |

> ⚠ **`tracking_err_mean_m` 在提前终止机制下不是有区分度的指标。**
> episode 在误差 > `reset_thres`(0.5 m) 时终止，而误差从 0 近似**线性**增长到该阈值，
> 因此整段的均值必然 ≈ 0.19 m —— 与增长速率、控制质量、观测信息**都无关**。
> 实测三个参考速度档（ep_len 57.9 / 128.7 / 182.1，相差 3.1 倍）下，
> `tracking_err_mean_m` 分别为 0.1929 / 0.1939 / 0.1943，几乎不动。
> 论文中应以 `episode_len`（等价于"失控前跟踪时长"）为主指标，
> `tracking_err_mean_m` 仅作辅助报告，并注明其被终止规则钉死这一性质。

**成本侧指标**：参数量（actor 编码器 + 头）↓、每步推理延迟（见 §8）↓。

**辅助指标**：`tracking_err_p90_m`（尾部）、`success_rate`（跑满 600 步未跟丢的比例）、
`episode_len` 的 p10/p50、`avg_power_W`（电功率，AUV 上关键）、`action_smoothness`。

> **口径警告（继承自论文①）**：episode 因**跟踪失败提前终止**（平均 ~76/600 步），
> 所以 `episode_len` 本身就是性能指标，而 `return` 已把长度计入。
> **不可按 episode 长度归一化 return** —— 那会奖励"失败得早"。

---

## 8. 推理代价测量

`scripts/p3_efficiency.py`，NVIDIA L40，150 次平均，两种部署模式：

- **window**：每个控制步重新前向整个长度 L 的窗口（Transformer 的唯一选项）
- **stream**：携带固定大小的递归状态，每步只前向 1 帧（仅 Mamba/GRU 可行）

已验证 Mamba 的流式递推与整窗并行扫描**数值一致**（最大误差 1.19e-07）。

> **诚实说明（必须写）**：batch=1、d_model≈128 的小模型下，各臂都是
> **kernel 启动开销受限**，Transformer 的 O(L²) 在 L ≤ 256 内看不出来。
> 必须同时报 batch=256（计算受限区）才能看到二次项。
> 另外 Mamba 的 **window 模式在小 L 时反而比 Transformer 慢**
> （选择性扫描 kernel 有固定开销），优势只存在于 **stream 模式**。

---

## 9. 复现命令

```bash
# 单格
bash scripts/run_mamba.sh train.py task=Track headless=true \
  task.env.num_envs=64 max_iters=120 \
  task.context_len=16 task.context_stride=4 \
  algo.encoder.name=mamba algo.encoder.d_model=128 algo.encoder.n_layers=1 \
  task.pomdp.thrust_gain_range=[0.5,1.0] \
  +p3_out=out.json +p3_eval_seeds=10 +p3_eval_episodes=2000

# 全矩阵
bash scripts/p3_chain.sh

# 出图表
bash scripts/p3_paper_bundle.sh
```
