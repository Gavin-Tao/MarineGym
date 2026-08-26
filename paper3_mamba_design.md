# 论文③ 设计:序列模型驱动的 AUV 部分可观测轨迹跟踪 (Mamba)

分支 `mamba`(从 `backup` @ 81dc735 建),worktree `/home/jovyan/MarineGym-mamba`。
不碰 `main`(论文①)与 `flow-safety`(论文②)。

---

## 0. 一句话主张

**AUV 在真实传感条件下是 POMDP,单帧观测的策略必然失效;历史序列编码是必需的,而 Mamba 以线性/常数代价达到 Transformer 的精度,因而是唯一可机载部署的选择。**

两个一级贡献,缺一不可:
- **C1 必要性**:历史信息在 POMDP 下带来大幅增益,在 MDP 下不带来增益(→ 增益来自信息,不是容量)。
- **C2 效率**:Mamba ≈ Transformer 精度,但每步推理 O(1)、显存常数(→ 可部署)。

---

## 1. 为什么现在的 Track 任务不能直接用

`marinegym/robots/drone/underwaterVehicleFin.py:get_state()` 输出:

```
pos(3) + quat(4) + vel(6: 线速度+角速度) + heading(3) + up(3) + throttle(num_rotors+num_fins)
```

`marinegym/envs/single/track.py:_set_specs()` 再拼 `3*(future_traj_steps-1)=9` 维未来参考点
(+ time encoding + 可选 intrinsics)。

**位置、姿态、线角速度、未来参考全给了 → 完全可观测的 MDP。**
MDP 下马尔可夫策略即最优,历史不含额外信息。在这上面比 MLP / GRU / Transformer / Mamba
只会得到"打平或更差",证明不了任何东西。

**所以第一步不是接算法,是造部分可观测性。**

---

## 2. POMDP 机制(新增 `cfg/task/Track.yaml: pomdp:` 段,默认全关)

全部在 `track.py:_compute_state_and_obs()` 出口处施加,不动物理仿真:

| 开关 | 语义 | AUV 物理依据 | 为什么必须要历史 |
|---|---|---|---|
| `drop_linear_vel` | 观测中去掉线速度(3维) | DVL 离底超程 / 气泡 / 故障 | 速度只能从位置历史差分推断 |
| `drop_angular_vel` | 去掉角速度(3维) | 陀螺失效 / 低成本 IMU | 同上 |
| `obs_delay: k` | 观测延迟 k 步 | 声学链路 + 滤波延迟 | 需要用历史外推当前真实状态 |
| `sparse_pos_period: k` | 位置每 k 步才更新,中间保持 | USBL 低频稀疏定位 | 需要在两次定位间航位推算 |
| `obs_noise_std` | 加性高斯噪声 | 传感器噪声 | 需要时序滤波 |
| `obs_dropout: p` | 以概率 p 丢帧(沿用上一帧) | 声呐丢包 | 需要缺失鲁棒 |
| `hydro_randomize` | 每 episode 随机化质量/阻尼/推力系数,**且不进观测** | 载荷变化、生物附着、标定漂移 | 需要在线系统辨识 |

**主实验用 `drop_linear_vel`**(最干净、物理最硬、审稿人最容易接受)。
`hydro_randomize` 作为第二个 POMDP 场景,支撑"需要长上下文"的论证(见 §5)。
`sparse_pos_period` 用于长上下文实验。

> ⚠️ 与论文② 的边界:洋流扰动是论文②(flow-safety)的主题,本文**不以未观测洋流为主 POMDP 源**,
> 只在讨论/附录里提一句可推广性,避免自我重复被拒。

---

## 3. 架构:上下文窗口,不是循环隐状态

### 决策与理由

`marinegym/learning/ppo/ppo.py:255 make_batch()` 做 `tensordict.reshape(-1)` + `randperm`,
**彻底打乱时序**。真·循环策略要求:按 env 采连续序列片段、BPTT、跨 minibatch 携带 hidden state ——
需要重写 PPO 训练循环,改动大且易出静默 bug。

改为**在环境层维护长度 L 的观测 ring buffer**,观测 spec 变成 `[L, obs_dim]`:
- 策略无状态 → PPO 训练循环**一行不用改**;
- DTQN 本身就是 context window(不是无限循环)→ 对标忠实;
- GRU 基线在窗口内跑 = DRQN-with-context,公平。

实现:`marinegym/envs/utils/obs_history.py`,一个 `[E, L, D]` 的 GPU ring buffer,
reset 时用当前帧填满(避免零填充引入的分布偏移),`is_initial` 由 `InitTracker` 提供。

### 编码器(新增 `marinegym/learning/modules/encoders.py`,可插拔)

输入统一 `[B, L, D]`,输出 `[B, H]`(取最后一步),接现有 `Actor`/critic 头:

| name | 说明 | 角色 |
|---|---|---|
| `mlp` | 只用最后一帧(L=1 退化) | **下界基线**,复现当前行为 |
| `stack` | flatten `[L*D]` → MLP | 朴素历史基线 |
| `gru` | 窗口内 GRU,取末态 | DRQN 类基线 |
| `transformer` | causal decoder + 位置编码,取末位 | **DTQN 对标** |
| `mamba` | 纯 PyTorch selective-scan SSM | **ours** |

**参数量对齐是必须的**:每个编码器提供 param-matched 配置(目标 ±5%),否则审稿人直接质疑
"赢的是容量不是架构"。在 `PPOConfig` 增 `encoder: {name, d_model, n_layers, context_len, ...}`,
`ppo.py:124` 处把 `make_mlp([256,256,256])` 换成 `build_encoder(...)`。

### Mamba 依赖:克隆独立 conda 环境

`/home/jovyan/envs/sim`(python3.10, torch 2.4.0+cu121, `CXX11_ABI=False`)被论文①②③共用,
且**没有 nvcc**(`CUDA_HOME=None`),不能源码编译 `selective_scan_cuda`。

→ **克隆出独立环境 `/home/jovyan/envs/sim-mamba`**,只在克隆环境里装 mamba,
论文①②的复现环境一个字节都不动。

```bash
conda create --clone /home/jovyan/envs/sim -p /home/jovyan/envs/sim-mamba -y
conda activate /home/jovyan/envs/sim-mamba
MAMBA_KEEP_CUDA_BUILD=TRUE pip install causal-conv1d mamba-ssm --no-build-isolation
```

`MAMBA_KEEP_CUDA_BUILD=TRUE` 会优先拉匹配的预编译 CUDA wheel(本环境对应
`cu12torch2.4cxx11abiFALSE-cp310`),没有 nvcc 也能装上融合 kernel;
`--no-build-isolation` 保证 pip 用现有的 CUDA 版 torch 而不是去装 torch-cpu。

拿到融合 kernel 的好处:§5b 的效率曲线是**真实 kernel 的实测**,而不是纯 python 递推的
数字,效率主张才立得住。

---

## 4. 主实验矩阵(C1:必要性)

```
                      MDP(全观测)        POMDP(drop_linear_vel)
mlp   (单帧)              A                      A'      ← 应崩
stack (L=16)             ≈A                      中
gru   (L=16)             ≈A                      好
transformer (L=16)       ≈A                      好
mamba (L=16)             ≈A                      好(≥transformer)
```

- **左列全部打平** = 增益不是模型容量带来的 ← 这一列不能省,是论证的另一半
- **右列拉开差距** = 增益来自历史信息
- 3 seeds × 10 cells = 30 runs
- 指标:`tracking_error`(主), `episode_len`(存活), `action_smoothness`, return
- 评估复用论文①②已有的 `scripts/evaluate.py` / eval 管线与统计口径

---

## 5. 效率实验(C2:Mamba 的正当性)—— 必须做成一级贡献

⚠️ **陷阱**:L=16 时 Transformer 的 O(L²) 根本不咬人,实测 wall-clock 几乎无差 →
会得到"理论更快但测不出来"的尴尬结果。所以必须:

**5a. 上下文长度扫描(精度侧)**
`L ∈ {1, 4, 16, 64, 256}` × {transformer, mamba},在需要长历史的 POMDP 场景
(`hydro_randomize` 系统辨识 / `sparse_pos_period=20` 稀疏定位)上跑。
论证 AUV 确实需要长上下文 → 二次复杂度才真正成为问题。

**5b. 推理成本曲线(部署侧)**
两种模式分别测,横轴 L,纵轴每步延迟 + 峰值显存:
- **窗口模式**:每步重新前向整个窗口(Transformer 只能这样)
- **流式模式**:Mamba carry 递归 state,每步 O(1)、显存常数

在受限算力档位上报(L40 上限流 + 明确标注目标平台算力等级),配合 50Hz 实时控制预算线。
这张图是"为什么必须是 Mamba 而不是 Transformer"的全部支撑。

**5c. 训练吞吐**:相同 wall-clock 预算下的学习曲线(实践者最关心)。

---

## 6. 实施顺序(每步都可独立验证)

1. **POMDP 开关** — `track.py` + `Track.yaml`;用现有 PPO-MLP 跑一轮
   `drop_linear_vel=true`,**确认单帧策略确实崩**。题目不成立就立刻止损,不投入后面。
2. **观测历史 ring buffer** — `obs_history.py`;`L=1` 时数值上必须与改动前完全一致(回归测试)。
3. **编码器模块** — `encoders.py`:mlp / stack / gru / transformer,先不含 mamba;
   跑 MDP 列确认全部打平(健全性检查)。
4. **Mamba 编码器** — 纯 torch selective scan;单元测试对拍(并行扫描 vs 朴素循环,数值一致)。
5. **主矩阵 30 runs** + 评估汇总。
6. **效率实验 5a/5b/5c** + 出图(用 `scientific-figure-making` 的 house style,与论文①②一致)。

**运行入口**:`scripts/run_mamba.sh`,照抄 `run_flow.sh` 的 PYTHONPATH 自检 ——
editable 安装(`__editable__.marinegym-1.0.pth`)指向主树 `/home/jovyan/MarineGym`,
不压过去会**静默地跑主树代码**,这个坑论文②已经踩过。

---

## 7. 已知风险

| 风险 | 缓解 |
|---|---|
| `drop_linear_vel` 后 MLP 没崩(反而学会了) | 说明 4 个未来参考点泄漏了太多信息;加 `obs_delay` 或减 `future_traj_steps` 加剧 POMDP |
| L=16 时 Transformer 与 Mamba 精度打平且延迟无差 | 靠 §5a 把 L 推到 64~256;若长上下文也无增益,则效率主张退为"同等精度下更省",仍成立但弱 |
| 预编译 wheel 与 torch2.4 不匹配 / 装不上 | 退路:纯 PyTorch selective scan(数学等价,~150 行),只影响训练吞吐与 5b 的绝对数值,不影响 C1 |
| 克隆环境与 Isaac Sim 激活脚本冲突 | 克隆后先跑一次现有训练冒烟测试,确认 Isaac 能正常起 |
| 参数量未对齐被质疑 | §3 的 param-matched 配置,论文里列表报告每个编码器参数量 |

---

## 8. 环境状态(2026-08-25 已验证)

**环境**:`/home/jovyan/envs/sim-mamba`(`/home/jovyan/envs/sim` 的克隆)。
论文①② 的 `envs/sim` **未做任何改动**。

| 组件 | 版本 | 状态 |
|---|---|---|
| python / torch | 3.10.20 / 2.4.0+cu121 (`CXX11_ABI=False`) | 与 sim 一致,未被 pip 改动 |
| `causal_conv1d` | 1.5.0.post8 (`cu12torch2.4cxx11abiFALSE-cp310` 预编译 wheel) | ✓ `causal_conv1d_cuda` 已加载 |
| `mamba_ssm` | **2.2.4** (同上 wheel) | ✓ `selective_scan_cuda` 已加载 |
| 附加依赖 | `huggingface_hub`, `transformers==4.44.2`, `einops`, `ninja` | 仅 mamba_ssm 导入链需要 |
| Isaac Sim 4.10 + Track 训练 | — | ✓ 冒烟测试跑通 2 个 iter |
| Mamba1 / Mamba2 前向+反向 (L40) | — | ✓ 两者均可用,主用 Mamba1 |

### 版本踩坑(不要再试别的组合)

- `MAMBA_KEEP_CUDA_BUILD=TRUE pip install mamba-ssm` **失败**:pip 找不到匹配 torch2.4 的
  预编译 wheel → 回退源码编译 → 无 nvcc → `NameError: bare_metal_version is not defined`。
  必须**按精确 URL 装 wheel**。
- `mamba_ssm 2.2.6.post3` **不能用**:它 import `causal_conv1d.cpp_functions`(1.6+ 才有),
  而 causal_conv1d 只有 ≤1.5.4 提供 torch2.4 的 wheel → 融合 kernel 静默失效,
  运行时报 `causal_conv1d_cuda is not available`。→ **锁定 mamba_ssm 2.2.4**。
- `transformers` 必须 `==4.44.2`:mamba_ssm 2.2.4 依赖已被新版删除的
  `GreedySearchDecoderOnlyOutput` 等符号。

### 流式推理已验证(§5b 的基础)

`Mamba.forward(x[:, t:t+1], inference_params=ip)` 递推与整窗并行扫描**数值一致**
(最大误差 1.19e-07),且每步延迟与上下文长度 L 无关:

| L | 整窗重算 (ms) | 流式 1 步 (ms) |
|---:|---:|---:|
| 8 | 0.519 | 0.310 |
| 32 | 0.859 | 0.236 |
| 128 | 1.096 | 0.228 |
| 256 | 1.982 | 0.404 |
| 512 | 3.027 | 0.219 |

(B=64, d_model=128, d_state=16, L40, eval 模式, 50 次平均)

→ 流式一步是平的(~0.22-0.4 ms),整窗重算随 L 增长。**注意**:这里的"整窗重算"是 Mamba
自己的并行扫描(O(L)),论文里真正要对比的是 Transformer 的 O(L²) 窗口重算,曲线会更陡。

### 启动方式

```bash
bash /home/jovyan/MarineGym-mamba/scripts/run_mamba.sh train.py task=Track ...
```
脚本内含两道自检:`marinegym` 必须解析到本 worktree(editable 安装默认指向主树),
`sys.prefix` 必须是 `envs/sim-mamba`。**不要用 `wandb.mode=disabled`** —— 会导致
`train.py:326` 存 ckpt 时目录不存在而报错;用默认的 `offline`。

---

## 9. 实现状态

### 新增/改动文件

| 文件 | 作用 |
|---|---|
| `marinegym/envs/utils/pomdp.py` | **新增** `ObsDegrader`(传感器缺失/延迟/稀疏定位/噪声/丢包) + `ObsHistory`(GPU 环形缓冲) |
| `marinegym/learning/modules/encoders.py` | **新增** 5 个可插拔编码器,统一契约 `[..., L, D] → [..., H]` |
| `marinegym/envs/single/track.py` | 观测出口施加退化 + 推入历史窗口;观测 spec 改为 `(1, L, obs_dim)` |
| `marinegym/learning/ppo/ppo.py` | `make_mlp([256,256,256])` → `build_encoder(cfg.encoder)`;打印真实参数量 |
| `cfg/task/Track.yaml` | 新增 `context_len` 与 `pomdp:` 段(默认全关 = 原全观测 MDP) |
| `scripts/train.py` | 新增 `+p3_out=<json>` 钩子:落盘学习曲线 + 多 seed 确定性评测 |
| `scripts/run_mamba.sh` | 启动器(PYTHONPATH + conda 前缀双自检) |
| `scripts/p3_sweep.sh` | 主实验矩阵 |
| `scripts/p3_report.py` | 结果汇总 → 论文表格 + 主图/学习曲线 |
| `scripts/p3_efficiency.py` | §5b 效率实验 → 延迟-上下文长度曲线 |

### 参数量对齐结果(obs_dim=43, L=16)

| arm | actor 参数量 | 说明 |
|---|---:|---|
| mlp | 144,384 | 只看当前帧 —— 下界基线 |
| mlp_wide | 314,880 | **容量对照**:参数量约 2 倍,堵住"MLP 只是容量不够"的质疑 |
| stack | 188,288 | 首层输入 L×D,参数量随 L 线性增长 |
| gru | 138,240 | |
| transformer | 171,648 | DTQN 对标 |
| **mamba** | **155,904** | **ours —— 比 transformer 参数更少** |

序列臂全部落在 138k–188k,±15% 以内。Mamba 参数比 Transformer 少 9%,若仍胜出则结论更强。

### 已验证(单元测试)

- `ObsHistory` 时间顺序 = 最老→最新,末位 = 当前帧
- 逐 env 重置隔离:重置的 env 用当前帧填满窗口,未重置的 env 不受影响(跨 episode 无污染)
- 传感器屏蔽只清零指定分量,其余原样
- 观测延迟 k=2:第 t 步输出第 t−2 步的值
- USBL 稀疏定位 period=3:位置每 3 步刷新,非位置分量不受影响
- 编码器形状折叠:批量算 vs 逐条算误差 1.4e-06
- 除 mlp 外各臂改动窗口最老帧都会改变输出(确认历史真的被用上)

### 为什么 POMDP 设定选"无 DVL + 无速率陀螺"

观测里 `rpos[k] = target(t+5k) − p(t)`,相邻项之差 `rpos[k+1] − rpos[k]` 只含**参考轨迹**
速度,不含载具自身速度。所以屏蔽 `vel(6)` 之后,单帧策略**无法**得知自身速度 ——
二阶系统只有位置反馈无法定阻尼,必须从历史估微分。这是控制论上最干净的论证。

> 诚实说明:观测里仍保留 `throttle`(当前推进器/舵面状态,即动作的滞后版本),
> 它与速度弱相关,是一个温和的信息泄漏。但它对所有臂一视同仁,且在真实 AUV 上
> 本体感知确实可得,故保留。

---

## 10. 与上游实现的对照(审稿人会查)

### Transformer 臂 —— 对照 `kevslinger/DTQN`

| 项 | DTQN 上游 | 本文实现 |
|---|---|---|
| 位置编码 | `--pos learned`(默认),按 context_len 分配 | 同 |
| 层归一化位置 | `--identity` 默认 False → `TransformerLayer`,LN 在跳连**之后** | 同(`identity=True` 可切 GTrXL 预归一化) |
| 门控 | `--gate res`(默认) / `gru`(GTrXL,`w_z.bias=-2`) | 两者都实现,默认 res |
| 子层输出 | 先过 ReLU 再进门控 | 同(照抄) |
| FFN | `Linear(d,4d) → ReLU → Linear(4d,d) → Dropout` | 同 |
| causal mask | `torch.triu(ones(L,L), diagonal=1)` 上三角置 `-inf` | 同(**已缓存**,否则每步重分配会人为拖慢基线) |
| dropout | 0.0 | 同 |
| 默认规模 | `--in-embed 128 --heads 8 --layers 2` | `transformer_dtqn` 臂用此原生配置 |

**有意偏离(一处,已在论文中声明)**:DTQN 会在窗口每个时刻都预测 Q 值并一起训练
(`--history`,intermediate Q-values)。那是 Q-learning 专有的样本效率技巧;本文是
on-policy actor-critic(PPO),损失只定义在当前动作上,故只取末位。
该差异对所有序列臂一视同仁,不影响 arm 之间的比较。

### Mamba 臂 —— 对照 `state-spaces/mamba`

不自己实现 SSM 或残差接法,全部用官方组件:

- `from mamba_ssm import Mamba` —— README 里的官方模块
- `create_block(d_model, d_intermediate=0, ssm_cfg={d_state,d_conv,expand}, ...)`
  —— 官方 Block 工厂(`d_intermediate=0` = 纯 Mamba block,与官方 Mamba-1 一致)
- 前向的 `residual` / `norm_f` 处理**逐行照抄** `MixerModel.forward`
  的 `fused_add_norm=False` 分支
- `rms_norm=False`(= `create_block` 签名默认值)。triton 版 RMSNorm 首次调用要
  autotune,会额外申请数百 MB 显存,在共享 GPU 上直接 OOM

### 一个会静默毁掉 Mamba 的坑

PPO 原本对**所有** `nn.Linear` 施加 `orthogonal_(0.01)` + `bias.fill_(0)`。
这会:(a) 在 Mamba 的 `in_proj/out_proj/x_proj`(`bias=False`)上抛 `NoneType`;
(b) 更严重 —— **覆盖掉 `dt_proj.bias` / `A_log` / `D` 的官方初始化**,毁掉 SSM 参数化。
GRU 的 `uniform(±1/√H)`、GRUGate 的 `w_z.bias=-2` 同理。

→ 各编码器提供 `protected_modules()`,PPO 只对策略头/值头/输入投影施加初始化。
已单测验证 `dt_proj.bias` / `A_log` / `D` / `in_proj.weight` 全部保持官方初始化,
而 head 第一层确实被 orthogonal 初始化。

---

## 11. 共享 GPU 的现实约束

这台机器有 4×L40(46 GB),但**其他用户常年占掉每卡约 44 GB**,实测只剩 0.3–5.7 GB。
单个 Isaac job 峰值约 2.5–3 GB,4 并行必 OOM(pilot 已实测掉了一格)。

`p3_sweep.sh` 因此实现了:
- `pick_gpu`:选剩余显存最多且 ≥ `MIN_FREE_MB`(默认 3500)的卡
- `wait_for_gpu`:没卡就等(最多 `WAIT_MAX`,默认 2h),不硬投
- `run_with_retry`:OOM 自动重试(默认 3 次,每次重新选卡,间隔 3 分钟)
- `MAX_PAR` 默认 2,启动错开 20 s

单 run 耗时:训练 ~33 s/iter × 120 iter ≈ 66 min,评测 ~31 s(聚合 ≥300 条 episode)。

---

## 12. Pilot 结果 —— 原定 POMDP 设定不成立(2026-08-25)

按 §6 的止损设计,先跑 40 iter 的 pilot 验证效应方向。**结论:原设定不可用。**

| run (40 iter, seed 0) | ep_len ↑ | return ↑ | tracking_ema ↓ | entropy 变化 |
|---|---:|---:|---:|---|
| MDP / MLP | 77.96 | 65.51 | 0.4525 | 4.251 → **4.059** |
| POMDP / MLP(去速度) | 76.26 | 64.31 | 0.4513 | 4.249 → 4.238 |
| POMDP / Mamba | 63.12 | 53.78 | 0.4444 | 4.254 → 4.242 |

**问题一:屏蔽速度不构成有效的部分可观测。** MLP 在 POMDP(76.26)与 MDP(77.96)几乎无差。

原因分析:
- 水下载具**阻力主导**,系统接近一阶(速度 ≈ 推力的瞬时函数),
  而观测里保留的 `throttle`(当前推进器状态)几乎完整泄漏了速度;
- 因此"二阶系统需要微分反馈"这条对无人机成立的论证,**对 AUV 不成立**。

**问题二(更根本):任务疑似能力受限而非信息受限。**
`tracking_error_ema ≈ 0.45` 紧贴 `reset_thres = 0.5`,episode 只有 76/600 步 ——
载具约 1.2 s 就跟丢。论文① 的数据也只有 ~67 步(且那是放宽到 `reset_thres=1.5` 之后)。
参考速度 ≈ `traj_scale`(1.4–2.6 m) × `traj_w`(0.7–0.9 rad/s) ≈ **1.6 m/s**,
对小型 iAUV 偏快。**任务若是能力受限,再多信息也无法改善,POMDP 自然测不出差别。**

### 已排除:不是 Mamba 实现的问题

用一个**解析上可判定**的监督探针隔离"实现/优化"与"任务"两件事
(`scripts/p3_probe.py`,3 seeds):

> 输入随机游走窗口 `x[t]=x[t-1]+d[t]`,目标为最后一步增量 `d[L-1]`。
> 单帧观测与目标**统计独立** ⇒ 只看当前帧的最优 MSE = 目标方差 0.09。

| arm | step 0 | step 500 | final | final/目标方差 | 判定 |
|---|---:|---:|---:|---:|---|
| MLP(只看当前帧) | 0.4253 | 0.0860 | 0.0887 | **0.986** | 学不会(=猜均值) |
| Frame-stack | 0.4291 | 0.0491 | 0.0013 | 0.014 | ✓ |
| GRU | 0.3665 | 0.0069 | 0.0010 | 0.011 | ✓ |
| Transformer | 0.3971 | 0.0492 | 0.0003 | 0.003 | ✓ |
| **Mamba** | 0.4312 | **0.0060**(最快) | 0.0003 | 0.004 | ✓ |

→ **Mamba 的实现与优化没问题,而且是收敛最快的臂。** 问题出在任务设定。
这张图(`fig0_probe`)本身也进论文,作为"编码器确实能从历史中恢复所需量"的验证。

### 后续筛选(进行中)

| 档 | 设定 | 假设 |
|---|---|---|
| `slow` | `traj_w_range=[0.3,0.5]` | 参考降到 ~0.8 m/s,任务变可跟踪 ⇒ 指标有区分空间 |
| `slow_pomdp` | slow + 去速度 | 可跟踪任务上,信息缺失才可能显现 |
| `pomdp_d10` | 去速度 + 观测延迟 10 步(160 ms) | 延迟迫使外推,单帧无法补偿(声学链路延迟) |
| `pomdp_sp10` | 去速度 + 位置每 10 步刷新 | USBL 稀疏定位,必须航位推算 |
| `drift`(兜底) | 未观测的逐 episode 推进器增益 [0.5,1.0] | 教科书式在线系统辨识,单帧原理上不可解 |

### 共享 GPU 的额外治理

其他租户 + 另两个 session 把每卡吃到只剩 ~2.5 GB,mamba 反复 OOM。
新增 `LOWMEM_OVERRIDES`:把 PhysX 的 GPU 缓冲(`found_lost_aggregate_pairs` 3355 万、
`heap` 64 MB、软体/粒子接触各 100 万)按实际场景(64 刚体、无软体无粒子)调小。
**只改缓冲容量,不动任何物理参数**(dt/solver/摩擦等一律不变)。
实测单进程占用 **2.35 GB → 1.42 GB**。

---

## 13. 窗口必须按步长采样(帧跳) —— 一个非平凡的实现要求

pilot 里所有序列臂(尤其 Mamba)在 RL 上完全学不动,而监督探针 A 里它们学得又快又好。
差异在**信噪比**:

- 探针 A 的增量 σ=0.3,是大信号;
- Track 任务:控制 62.5 Hz(dt=0.016),位置量级 ~2 m,每步位移仅
  1.6 m/s × 0.016 s ≈ **2.6 cm**,**相邻帧相对差异约 1%**。
  stride=1 的 16 帧窗口只跨越 256 ms,序列模型要在极差信噪比下分辨这 1%。

### 探针 B(`p3_probe.py`,量纲对齐真实任务)

构造:`x[t] = offset + v·t + ε[t]`,`ε ~ N(0, 0.01 m)`,offset 量级 2 m,
每步位移 0.026 m,目标 = v(已归一化,MSE≈1.0 表示完全学不会)。
相隔 k 帧估计 v 的噪声为 `√2·σ/k` ⇒ **信噪比正比于帧间隔**。相邻帧信噪比仅 ≈1.8。

| arm | stride=1 | stride=2 | stride=4 | stride=8 | stride=16 |
|---|---:|---:|---:|---:|---:|
| Frame-stack | 0.058 | 0.036 | 0.027 | 0.024 | 0.023 |
| GRU | 0.107 | 0.034 | 0.021 | 0.013 | 0.007 |
| Transformer | 0.029 | 0.023 | 0.015 | 0.009 | 0.005 |
| **Mamba** | **0.288** | 0.020 | 0.012 | 0.011 | 0.011 |

**Mamba 在 stride=1 时比 stride=2 差 24 倍 —— 它对"相邻帧近乎重复"最敏感**
(因果 conv1d 与选择性扫描正是在相邻帧之间做差分/门控)。
stride≥2 之后 Mamba 立刻成为最好或接近最好的臂。

这解释了 pilot 中 `pomdp__mamba` 曲线全平(60.1 → 59.8)的现象 ——
**不是 Mamba 不行,是窗口采样方式不对**。

### 选定 stride=4,并对所有序列臂统一施加

- L=16 × stride=4 ⇒ 窗口跨越 61 步 ≈ 0.98 s,与 episode 长度(~76 步)同量级;
  再大(stride=8 → 121 步)就超出 episode,窗口里大半是 reset 帧,无意义。
- Mamba 在 stride=4 已到平台期(0.012 vs 最佳 0.011);
- **stride 是所有序列臂共享的超参,取值相同 ⇒ 公平**。

> 这也是论文里该诚实写出的一条 Mamba 适用条件:高控制频率下必须做帧跳,
> 否则 SSM 的性能塌陷得比 Transformer 更严重。`fig0b_stride` 即为此图。

---

## 14. 最终实验设定:未观测的执行器增益 + oracle 对照(2026-08-26 定)

前面两版 POMDP 都被否掉:

- **v1 屏蔽速度**:MLP 在 POMDP(76.26)与 MDP(77.96)几乎无差 —— 水下阻力主导、
  系统接近一阶,且观测里的 `throttle` 泄漏了速度。
- **v2 观测延迟 160 ms**:确实拖慢了 MLP(76.26 → 63.11),但 40 iter 下所有臂都
  还没脱离随机策略阶段(entropy 4.25 → 4.19),无法区分;而且"任务变难了"与
  "信息变少了"两种解释纠缠在一起,审稿人会问。

### v3(采用):`drift` vs `drift_oracle`

```
drift          u_applied = gain ⊙ u_policy,  gain ~ U(0.5, 1.0) 逐通道逐 episode
               gain 不进观测                                    → POMDP
drift_oracle   完全相同的 gain 采样与施加，但 gain 归一化后拼进观测  → MDP(oracle)
```

**这一对的动力学完全相同、任务难度完全相同,唯一差别是策略知不知道 gain。**
于是:

- `drift_oracle` 列各臂应打平 ⇒ 差距不是模型容量、也不是任务变难;
- `drift` 列拉开 ⇒ 差距**只能**来自"必须从历史在线辨识 gain"。

为什么必须要历史:gain 不出现在任何单帧量里,也无法由单帧推出;
只有把"下了多少油门"与"实际产生了多少运动"在时间上对照才能辨识。
而且**不知道 gain 就无法正确控制**(推力可能只有名义值的一半)——
所以这个信息既是历史独有的,又是任务必需的。这是教科书式的在线系统辨识 POMDP。

实测 obs_dim:`drift` 37 维,`drift_oracle` 40 维(iAUV 动作维度 3,逐通道增益)。

### 训练预算

40 iter 下 entropy 几乎不动(4.25 → 4.19),策略仍近乎随机,**无法用于比较**。
120 iter 下 `drift__mlp` 的 entropy 降到 3.804、训练 ep_len 62 → 87,才是可比的状态。
故最终矩阵一律 **120 iter**(与论文①② 一致),单格约 69 min。

---

## 15. seed 0 结果:`drift` 列(2026-08-26)

120 iter,单 seed,评测聚合 ≥300 条 episode(确定性策略)。

| arm | ep_len ↑ | return ↑ | 参数量 | 备注 |
|---|---:|---:|---:|---|
| Transformer (DTQN-style) | **82.58** | 70.39 | 143,142 | |
| **Mamba (ours)** | **81.84** | 69.13 | 156,166 | 训练到 120 iter **仍在上升** |
| MLP (单帧) | 76.12 | 64.85 | 143,622 | iter~85 见顶后回落 |
| Frame-stack | 74.18 | — | 176,000 | **拿到同样的历史却比单帧还差** |
| GRU | 58.63 | 50.91 | 137,472 | 崩了 |

### 三条必须如实写的结论

**1. 不是"有历史就行",而是"要有合适的序列归纳偏置"。**
Frame-stack 拿到了与 Transformer/Mamba **完全相同**的窗口,却比单帧 MLP 还差
(74.18 vs 76.12);GRU 更是崩到 58.63。只有 Transformer 与 Mamba 提取出了有用信息。
论文的主张应写成架构主张,而不是笼统的"历史有用"。

**2. Transformer 略高于 Mamba(82.58 vs 81.84),差距很可能在噪声内。**
所以本文的主张必须是 **"Mamba 以远低的推理代价达到 Transformer 的精度"**,
而不是"Mamba 精度更高"。这也正是 Mamba 原论文的标准主张。

**3. 训练形态不同:** Mamba 前 ~40 iter 几乎不动(曲线 61→62),之后才起飞并
**在 120 iter 仍未收敛**;MLP 早期爬得快但 iter~85 见顶后回落。
若延长训练,差距很可能进一步扩大 —— 但本文按论文①② 的口径统一 120 iter,
并在文中注明 Mamba 尚未收敛。

### 意外:oracle 对照不成立

`drift_oracle__mlp = 76.43` vs `drift__mlp = 76.12`,**差 -0.32** ——
把真实增益直接喂给 MLP,它一点没变好。**增益信息对策略根本没用。**

⇒ 序列模型的优势**不是**来自"辨识执行器增益",§14 设想的机制不成立。

### 新假说:优势来自恢复**加速度**(附加质量项)

水下载具的水动力含**附加质量**项,依赖于**加速度**
(`hydro_wrench(vb, v_prev, a_prev, ...)`),而观测里有位置/速度/姿态/油门,
**唯独没有加速度**。加速度只能对速度做时间差分得到 ——
这是 Track 任务**固有的**部分可观测性,与人为加的增益漂移无关。

已实现 `pomdp.accel_in_obs`(把 `(v_t − v_{t−1})/dt` 白送给策略)作为**机制验证**:
若序列模型的优势因此消失 ⇒ 机制被证实;若不变 ⇒ 另有原因,继续查。

> 这条若成立,比原设定更有价值:**"AUV 控制需要历史"的根源是附加质量项需要加速度**,
> 是水下机器人特有的机制,而不是一句泛泛的"POMDP 需要记忆"。

---

## 16. POMDP 机制系统性筛选(2026-08-26)—— 找到唯一有效的条件

### 16.1 方法

固定策略为**单帧 MLP**(无历史),逐一施加部分可观测机制,看 `episode_len`
相对**同任务**无 POMDP 基准掉多少。只有明显掉下去的条件,才值得用来比较序列架构。
2048 envs × 40 iter,5 个独立评测 seed(共同随机数),误差棒约 ±0.5%。

### 16.2 结果

**任务 A:原始参考速度(能力受限,基准 ep_len 57.9)**

| 机制 | ep_len | vs 基准 |
|---|---:|---:|
| 无 | 57.9 | — |
| 执行器增益漂移 U(0.5,1.0) | 57.9 | +0.0% |
| 增益 oracle(把增益告诉策略) | 57.9 | +0.0% |
| 观测延迟 160 ms | 57.9 | +0.0% |
| **切断前馈(去未来参考点+时间编码)** | **74.8** | **+29.3%** |

> ⚠ 注意最后一行:**去掉信息反而变好**。原因是载具追不上参考时,
> "瞄准前方"有害 —— 给它未来参考点它会往前赶,反而更快被甩开;
> 只给当前点它老实追最近目标,撑得更久。
> **这是"能力受限"最有力的旁证:信息在能力受限的任务里可以是负价值的。**

**任务 B:调慢参考(可跟踪,基准 ep_len 216.2)**

| 机制 | ep_len | vs 基准 | 判定 |
|---|---:|---:|---|
| 无 | 216.2 ± 0.75 | — | 基准 |
| 无 DVL + 无速率陀螺 | 215.6 ± 0.83 | −0.2% | 无效 |
| USBL 定位 6.25 Hz(每 10 步) | 215.6 ± 1.04 | −0.3% | 无效 |
| 切断前馈 | 215.3 ± 0.67 | −0.4% | 无效 |
| 切断前馈 + 无 DVL | 213.9 ± 0.88 | −1.1% | 无效 |
| **USBL 1 Hz(每 60 步) + 无 DVL** | **167.9 ± 1.08** | **−22.3%** | **有效** |
| **USBL 0.5 Hz(每 125 步) + 无 DVL** | **166.6 ± 0.78** | **−23.0%** | **有效(已饱和)** |

### 16.3 为什么只有这一个有效 —— 交互效应

| 无 DVL | USBL 稀疏 | ep_len | |
|---|---|---:|---|
| ✗ | ✗ | 216.2 | 基准 |
| ✓ | ✗ | 215.6 | −0.2% |
| ✗ | 6.25 Hz | 215.6 | −0.3% |
| **✓** | **1 Hz** | **167.9** | **−22.3%** |

**两个单独都无效,合起来才致命。** 物理上说得通:

- 有 DVL 时,可以靠速度积分在两次定位之间推算位置;
- 有高频定位时,不需要速度;
- **只有两者同时缺失,载具在两次定位之间才真的"瞎了"** ——
  唯一的信息来源是自己下过的推力序列,必须**积分推力历史做航位推算**。
  这是单帧观测在信息论上不可能完成的。

而前面 6 种失效的机制,本质都只是"让策略看得少一点",
但**只要目标位置持续可见,「追当前目标点」就是无记忆最优策略**,比例控制就够。

### 16.4 这个场景的论文价值

**声学定位速率受限是 AUV 的物理约束,不是人为设定的困难。**
水中声速 ~1500 m/s,USBL 往返时延决定了定位速率只能是 0.5–2 Hz;
而 DVL 在离底超程、气泡、软底质时失效也是常态。
"用序列建模在低频定位 + DVL 失效下维持跟踪精度"是**真实的工程问题**。

> 效应在 1 Hz 就饱和(0.5 Hz 与 1 Hz 几乎相同),说明损失来自
> "失去连续定位"这一质变,而非稀疏程度的量变。
