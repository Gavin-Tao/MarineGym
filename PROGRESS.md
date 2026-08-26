# 论文③ 进度（2026-08-26 14:30 更新）

分支 `mamba`，worktree `/home/jovyan/MarineGym-mamba`，环境 `/home/jovyan/envs/sim-mamba`。
设计取舍见 `paper3_mamba_design.md`，方法规格见 `METHODS.md`。

---

## 0. 一句话现状

**基础设施与所有已知 bug 已清理干净；今天最大的收获是两个"关于基准本身"的发现
（原始 Track 配置不可用于架构比较、评测协议有多个静默 bug）。
目前正在可跟踪的任务设定上重筛 POMDP 机制，之后才做头条表。**

---

## 1. 今天确立的三个结论（都可写进论文）

### 1.1 原始 Track 配置是**能力受限**，不能用于比较策略架构

4096 envs × 20M frames 下，所有臂结果一致到小数点后第三位：

| 条件 | 编码器 | 参数量 | 跟踪误差(m) | ep_len |
|---|---|---:|---|---|
| drift | MLP（单帧） | 143,622 | 0.1929 ± 0.0001 | 57.86 |
| drift | Frame-stack（16帧） | 176,774 | 0.1933 ± 0.0001 | 57.85 |
| drift | Transformer（DTQN，3×参数） | 437,766 | 0.1929 ± 0.0001 | 57.87 |
| delay10 | MLP（+160 ms 延迟） | 143,622 | 0.1942 ± 0.0002 | 57.86 |
| delay10 | Transformer | 437,766 | 0.1938 ± 0.0002 | 57.84 |

机制（实测轨迹，`fig5_traj`）：参考点速度 0.543 m/s，载具只跑出 0.279 m/s（**51%**），
误差以 0.413 m/s 线性增长，撞 `reset_thres=0.5 m` 的时刻由**运动学**决定。

### 1.2 `tracking_err_mean` 在提前终止机制下不是有区分度的指标

误差从 0 线性长到阈值就终止 ⇒ 均值必然 ≈0.19，与增长速率无关。
三个参考速度档（ep_len 57.9 / 128.7 / 182.1，差 3.1 倍）下它是 0.1929 / 0.1939 / 0.1943。
**主指标应为 `episode_len`。**（已更正 `METHODS.md` 与 `p3_report.py`）

### 1.3 调慢参考轨迹后任务变为可跟踪

| traj_w | ep_len（MLP, 40 iter） |
|---|---:|
| [0.7,0.9]（原始） | 57.9 |
| [0.3,0.5] `slow` | 128.7 |
| [0.2,0.35] `slow2` | 182.1（4096 envs）/ 216.2（2048 envs） |

---

## 2. 正在做

**POMDP 机制重筛**（MLP 单帧，slow2 可跟踪任务，2048 envs，40 iter，3 卡并行）：

| 条件 | 机制 | 状态 |
|---|---|---|
| `s2` | 无 POMDP（基准） | ✅ ep_len 216.2 |
| `s2_novel` | 无 DVL + 无陀螺 | 跑中 |
| `s2_d20` | 观测延迟 320 ms | 跑中 |
| `s2_sparse` | USBL 稀疏定位（位置每 10 步刷新） | 排队 |
| `s2_noise` | 观测噪声 σ=0.02 | 排队 |
| `s2_hard` | 上述叠加 | 排队 |

**判据**：哪个让 MLP 的 `episode_len` 相对 216.2 显著下降，哪个才是有效 POMDP。

---

## 3. 后续步骤

1. POMDP 重筛出结果 → 选定主条件（~1 h）
2. 在主条件上跑头条表：DTQN 原生 Transformer(437k) vs Mamba 1层(155k) vs MLP(143k)，
   4096 envs × 20M frames（~3 h）
3. 容量扫描 Mamba d96/d64 + 反向对照 → 支撑"更少参数、相当精度"
4. `bash scripts/p3_paper_bundle.sh` 出完整图表包

---

## 4. 已定稿、不依赖后续训练的产物

| 产物 | 内容 |
|---|---|
| `fig0_probe` + `table_probe` | 编码器验证探针：单帧 MLP 的 MSE 卡在理论下界，序列臂全部远低于它 |
| `fig0b_stride` + `table_stride` | **Mamba 在 stride=1 时比 stride=2 差 24 倍**（62.5 Hz 下相邻帧近乎重复） |
| `fig3_efficiency_b1/b256` | 推理代价 vs 上下文长度（batch=256, L=256：Transformer 18.3 ms vs Mamba 流式 0.24 ms） |
| `fig5_traj` + `table_capability` | 轨迹对比 + 能力受限的定量证据 |

---

## 5. 已修复的 bug（共 13 个）

| # | 问题 | 影响 |
|---|---|---|
| 1 | PPO 的 `orthogonal_(0.01)` 覆盖 Mamba/GRU 的上游初始化 | 会毁掉 SSM 参数化 |
| 2 | DTQN 位置编码按 `max_len=1024` 分配 | 虚增 13 万参数 |
| 3 | Transformer 每步重建 causal mask | 人为拖慢基线 |
| 4 | 用 `nn.TransformerEncoder` 封装 | 推理延迟虚高 20 倍 |
| 5 | 评测只取每 env 第一条 episode | 浪费 90% 数据 |
| 6 | GRU 缺 `LayerNorm(h+input)` | 训练发散，NaN 静默变 0 |
| 7 | 窗口 stride=1 | Mamba 性能塌陷 24 倍 |
| 8 | num_envs=64 | 策略欠训练，架构差异被压缩 |
| 9 | 4096 envs 下 critic 整批前向 | OOM |
| 10 | `NameError: diverged` | 5 格训练完在写结果时崩掉 |
| 11 | 评测批数按快速轨迹写死 | 慢速任务下收不到 episode |
| 12 | **`EpisodeStats._episodes` 是累计计数器** | 评测第一批就退出，n 被严重误报 |
| 13 | GPU 双订（nvidia-smi free 有 2 min 滞后） | 并行 job OOM |

> 另有一个**认知**错误：`train.py` 调 `setproctitle(run.name)` 把命令行替换成
> `Track-ppo-iAUV/<时间>`，导致 `pgrep -f "p3_out="` / `grep train.py` 全都匹配不到，
> 我多次误判"job 已死"。查我方 job 应查 `pgrep -f "Track-ppo"` 或 GPU 锁文件。

---

## 6. 诚实评估

今天**尚未产出可用于论文的最终数据**。大部分时间用在发现并修复上述问题上。
但 §1 的三个结论本身有价值，尤其 1.1 —— 它说明这个基准在默认配置下
无法用于策略架构研究，这对使用 MarineGym 的其他人也是实质信息。
