# AEI 论文数据总表

> 生成于 2026-08-25，全部数字由 `scripts/outputs_aei/data/*.csv` 直接汇总，未手工誊写。
> 复现：`/home/jovyan/envs/sim/bin/python scripts/aei_report.py`

---
## 1. 方法

**RP-PSF** = A(风险监视器) → λ → B(MPPI 精确滤波) → soft blend

```
A: min_clear_pred = min_{k=1..H} ( ‖p_k − p_o,k‖ − (r_o + r_v) )   名义模型零阶保持前滚 H 步
λ: λ = clamp( (h_hi − min_clear_pred) / (h_hi − h_lo), 0, 1 )
B: u_safe = MPPI(K 采样, N 步, 精确 6-DOF 名义模型)
   u_t = u_rl + λ·(u_safe − u_rl)
```

## 2. 冻结超参（全部消融格共用）

| 参数 | 值 | 说明 |
|---|---|---|
| `keepout.radius` | 0.8 | 障碍半径 r_o (m) |
| `vehicle_radius` | 0.3 | 载具等效半径 r_v (m) |
| `违约面` | 中心距 1.1 m | r_o + r_v |
| `risk.horizon H` | 15 | 前滚步数 |
| `risk.risk_norm` | 2.0 | risk 标量归一化 |
| `risk.in_obs` | true | risk 进观测(45 维) |
| `门控边界` | 0.6 m | risk.threshold = mppi.soft_hi，全套唯一边界 |
| `mppi.soft_lo` | 0.0 | 斜坡下界 |
| `mppi.num_samples K` | 128 |  |
| `mppi.horizon N` | 20 |  |
| `mppi.exact` | true | 精确 6-DOF 名义模型，一步误差 <2% |
| `mppi.noise_sigma` | 0.4 |  |
| `mppi.temperature` | 0.05 |  |
| `mppi.w_coll / w_track` | 5.0 / 1.0 |  |
| `训练` | 120 iters, train_every=16, seed=0 | 单训练 seed |
| `载具/任务` | BlueROV / Track, traj_scale_mult=2.5 |  |
| `动态障碍` | intercept_steps=[150,350], re_aim=200 | 侧向拦截，瞄准未来参考位置 |

场景由障碍速度区分：**nominal** `[0.4,0.9] m/s`，**ood** `[2.2,2.8] m/s`，**hard** = nominal 速度 + `intercept_steps=[25,55]`。

## 3. 消融矩阵

|  | 斜坡接管 soft | 阶跃接管 binary |
|---|---|---|
| **预测门控** | **RP-PSF (ours)** | Predictive + binary |
| **几何门控** | Geometric + proportional | Geometric + binary |

每格相对 ours 只差一个开关。另有 A-only（`mppi.enable=false`）与 PPO baseline（无 A 无 B）。

### 各格 checkpoint 溯源

| 格子 | 训练 run |
|---|---|
| RP-PSF (ours) | `offline-run-20260711_165556-y7en5cfg` |
| Predictive + binary | `offline-run-20260825_091024-w9bzerfg` |
| Geometric + proportional | `offline-run-20260825_113758-vo772cdb` |
| Geometric + binary | `offline-run-20260825_125027-q19nmx6r` |
| A-only | `offline-run-20260824_211713-sjghvmgu` |
| PPO baseline | `offline-run-20260711_094733-ck77bi9a` |

## 4. 评测结果（ood，每格 10 个独立评测 seed，mean ± std）

| 方法 | episodes | **ep_len** | collision ↓ | return ↑ | detour ↓ | smoothness ↑ | correction ↓ | tracking ↓ | 激活率 |
|---|---|---|---|---|---|---|---|---|---|
| RP-PSF (ours) | 1429 | 67.0 | 0.0169±0.0102 | 29.5125±0.5338 | 0.2215±0.0099 | -0.7829±0.0060 | 0.0220±0.0143 | 1.3484±0.0017 | 0.0176±0.0098 |
| Predictive + binary | 1373 | 68.3 | 0.0153±0.0094 | 30.7512±0.7841 | 0.1197±0.0036 | -0.8116±0.0079 | 0.0357±0.0129 | 1.3540±0.0019 | 0.0245±0.0086 |
| Geometric + proportional | 1067 | 66.2 | 0.0204±0.0125 | 28.9783±0.6690 | 0.2866±0.0114 | -0.7830±0.0079 | 0.0319±0.0227 | 1.3469±0.0020 | 0.0212±0.0128 |
| Geometric + binary | 1087 | 67.2 | 0.0215±0.0126 | 29.3518±0.9063 | 0.1915±0.0087 | -0.8052±0.0063 | 0.0394±0.0203 | 1.3507±0.0030 | 0.0276±0.0134 |
| A-only (no filter) | 1392 | 67.8 | 0.0166±0.0104 | 30.4767±0.4216 | 0.1985±0.0066 | -0.7884±0.0060 | 0.0000±0.0000 | 1.3508±0.0023 | 0.0185±0.0095 |
| PPO baseline | 1418 | 67.9 | 0.0164±0.0102 | 29.3018±0.8144 | 0.2138±0.0060 | -0.7977±0.0057 | 0.0000±0.0000 | 1.3496±0.0032 | 0.0000±0.0000 |

> `ep_len` = 平均 episode 长度（步）。episode 因**跟踪失败**提前终止（`reset_thres=1.5`，
> 上限 600），故它本身是任务性能指标，**不可用它归一化 return**，详见 §9。

### 主结果三场景（PPO vs ours）

| 场景 | PPO collision | ours collision | PPO min_clear | ours min_clear |
|---|---|---|---|---|
| nominal | 0.0078 | 0.0084 | 2.29 m | 2.33 m |
| ood | 0.0164 | 0.0169 | 6.70 m | 6.68 m |
| hard | 0.7476 | 0.7487 | -0.04 m | -0.02 m |

## 5. 全部成对显著性检验（ood，Welch t 检验，真 t 分布）


**接管律 @预测门控** — ours vs P+bin

| 指标 | ours | P+bin | p | \|d\| | 优 |
|---|---|---|---|---|---|
| collision | 0.0169 | 0.0153 | 0.734 n.s. | 0.2 | — |
| return | 29.5125 | 30.7512 | 0.001245 ** | 1.8 | P+bin |
| detour_ratio | 0.2215 | 0.1197 | 5.419e-12 *** | 13.0 | P+bin |
| action_smoothness | -0.7829 | -0.8116 | 1.259e-07 *** | 3.9 | ours |
| correction | 0.0220 | 0.0357 | 0.04753 * | 1.0 | ours |
| tracking_error_ema | 1.3484 | 1.3540 | 3.434e-06 *** | 3.0 | ours |

**接管律 @几何门控** — G+prop vs G+bin

| 指标 | G+prop | G+bin | p | \|d\| | 优 |
|---|---|---|---|---|---|
| collision | 0.0204 | 0.0215 | 0.861 n.s. | 0.1 | — |
| return | 28.9783 | 29.3518 | 0.3343 n.s. | 0.4 | — |
| detour_ratio | 0.2866 | 0.1915 | 3.863e-13 *** | 8.9 | G+bin |
| action_smoothness | -0.7830 | -0.8052 | 4.237e-06 *** | 3.0 | G+prop |
| correction | 0.0319 | 0.0394 | 0.4708 n.s. | 0.3 | — |
| tracking_error_ema | 1.3469 | 1.3507 | 0.006466 ** | 1.4 | G+prop |

**门控信号 @斜坡接管** — ours vs G+prop

| 指标 | ours | G+prop | p | \|d\| | 优 |
|---|---|---|---|---|---|
| collision | 0.0169 | 0.0204 | 0.5173 n.s. | 0.3 | — |
| return | 29.5125 | 28.9783 | 0.07828 n.s. | 0.8 | — |
| detour_ratio | 0.2215 | 0.2866 | 1.922e-10 *** | 5.8 | ours |
| action_smoothness | -0.7829 | -0.7830 | 0.9714 n.s. | 0.0 | — |
| correction | 0.0220 | 0.0319 | 0.2841 n.s. | 0.5 | — |
| tracking_error_ema | 1.3484 | 1.3469 | 0.1166 n.s. | 0.7 | — |

**门控信号 @阶跃接管** — P+bin vs G+bin

| 指标 | P+bin | G+bin | p | \|d\| | 优 |
|---|---|---|---|---|---|
| collision | 0.0153 | 0.0215 | 0.2522 n.s. | 0.5 | — |
| return | 30.7512 | 29.3518 | 0.002604 ** | 1.6 | P+bin |
| detour_ratio | 0.1197 | 0.1915 | 3.103e-11 *** | 10.2 | P+bin |
| action_smoothness | -0.8116 | -0.8052 | 0.07412 n.s. | 0.9 | — |
| correction | 0.0357 | 0.0394 | 0.651 n.s. | 0.2 | — |
| tracking_error_ema | 1.3540 | 1.3507 | 0.01349 * | 1.3 | G+bin |

**有无 B 滤波** — ours vs A-only

| 指标 | ours | A-only | p | \|d\| | 优 |
|---|---|---|---|---|---|
| collision | 0.0169 | 0.0166 | 0.9579 n.s. | 0.0 | — |
| return | 29.5125 | 30.4767 | 0.0005319 *** | 1.9 | A-only |
| detour_ratio | 0.2215 | 0.1985 | 2.863e-05 *** | 2.6 | A-only |
| action_smoothness | -0.7829 | -0.7884 | 0.06545 n.s. | 0.9 | — |
| correction | 0.0220 | 0.0000 | 0.001263 ** | 2.1 | A-only |
| tracking_error_ema | 1.3484 | 1.3508 | 0.02451 * | 1.1 | ours |

**vs baseline** — ours vs PPO

| 指标 | ours | PPO | p | \|d\| | 优 |
|---|---|---|---|---|---|
| collision | 0.0169 | 0.0164 | 0.9186 n.s. | 0.0 | — |
| return | 29.5125 | 29.3018 | 0.5256 n.s. | 0.3 | — |
| detour_ratio | 0.2215 | 0.2138 | 0.06449 n.s. | 0.9 | — |
| action_smoothness | -0.7829 | -0.7977 | 4.303e-05 *** | 2.4 | ours |
| correction | 0.0220 | 0.0000 | 0.001263 ** | 2.1 | PPO |
| tracking_error_ema | 1.3484 | 1.3496 | 0.3186 n.s. | 0.5 | — |

## 6. 训练期违约（单训练 run，episode 级 Wilson 95% CI）

| 变体 | 碰撞 | episodes | 违约率 | 95% CI |
|---|---|---|---|---|
| Geometric + prop. | 2 | 448 | 0.45% | [0.12%, 1.61%] |
| RP-PSF (ours) | 2 | 426 | 0.47% | [0.13%, 1.70%] |
| A-only | 3 | 433 | 0.69% | [0.24%, 2.02%] |
| Geometric + binary | 4 | 449 | 0.89% | [0.35%, 2.27%] |
| Predictive + binary | 6 | 452 | 1.33% | [0.61%, 2.86%] |
| PPO baseline | 6 | 420 | 1.43% | [0.66%, 3.08%] |

PPO vs ours 两比例检验：z=1.44，**p=0.15，不显著**（事件仅 6 次 vs 2 次）。

## 7. 检验功效

| 场景 | PPO 碰撞率 | 每组 episodes | 80% 功效可检出的最小相对下降 |
|---|---|---|---|
| nominal | 0.78% | 1405 | 89% |
| ood | 1.64% | 1418 | 67% |
| hard | 74.76% | 402 | 12% |

## 8. 图

| 图 | 内容 |
|---|---|
| `fig1_main_results.pdf` | PPO vs ours × 3 场景 × 4 指标 |
| `fig2_softhi_sensitivity.pdf` | h_hi ∈ {0.3,0.6,0.9,1.2} 敏感性（各档独立训练） |
| `fig3_ablation.pdf` | 完整 2×2 + A-only + baseline，6 指标点图 |
| `fig3b_training_violation_rate.pdf` | 训练期违约率 + Wilson CI |
| `fig4_training_violations.pdf` | 训练期累计违约曲线 |
| `fig6_takeover_law.pdf` | 接管律对决：斜坡 vs 阶跃 |
| `fig7_gate_signal.pdf` | 门控信号对决：预测 vs 几何 |

## 9. 指标口径（审稿人会问）

### 9.1 表中数字是「逐 episode 均值」，不要再除以 episodes

`EpisodeStats`（`marinegym/utils/torchrl/env.py:105`）在每条 episode 终止时记一行，
`train.py:216` 对这些行取 `.mean()`。因此 §4 表中每个数字已是 per-episode 平均，
`episodes` 一列只是**样本量**。

### 9.2 episode 为何结束 —— 决定了 return 该不该归一化

`track.py:700`：`terminated |= (distance > reset_thres)`，`reset_thres = 1.5`；
`max_episode_length = 600`，而实测平均长度约 **67 步** ——
**绝大多数 episode 是跟踪失败提前终止，不是跑满时长**。碰撞不终止（`terminate_on_collision=false`）。

因此 **`episode_len` 本身就是任务性能指标**（失控前坚持多久），
`return`（`track.py:713` 逐步累加）已正确地把它计入。
**按 episode 长度归一化 return 是错的** —— 那会奖励「失败得早」：
一个 20 步就跟丢的策略 per-step return 可以很高。

> 实测：若强行按 /step 归一化，`ours vs PPO baseline` 的 return 会从 p=0.53（打平）
> 变成 p=0.048（ours 显著更优）—— 这个「优势」纯粹来自抹掉 ours episode 更短的劣势。**不要这样报。**

### 9.3 各格 episode 长度（必须与 return 一并报告）

| 方法 | episodes | episode_len | return | collision |
|---|---|---|---|---|
| RP-PSF (ours) | 1429 | 67.0 | 29.513 | 0.0169 |
| Predictive + binary | 1373 | 68.3 | 30.751 | 0.0153 |
| Geometric + proportional | 1067 | 66.2 | 28.978 | 0.0204 |
| Geometric + binary | 1087 | 67.2 | 29.352 | 0.0215 |
| A-only (no filter) | 1392 | 67.8 | 30.477 | 0.0166 |
| PPO baseline | 1418 | 67.9 | 29.302 | 0.0164 |

跨格长度极差 2.1 步（3.2%）。但 **Predictive+binary 的 episode 显著长于 ours**
（68.3 vs 67.0，Welch p=0.0009，|d|=1.77）—— 即它跟丢得更晚，这与其 return 更高一致，
**须如实报告**。其余三组的长度差异不显著（p=0.067~0.080）。

### 9.4 collision 的口径局限

`collision` 是整条 episode 上 `min_clearance < 0` 的**逐 episode 取 max**（`track.py:627-628`），
即「本条 episode 是否曾发生侵入」的 0/1 指示。两点局限：

1. **不区分严重程度**：擦一次边与穿过去都记 1；深度由 `min_obstacle_dist` 单独承担。
2. **episode 越长暴露越多**，对「活得久」的方法不利。本数据各格长度差仅 3.2%，
   按每千步归一化后**排序完全不变**，故结论不受影响；但若日后场景改动导致长度差异变大，
   应改用**逐步违约率**或**侵入深度积分** `∫max(0,−clearance)dt`（当前未记录）。

### 9.5 暴露量归一化：collision 的排名会翻转

`collision` 是逐 episode 的 0/1，**未按暴露量归一化**。但各格实际行进距离差异很大
（相对行程 ∝ `detour_ratio × episode_len`）：

| 格子 | collision(原始) | 相对行程 | collision/行程 | 排名 |
|---|---|---|---|---|
| Predictive + binary | 0.0153 | **8.18** | 1.885 | **第1 → 第6** |
| PPO baseline | 0.0164 | 14.52 | 1.122 | 第2 → 第2 |
| A-only | 0.0166 | 13.46 | 1.227 | 第3 → 第4 |
| RP-PSF (ours) | 0.0169 | 14.83 | 1.141 | 第4 → 第3 |
| Geometric + prop. | 0.0204 | **18.99** | 1.096 | **第5 → 第1** |
| Geometric + binary | 0.0215 | 12.87 | 1.693 | 第6 → 第5 |

Predictive+binary 的行程仅为 Geometric+prop 的 **43%**，其「碰撞率最低」主要是**走得少**换来的。

> **但结论不变**：全部成对检验在两种口径下均不显著
> （ours vs P+bin：原始 p=0.734 → 归一化 p=0.128）。**「安全性全部打平」对口径稳健。**

### 9.6 ⚠️ detour_ratio 的语义存疑 —— 使用前必须核实

`track.py:640-644`：`detour_ratio = _flown / _ref_len`，两者均为逐步累加的米数
（载具实际路径长 / 参考点路径长）。代码注释写 **「≥1, →1 best」**，但**实测值全部在 0.12–0.29**。

三条互相印证的证据表明**载具根本跟不上参考轨迹**：

1. `traj_scale_mult = 2.5` 放大 lemniscate 幅值而角速度不变 → 参考点线速度亦为 2.5 倍
2. `tracking_error_ema ≈ 1.35`，而终止阈值 `reset_thres = 1.5` —— 稳定停在阈值的 90%
3. episode 平均仅 **67 步**（上限 600），且 `track.py:700` 的终止条件正是 `distance > reset_thres`
   —— **每条 episode 都以跟踪失败结束**

因此 `detour_ratio` 实际衡量的是「跟丢之前覆盖了参考路径的百分之几」，
**越低 = 走得越少 = 落后越多**，与注释所述方向相反。佐证：
跨格 `corr(detour_ratio, tracking_error) = −0.977`，逐 seed（n=60）为 −0.621。

> **影响**：本文档 §5 与 fig6/fig7 中把 detour 按「低者优」判定，
> 据此得出的最大效应「Predictive+binary 绕行比 ours 少 46%（|d|=13）」**方向可能相反**
> —— 该格覆盖参考路径最少（0.1197），且跟踪误差最高（1.3540）。
>
> **投稿前必须**：(a) 直接记录 `_flown` 与 `_ref_len` 的绝对米数确认量级；
> (b) 若确为跟踪失败工况，则本任务设定（`traj_scale_mult=2.5`）需重新标定到载具能力范围内，
> 否则所有任务性能指标都是在失败工况下测得的。

### 9.7 return 不应按行程或 episode 长度归一化

`return` 是逐步累加的任务奖励（`track.py:713`）。由于 episode 因跟踪失败提前终止，
`episode_len` 本身即「失控前坚持多久」的性能指标，`return` 已正确将其计入。
按长度归一化会**奖励失败得早**（实测：强行归一化会让 ours vs PPO 的 return
从 p=0.53 假性变为 p=0.048）。按行程归一化同理不适用 —— return 不是暴露量计数。

## 10. 结论：能写什么，不能写什么

### 不能写

**「本方法降低碰撞率」——六组对比全部不显著**（按行程归一化后依然全部不显著，见 §9.5）。
**「绕行更短」的声称暂不可用** —— `detour_ratio` 语义存疑（§9.6），方向可能相反。
，包括 vs PPO baseline（0.0169 vs 0.0164，p=0.90）。

原因是场景没有安全压力，不是方法无效：ood 中一条 episode 的最近距离均值 **6.68 m**，
滤波器仅在 1.8% 的步上介入；hard 场景两臂碰撞率均 0.75、最近距离为负，处于饱和区。
按第 7 节功效表，nominal 每组 1405 个 episode 只能检出「下降 89% 以上」的效应。

**准确表述是「当前实验测不出安全性差异」，而非「方法在安全性上无效」。**

### 能写（两条模式在正交轴上均复现）

1. **比例接管 vs 阶跃接管**：在预测门控行与几何门控行**都**成立 ——
   比例接管给出更平滑的执行（|d|=3.9 / 3.0）与更低的跟踪误差（|d|=3.0 / 1.4）；
   阶跃接管给出更短的绕行（|d|=13.0 / 8.9）。安全性两行均打平。

2. **预测门控 vs 几何门控**：在斜坡列与阶跃列**都**成立 ——
   预测门控显著减少绕行（|d|=5.8 / 10.2）。安全性两列均打平。

建议表述：

> 在同等安全水平下，比例接管以更长绕行为代价换取显著更平滑的执行与更低的跟踪误差；
> 预测式门控则在两种接管律下都显著减少绕行。

### 必须主动交代的三点

1. **A-only 对照**：仅保留风险监视器（risk 进观测）而关闭 MPPI 滤波时，安全性与完整方法
   打平，且 return（|d|=1.9）与 detour（|d|=2.6）**更优**。当前场景下组件 B 未测出收益。
2. **PPO 训练目标错配**（`ppo.py:220`）：重要性比率为
   `exp(log π_θ(u_t) − log π_old(u_rl))`，分子分母是不同动作
   （`track.py:519` 把采样动作 u_rl 覆盖为滤波后动作 u_t，而 `sample_log_prob` 仍为
   log π(u_rl)）。判据：epoch 0 时 ratio 应恒为 1，实际在滤波生效的步上 ≠1。
   正确做法：shield 属于环境，学习器存 u_rl。
   **扭曲程度沿消融轴单调**（不滤波=无 → 斜坡 λ 小=轻微 → 阶跃全量替换=严重），
   故「斜坡 vs 阶跃」的差异中部署效应与训练扭曲**尚未分离**。
3. **统计口径**：表中 seeds 为**评测 seed**（同一训练策略的多次独立评测），非训练 seed；
   训练期违约每格仅 1 个训练 run。

### 未产出/已作废

| 项 | 原因 |
|---|---|
| 名义模型单步误差验证 | config key `task.keepout.validate_full` 不存在 |
| 模型失配敏感性（18 次） | key `nom_scale_mass/drag/thrust` 不存在 |
| 计算延迟图 | 测的是 Isaac rollout fps，K=32→256 数值不变，测量无效 |
| hard 场景 | 饱和区（两臂 0.75），无区分度 |
| 内化 C | 不属于本方法，全链路剔除 |
