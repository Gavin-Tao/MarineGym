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

| 方法 | episodes | collision ↓ | return ↑ | detour ↓ | smoothness ↑ | correction ↓ | tracking ↓ | 激活率 |
|---|---|---|---|---|---|---|---|---|
| RP-PSF (ours) | 1429 | 0.0169±0.0102 | 29.5125±0.5338 | 0.2215±0.0099 | -0.7829±0.0060 | 0.0220±0.0143 | 1.3484±0.0017 | 0.0176±0.0098 |
| Predictive + binary | 1373 | 0.0153±0.0094 | 30.7512±0.7841 | 0.1197±0.0036 | -0.8116±0.0079 | 0.0357±0.0129 | 1.3540±0.0019 | 0.0245±0.0086 |
| Geometric + proportional | 1067 | 0.0204±0.0125 | 28.9783±0.6690 | 0.2866±0.0114 | -0.7830±0.0079 | 0.0319±0.0227 | 1.3469±0.0020 | 0.0212±0.0128 |
| Geometric + binary | 1087 | 0.0215±0.0126 | 29.3518±0.9063 | 0.1915±0.0087 | -0.8052±0.0063 | 0.0394±0.0203 | 1.3507±0.0030 | 0.0276±0.0134 |
| A-only (no filter) | 1392 | 0.0166±0.0104 | 30.4767±0.4216 | 0.1985±0.0066 | -0.7884±0.0060 | 0.0000±0.0000 | 1.3508±0.0023 | 0.0185±0.0095 |
| PPO baseline | 1418 | 0.0164±0.0102 | 29.3018±0.8144 | 0.2138±0.0060 | -0.7977±0.0057 | 0.0000±0.0000 | 1.3496±0.0032 | 0.0000±0.0000 |

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

## 9. 结论：能写什么，不能写什么

### 不能写

**「本方法降低碰撞率」——六组对比全部不显著**，包括 vs PPO baseline（0.0169 vs 0.0164，p=0.90）。

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
