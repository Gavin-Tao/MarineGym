# IEEE TTE 投稿最小实验计划 (v3)

> 方法: **A+B-soft (Risk-Proportional Predictive Safety Filter)**
> 核心: 策略预测风险 → 按预测风险比例融合 MPPI 安全动作 (软接管)
> 策略: **只训必须训的, 其余全部 eval-only**

---

## 已有实验 (可直接用,不需重跑)

| 编号 | 实验 | 核心结论 |
|---|---|---|
| E0.1 | PPO 基线 (多训练长度/seed) | cum=0.18–0.24, coll≈0.01 |
| E0.2 | S2 动态拦截: A+B-soft vs A+B-hard | soft 0.062 最优, hard 0.106 |
| E0.3 | S2 消融: B-only 距离门控 | 0.109 → A 的预测必要 |
| E0.4 | S4 OOD 部署 (未见 2.2–2.8 m/s) | B 挂上碰撞砍半 |
| E0.5 | S1 静态障碍 | A+B 降 55% |
| E0.6 | CBF shield vs PPO | CBF +57% 有害 → 论证预测式 |
| E0.7 | PPO + Lagrangian (固定/自适应) | 已有数据, 不如 A+B-soft |

---

## 需要补的实验

### E1 — 多 Evaluation Seed 统计验证 ⭐ 最高优先级

**方式**: 训练 **1 次** (1 个 training seed), 用 **10 个不同于训练的 test seed** 做 evaluation rollout, 得到 mean ± std + 显著性检验。

| 方法 | 训练 | 评测 |
|---|---|---|
| PPO 基线 | 1 次 (seed=0, 10M frames) | 10 test seeds rollout, 每 seed 跑满 episode |
| A+B-soft ★ | 1 次 (seed=0, 10M frames) | 同上 |

**每个 test seed 的评测**: 固定 checkpoint, 设 `env.set_seed(test_seed)`, rollout 到 done, 记录 coll / return / minDist / trkEMA / 激活率 / detour_ratio / powerW。

**产出**:
- **Table 1**: 各指标 mean ± std (10 seeds), PPO vs A+B-soft, 带 Welch's t-test p-value
- **Figure 1**: 柱状图 + 误差棒 (coll, return, minDist)

---

### E2 — 跨轨迹零样本泛化 (eval-only)

**方式**: 加载 E1 训练的 A+B-soft checkpoint, 在新轨迹上 eval (不改训练, 不改模型权重, 不改 MPPI 参数), 每个轨迹 10 test seeds。

| 轨迹 | 来源 | 评测 |
|---|---|---|
| Lemniscate (训练轨迹) | 参照 | 10 test seeds |
| Circle | `circle()` 已有 | 10 test seeds |
| Helical (3D 螺旋) | `helical()` 已有 | 10 test seeds |

**产出**: **Table 2** — 跨轨迹 coll / return / minDist / 激活率; 证明泛化性

---

### E3 — 超参数敏感性

**说明**: A+B-soft 中 B 的参数在训练时就参与交互——策略学的是适应特定滤波器参数的行为。因此核心参数 `soft_hi` 必须每个值单独训练。其余参数做 eval-only 并标注为 "部署 mismatch 分析"。

#### E3a — `soft_hi` 训练级消融 (核心)

| `soft_hi` | 说明 | 方式 |
|---|---|---|
| 0.3 | 保守: 更早接管 | **训练** (10M frames, seed=0) |
| **0.6** | 默认 ✅ (E1 已有) | 已有 |
| 0.9 | 激进: 更晚接管 | **训练** (10M frames, seed=0) |
| 1.2 | 几乎不管 (≈ A-only) | **训练** (10M frames, seed=0) |

**额外训练**: 3 次 × 15min = 45min

每个训练完成后, 用 10 test seeds eval。

**产出**: **Figure 2a** — `soft_hi` vs coll / return / 激活率, 10 seeds 误差棒

#### E3b — `risk.threshold` + `soft_lo` 部署 mismatch 分析 (eval-only)

| 参数 | 默认值 | 扫描值 | 方式 |
|---|---|---|---|
| `risk.threshold` | 0.3 | {0.1, 0.3, 0.5} | eval-only (加载 E1 默认模型, 改参数 eval) |
| `soft_lo` | 0.0 | {0.0, 0.1, 0.2} | eval-only (同上) |

**产出**: **Figure 2b** — 两个参数 vs coll/return; 标注为 "deployment mismatch, not retrained"

---

### E4 — 规则方法对比 (已有数据, 整理即可)

**方式**: 直接用已有数据, 不跑新实验。

| 方法 | 类型 | cum违约 | coll | 来源 |
|---|---|---|---|---|
| CBF shield | 一步 CBF | 3.30 (+57%) | 0.167 | E0.6 |
| B-only (距离门控 MPPI) | 反应式 | 0.23 | 0.025 | E0.3 |
| **A+B-soft** ★ | 预测式软接管 | **0.062** | **0.000** | E0.2 |

**产出**: **Table 3** — 规则/反应式 vs 预测式对比; 论证 "必须预测、必须软接管"

---

### E5 — 计算效率分析 (eval-only)

**方式**: 在 E1 的 A+B-soft eval rollout 中插桩计时。

| 测量项 | 方法 |
|---|---|
| 纯 PPO 单步延迟 (ms) | PPO eval 时计时, 16 envs |
| A+B-soft 单步延迟 (ms) | MPPI + Risk Monitor 总量, 16 envs |
| A+B-soft overhead (%) | (A+B-soft 延迟 − PPO 延迟) / PPO 延迟 |
| MPPI filter 单独延迟 | 按 num_samples={64, 128, 256} 扫一次 |
| Risk Monitor 单独延迟 | 按 horizon={10, 15, 20} 扫一次 |

**产出**: **Table 4** — 延迟表; 证明 overhead 可接受

---

### E6 — 扰动鲁棒性 (eval-only, 最后做)

**方式**: 加载 E1 训练的 A+B-soft checkpoint (在 nominal 条件下训练的模型), **不重训**, 直接在扰动场景 eval。同时用 E1 的 PPO checkpoint 做对照。每个场景 10 test seeds。

| 场景 | 扰动 | 评测 (均用 nominal 训练的模型) |
|---|---|---|
| 动态拦截 + payload | 载荷 ∈ [0.1,0.5]×本体, 随机深度 | PPO vs A+B-soft, 各 10 seeds |
| 动态拦截 + flow | 水流 v∈[0,0.5]m/s, 随机方向 | PPO vs A+B-soft, 各 10 seeds |

**产出**: **Table 5** — 扰动下 coll/return 对比; 证明零样本鲁棒性

---

## 训练清单汇总

| # | 方法 | 参数 | 帧数 | 时间 |
|---|---|---|---|---|
| T1 | PPO 基线 | — | 10M | ~15min |
| T2 | A+B-soft (默认) | `soft_hi=0.6` | 10M | ~15min |
| T3 | A+B-soft | `soft_hi=0.3` | 10M | ~15min |
| T4 | A+B-soft | `soft_hi=0.9` | 10M | ~15min |
| T5 | A+B-soft | `soft_hi=1.2` | 10M | ~15min |

**总训练时间: ~1.25h**

---

## 执行顺序

```
Step 0: 训练 T1–T5 (5 次训练, 可并行 2-3 个)              ~1.25h
        ↓ 得到 5 个 checkpoint

Step 1: E1 多 seed eval (10 seeds × 2 模型)                ~1h
        ↓ 得到 Table 1 + Figure 1

Step 2: E3a soft_hi 曲线的 eval 部分 (3 模型 × 10 seeds)  ~0.5h
        E3b risk.threshold + soft_lo (eval-only)           ~0.3h
        E2 跨轨迹泛化 (2 轨迹 × 10 seeds)                  ~0.3h
        E5 计算效率 (插桩计时)                             ~0.2h
        ↓ 全部 eval, 可并行

Step 3: E4 规则方法对比                                    ~0h (整理)

Step 4: E6 扰动鲁棒性 (2 场景 × 2 模型 × 10 seeds)        ~0.5h
```

**总计: ~1.25h 训练 + ~3h eval, 一天半内完成**

---

## 完成后论文图表清单

| # | 类型 | 内容 |
|---|---|---|
| Table 1 | 主结果 | E1: 多 seed 统计验证 (mean±std + p-value) |
| Table 2 | 泛化 | E2: 跨轨迹零样本 coll/return |
| Table 3 | 对比 | E4: 规则方法 vs 预测方法 |
| Table 4 | 效率 | E5: 计算延迟分析 |
| Table 5 | 鲁棒性 | E6: 扰动下零样本性能 |
| Figure 1 | 主图 | E1: PPO vs A+B-soft 柱状图 (coll, return, minDist) + 误差棒 |
| Figure 2a | 敏感度 | E3a: `soft_hi` vs coll/return/激活率 (每个值单独训练) |
| Figure 2b | 敏感度 | E3b: `risk.threshold` + `soft_lo` vs coll/return (eval-only, 标注 mismatch) |

---

## 已有消融矩阵 (E0 直接用于论文)

| 方法 | cum违约↓ | coll↓ | minDist↑ | return↑ | 激活率 |
|---|---|---|---|---|---|
| PPO 基线 | 0.18–0.24 | 0.011 | 1.87 | 32.9 | — |
| B-only (距离门控) | 0.23 | 0.025 | 1.92 | 29.6 | — |
| A-only (风险观测) | 0.17 | 0.011 | 1.92 | 31.8 | — |
| A+B-hard (风险触发) | **0.106** | **0.000** | 2.07 | 30.8 | 0.12 |
| **A+B-soft (风险比例)** ★ | **0.062** | **0.000** | **2.36** | **32.7** | 0.10 |

> 消融结论: A 预测 > B 距离门控; soft 比例 > hard 开关
