# TTE 实验结果 (2026-07-11)

> 参数: `max_iters=120` (30,720 帧), `seed=0`, `BlueROV`
> MPPI 默认参数: `num_samples=128, horizon=20, risk.horizon=15`

---

## Step 0: 训练结果 (T1-T5)

| # | 方法 | soft_hi | collision ↓ | avg_power_W ↓ | rollout_fps | 状态 |
|---|------|---------|:-----------:|:------------:|:-----------:|:----:|
| T1 | PPO 基线 | — | 0.0625 | 1322.66 | 57.94 | ✅ |
| T2 | A+B-soft | 0.6 | **0** | 1168.48 | 8.20 | ✅ |
| T3 | A+B-soft | 0.3 | **0** | 1281.64 | 14.53 | ✅ |
| T4 | A+B-soft | 0.9 | 0.0625 | 1260.09 | 8.15 | ✅ |
| T5 | A+B-soft | 1.2 | **0** | 1214.75 | 8.05 | ✅ |

### 关键发现
- **T2 (soft_hi=0.6, 默认)**: 最优 — collision=0, power 最低 (1168W, 比 PPO 低 12%)
- **T3 (soft_hi=0.3, 保守)**: collision=0, 但 power 偏高 (1281W) — 过度过滤
- **T4 (soft_hi=0.9, 激进)**: collision=0.0625 — 等于 PPO, 过滤不足
- **T5 (soft_hi=1.2, 极激进)**: collision=0, power=1215W — 单 seed 波动

### Checkpoint 路径

| # | checkpoint |
|---|-----------|
| T1 | `wandb/offline-run-20260711_094733-ck77bi9a/files/checkpoint_final.pt` |
| T2 | `wandb/offline-run-20260711_165556-y7en5cfg/files/checkpoint_final.pt` |
| T3 | `wandb/offline-run-20260711_175104-vhwms10h/files/checkpoint_final.pt` |
| T4 | `wandb/offline-run-20260711_183826-tcnn2ht7/files/checkpoint_final.pt` |
| T5 | `wandb/offline-run-20260711_194323-eb8iodt1/files/checkpoint_final.pt` |

---

## E1: 多 Seed 统计验证 (⏳ 待做)

用默认参数 checkpoint 跑 10 test seeds × 2 模型 (T1 PPO vs T2 A+B-soft)。

---

## E2: 跨轨迹泛化 (⏳ 待做)

## E3: 超参数敏感性 (⏳ 待做)

## E4: 规则方法对比 (⏳ 待做)

## E5: 计算效率 (⏳ 待做)

## E6: 扰动鲁棒性 (⏳ 待做)

---

## 实验日志位置

- 训练日志: `/home/jovyan/MarineGym/scripts/outputs_tte/`
- Checkpoint: `/home/jovyan/MarineGym/scripts/wandb/`
- 运行脚本: `/home/jovyan/MarineGym/scripts/run_tte_t2t5_default.sh`
