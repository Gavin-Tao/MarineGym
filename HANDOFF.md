# TTE 实验交接文档

> 最后更新: 2026-07-11
> 方法: **A+B-soft (Risk-Proportional Predictive Safety Filter)** — 策略预测风险 → MPPI 按预测风险比例融合安全动作（软接管）
> 目标期刊: IEEE Transactions on Transportation Electrification (TTE)

---

## 环境配置

### Python 环境
```bash
# 唯一的 Python 环境
/home/jovyan/envs/sim/bin/python

# 关键依赖: isaacsim, torchrl, tensordict, wandb, hydra
```

### 工作目录
```bash
cd /home/jovyan/MarineGym/scripts   # 所有脚本和训练都从这里跑
```

### GPU
```bash
nvidia-smi   # 查看 GPU 状态
```

---

## 已完成的工作

### 1. 方法设计（文件: safe_rl_auv_design.md）

核心思路：
- **A（Risk Monitor）**: 策略网络输出 action + 预测未来 H 步碰撞风险（蒙特卡洛 rollout）
- **B（MPPI Safety Filter）**: 模型预测控制，在线优化安全轨迹
- **soft blend**: 按预测风险比例融合策略动作和 MPPI 动作（不是开关！）
  - `soft_lo=0.0`: 风险低于此值时完全信任策略
  - `soft_hi=0.6`: 风险高于此值时完全信任 MPPI
  - 中间线性插值

### 2. 已有实验数据（results_summary.md）

**S2 动态拦截（核心场景）**:
| 方法 | cum违约↓ | coll↓ | return↑ | 说明 |
|------|:---:|:---:|:---:|------|
| PPO 基线 | 0.18-0.24 | 0.011 | 32.9 | — |
| A+B-soft **★** | **0.062** | **0.000** | 32.7 | **降 66-74%，全场最优** |
| A+B-hard | 0.106 | 0.000 | 30.8 | 开关式，不如软接管 |
| B-only 距离门控 | 0.109 | 0.022 | 31.2 | 说明 A 的预测必要 |
| A-only 风险观测 | 0.171 | 0.011 | 31.8 | 信息有用但不够 |

**变体矩阵 3×3（门控 × 内化）**:
|  | 无C | 固定C | 自适应C |
|---|:---:|:---:|:---:|
| 硬触发 | 0.106 | 0.276-0.343 | 0.289 |
| **软接管** | **0.062 ★** | 0.118 | 0.309 |

结论: A+B-soft, 不加 C（C 始终有害）。

**其他场景**:
- S1 静态障碍: A+B 降 55%
- S4 OOD 部署（未见速度）: B 挂上碰撞砍半 (0.042→0.021)
- CBF shield: 反面教材，+57% 有害

### 3. TTE 训练（T1-T5）✅ 全部完成

训练参数: `max_iters=120`（30,720 帧），`seed=0`，BlueROV，Lemniscate 轨迹
MPPI 默认参数: `num_samples=128, horizon=20, risk.horizon=15`（代码默认值，训练时没有显式传参）

| # | 方法 | soft_hi | collision | power(W) | Checkpoint 路径 |
|---|------|:---:|:---:|:---:|---|
| T1 | PPO 基线 | — | 0.0625 | 1322.66 | `wandb/offline-run-20260711_094733-ck77bi9a/files/checkpoint_final.pt` |
| T2 | A+B-soft | 0.6 | **0** | 1168.48 | `wandb/offline-run-20260711_165556-y7en5cfg/files/checkpoint_final.pt` |
| T3 | A+B-soft | 0.3 | **0** | 1281.64 | `wandb/offline-run-20260711_175104-vhwms10h/files/checkpoint_final.pt` |
| T4 | A+B-soft | 0.9 | 0.0625 | 1260.09 | `wandb/offline-run-20260711_183826-tcnn2ht7/files/checkpoint_final.pt` |
| T5 | A+B-soft | 1.2 | **0** | 1214.75 | `wandb/offline-run-20260711_194323-eb8iodt1/files/checkpoint_final.pt` |

训练日志: `scripts/outputs_tte/`

### 4. 已写的文档
| 文件 | 内容 |
|------|------|
| `experiment_plan_tte.md` | 实验计划（含所有 E0-E6 设计和图表清单） |
| `results_tte.md` | 训练结果（T1-T5），评测部分待更新 |
| `results_summary.md` | 前期 E0 消融实验结果汇总 |
| `safe_rl_auv_design.md` | 方法设计文档 |

---

## ⚠️ 已跑但有问题、需要重跑的

### E1 多 Seed 统计（outputs_e1/ 下有日志，但**结果无效**）

**问题**: `run_e1_multiseed.sh` 里的 T2 checkpoint 路径写错了！
```bash
# ❌ 脚本里用的（错的）:
T2_CKPT=.../offline-run-20260711_104822-0hsw6qx5/files/checkpoint_final.pt

# ✅ 应该用:
T2_CKPT=.../offline-run-20260711_165556-y7en5cfg/files/checkpoint_final.pt
```

`104822` 是一次早期实验，MPPI 参数不同（`num_samples=8, horizon=6`），激活率只有 0.0001，基本等于没开 filter。所以 E1 日志里 PPO 和 ABsoft 结果几乎一样。

**必须用正确的 T2 checkpoint 重新跑 E1！**

### E3b 参数扫描（outputs_eval/ 下有日志，**结果可疑**）

所有配置（risk_thresh=0.1/0.5, soft_lo=0.1/0.2）return 完全一样 (30.0364)，min_obstacle_dist 完全一样 (2.5209)。参数可能没生效。需要验证。

### E6 扰动鲁棒性（outputs_eval/ 下有日志，**只有单 seed**）
需要跑 10 seeds。

---

## ❌ 还没跑的

| 实验 | 内容 | 优先级 |
|:---:|------|:---:|
| **E1 重跑** | 用正确的 T2 checkpoint，10 test seeds × 2 模型 | 🔴 P0 |
| **E2** | 跨轨迹泛化：Circle + Helical 轨迹 eval | 🟠 P1 |
| **E3a** | T3/T4/T5 多 seed eval（soft_hi 曲线 + 误差棒） | 🟠 P1 |
| **E4** | 规则方法对比表（已有数据，整理即可） | 🟡 P2 |
| **E5** | 计算效率计时 | 🟡 P2 |

---

## 如何跑剩余实验

### 前提：所有命令从 scripts/ 目录执行

```bash
cd /home/jovyan/MarineGym/scripts
```

### E1: 多 Seed 统计验证（最重要！）

```bash
# 方法1: 用 eval_tte.py 脚本（推荐，输出 JSON）
/home/jovyan/envs/sim/bin/python eval_tte.py \
    task=Track algo=ppo task.drone_model.name=BlueROV \
    task.traj_scale_mult=2.5 headless=true wandb.mode=offline \
    task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true \
    eval_ckpt=/home/jovyan/MarineGym/scripts/wandb/offline-run-20260711_094733-ck77bi9a/files/checkpoint_final.pt \
    eval_seeds=10 training_seed=0 eval_label=t1_ppo \
    eval_output=/home/jovyan/MarineGym/scripts/outputs_e1_v2/e1_ppo.json

# T2 (A+B-soft, soft_hi=0.6) — 注意用正确的 checkpoint!
/home/jovyan/envs/sim/bin/python eval_tte.py \
    task=Track algo=ppo task.drone_model.name=BlueROV \
    task.traj_scale_mult=2.5 headless=true wandb.mode=offline \
    task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true \
    task.keepout.mppi.enable=true task.keepout.mppi.exact=true \
    task.keepout.risk.enable=true \
    task.keepout.mppi.soft_blend=true task.keepout.mppi.soft_hi=0.6 \
    eval_ckpt=/home/jovyan/MarineGym/scripts/wandb/offline-run-20260711_165556-y7en5cfg/files/checkpoint_final.pt \
    eval_seeds=10 training_seed=0 eval_label=t2_absoft \
    eval_output=/home/jovyan/MarineGym/scripts/outputs_e1_v2/e1_absoft.json
```

**但 `eval_tte.py` 依赖 `scipy`，如果没装就用下面的 train.py 方式。**

```bash
# 方法2: 用 train.py 的 eval_only 模式（如果 eval_tte.py 有问题）
# 这个模式跑 32 个 episode 然后输出 stats.collision

# T1 PPO — 10 seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
    WANDB_MODE=offline /home/jovyan/envs/sim/bin/python train.py \
        task=Track algo=ppo task.drone_model.name=BlueROV \
        task.traj_scale_mult=2.5 headless=true wandb.mode=offline \
        eval_only=true eval_episodes=32 \
        task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true \
        seed=$seed \
        load_ckpt=/home/jovyan/MarineGym/scripts/wandb/offline-run-20260711_094733-ck77bi9a/files/checkpoint_final.pt \
        > outputs_e1_v2/e1_ppo_seed${seed}.log 2>&1
done

# T2 A+B-soft — 10 seeds（注意 checkpoint 路径！）
for seed in 1 2 3 4 5 6 7 8 9 10; do
    WANDB_MODE=offline /home/jovyan/envs/sim/bin/python train.py \
        task=Track algo=ppo task.drone_model.name=BlueROV \
        task.traj_scale_mult=2.5 headless=true wandb.mode=offline \
        eval_only=true eval_episodes=32 \
        task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true \
        task.keepout.mppi.enable=true task.keepout.mppi.exact=true \
        task.keepout.risk.enable=true \
        task.keepout.mppi.soft_blend=true task.keepout.mppi.soft_hi=0.6 \
        seed=$seed \
        load_ckpt=/home/jovyan/MarineGym/scripts/wandb/offline-run-20260711_165556-y7en5cfg/files/checkpoint_final.pt \
        > outputs_e1_v2/e1_absoft_seed${seed}.log 2>&1
done
```

### E2: 跨轨迹泛化

Circle 和 Helical 轨迹通过修改 `task.traj_type` 切换：

```bash
# Circle 轨迹 — 用 T2 checkpoint，10 seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
    WANDB_MODE=offline /home/jovyan/envs/sim/bin/python train.py \
        task=Track algo=ppo task.drone_model.name=BlueROV \
        task.traj_type=circle \
        task.traj_scale_mult=2.5 headless=true wandb.mode=offline \
        eval_only=true eval_episodes=32 \
        task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true \
        task.keepout.mppi.enable=true task.keepout.mppi.exact=true \
        task.keepout.risk.enable=true \
        task.keepout.mppi.soft_blend=true task.keepout.mppi.soft_hi=0.6 \
        seed=$seed \
        load_ckpt=/home/jovyan/MarineGym/scripts/wandb/offline-run-20260711_165556-y7en5cfg/files/checkpoint_final.pt \
        > outputs_e2/e2_circle_seed${seed}.log 2>&1
done

# Helical 轨迹 — 同理，改 task.traj_type=helical
for seed in 1 2 3 4 5 6 7 8 9 10; do
    WANDB_MODE=offline /home/jovyan/envs/sim/bin/python train.py \
        task=Track algo=ppo task.drone_model.name=BlueROV \
        task.traj_type=helical \
        task.traj_scale_mult=2.5 headless=true wandb.mode=offline \
        eval_only=true eval_episodes=32 \
        task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true \
        task.keepout.mppi.enable=true task.keepout.mppi.exact=true \
        task.keepout.risk.enable=true \
        task.keepout.mppi.soft_blend=true task.keepout.mppi.soft_hi=0.6 \
        seed=$seed \
        load_ckpt=/home/jovyan/MarineGym/scripts/wandb/offline-run-20260711_165556-y7en5cfg/files/checkpoint_final.pt \
        > outputs_e2/e2_helical_seed${seed}.log 2>&1
done
```

### E3a: soft_hi 多 Seed Eval（T3/T4/T5）

```bash
# T3 (soft_hi=0.3), 10 seeds
for seed in 1 2 3 4 5 6 7 8 9 10; do
    WANDB_MODE=offline /home/jovyan/envs/sim/bin/python train.py \
        task=Track algo=ppo task.drone_model.name=BlueROV \
        task.traj_scale_mult=2.5 headless=true wandb.mode=offline \
        eval_only=true eval_episodes=32 \
        task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true \
        task.keepout.mppi.enable=true task.keepout.mppi.exact=true \
        task.keepout.risk.enable=true \
        task.keepout.mppi.soft_blend=true task.keepout.mppi.soft_hi=0.3 \
        seed=$seed \
        load_ckpt=/home/jovyan/MarineGym/scripts/wandb/offline-run-20260711_175104-vhwms10h/files/checkpoint_final.pt \
        > outputs_e3a/e3a_hi03_seed${seed}.log 2>&1
done

# T4 (soft_hi=0.9), 10 seeds — checkpoint: 183826-tcnn2ht7
# T5 (soft_hi=1.2), 10 seeds — checkpoint: 194323-eb8iodt1
# 改 soft_hi 和 checkpoint 路径即可
```

### E3b: 参数敏感性（eval-only，不改模型）

```bash
# risk.threshold 扫描 (默认 0.3)
for rt in 0.1 0.5; do
    WANDB_MODE=offline /home/jovyan/envs/sim/bin/python train.py \
        task=Track algo=ppo task.drone_model.name=BlueROV \
        task.traj_scale_mult=2.5 headless=true wandb.mode=offline \
        eval_only=true eval_episodes=32 \
        task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true \
        task.keepout.mppi.enable=true task.keepout.mppi.exact=true \
        task.keepout.risk.enable=true task.keepout.risk.threshold=$rt \
        task.keepout.mppi.soft_blend=true task.keepout.mppi.soft_hi=0.6 \
        seed=1 \
        load_ckpt=/home/jovyan/MarineGym/scripts/wandb/offline-run-20260711_165556-y7en5cfg/files/checkpoint_final.pt \
        > outputs_e3b/e3b_risk_${rt}.log 2>&1
done

# soft_lo 扫描 (默认 0.0)
for sl in 0.1 0.2; do
    WANDB_MODE=offline /home/jovyan/envs/sim/bin/python train.py \
        task=Track algo=ppo task.drone_model.name=BlueROV \
        task.traj_scale_mult=2.5 headless=true wandb.mode=offline \
        eval_only=true eval_episodes=32 \
        task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true \
        task.keepout.mppi.enable=true task.keepout.mppi.exact=true \
        task.keepout.risk.enable=true \
        task.keepout.mppi.soft_blend=true task.keepout.mppi.soft_hi=0.6 \
        task.keepout.mppi.soft_lo=$sl \
        seed=1 \
        load_ckpt=/home/jovyan/MarineGym/scripts/wandb/offline-run-20260711_165556-y7en5cfg/files/checkpoint_final.pt \
        > outputs_e3b/e3b_softlo_${sl}.log 2>&1
done
```

### E6: 扰动鲁棒性（10 seeds）

```bash
# Payload 扰动
for seed in 1 2 3 4 5 6 7 8 9 10; do
    WANDB_MODE=offline /home/jovyan/envs/sim/bin/python train.py \
        task=Track algo=ppo task.drone_model.name=BlueROV \
        task.traj_scale_mult=2.5 headless=true wandb.mode=offline \
        eval_only=true eval_episodes=32 \
        task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true \
        task.keepout.mppi.enable=true task.keepout.mppi.exact=true \
        task.keepout.risk.enable=true \
        task.keepout.mppi.soft_blend=true task.keepout.mppi.soft_hi=0.6 \
        task.disturbances.train.payload.enable_payload=true \
        seed=$seed \
        load_ckpt=/home/jovyan/MarineGym/scripts/wandb/offline-run-20260711_165556-y7en5cfg/files/checkpoint_final.pt \
        > outputs_e6/e6_payload_seed${seed}.log 2>&1
done

# Flow 扰动 — 改 task.disturbances.train.flow.enable_flow=true
```

---

## 如何从日志提取结果

```bash
# 提取所有 E1 日志的 collision 值
grep "stats.collision" outputs_e1_v2/*.log

# 或者用 Python 批量处理：
python3 -c "
import re, os, numpy as np
logs = os.listdir('outputs_e1_v2')
for prefix in ['e1_ppo', 'e1_absoft']:
    colls = []
    for seed in range(1, 11):
        fname = f'outputs_e1_v2/{prefix}_seed{seed}.log'
        with open(fname) as f:
            m = re.search(r'stats\.collision:\s*([\d.]+)', f.read())
            if m: colls.append(float(m.group(1)))
    if colls:
        a = np.array(colls)
        print(f'{prefix}: collision={a.mean():.4f} ± {a.std(ddof=1):.4f} (n={len(a)})')
"
```

---

## 预期工作量

| 实验 | 预计时间 | 备注 |
|:---|:---:|------|
| E1 重跑（2模型×10seeds） | ~1h | 每 seed ~3min |
| E2 跨轨迹（2轨迹×10seeds） | ~1h | |
| E3a soft_hi（3模型×10seeds） | ~1.5h | |
| E3b 参数扫描 | ~0.5h | |
| E6 扰动（2场景×10seeds） | ~1h | |
| E4/E5 整理 | ~0.5h | 不需要 GPU |
| **总计** | **~5.5h** | 可并行跑多个 shell |

---

## 重要注意事项

1. **T2 checkpoint 一定要用 `165556-y7en5cfg`**，不是 `104822-0hsw6qx5`！之前 E1 跑错就是因为这个。

2. **训练时 MPPI 用了默认参数**（`num_samples=128, horizon=20`），eval 时不要覆盖这些参数，保持和训练一致。

3. **所有 eval 命令都要带 `eval_only=true`**，否则会触发训练而不是评测。

4. **评测结果写入 `results_tte.md`**，按 `experiment_plan_tte.md` 里的 Table/Figure 编号整理。

5. **遇到 Isaac Sim 崩溃**（CUDA OOM 等），等几秒重新跑即可。每个 eval 独立，不影响其他 seed。

---

## 联系与参考

- 方法文档: `safe_rl_auv_design.md`
- 实验计划（含完整图表清单）: `experiment_plan_tte.md`
- 已有实验数据: `results_summary.md`
- 当前结果: `results_tte.md`
- 环境: `/home/jovyan/envs/sim/bin/python`
