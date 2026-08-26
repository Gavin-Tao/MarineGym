#!/usr/bin/env bash
# 论文③ 实验链（v6）—— 最终配置，**单次训练 + 多 episode 评测**（用户指定）。
#
# 对照设计：不做参数量对齐。主张 = "Mamba 用更少参数达到相当精度"。
#   Transformer = DTQN 原生默认 d128 × 2层 × 8heads × ff4   436,992  (1.00×)
#   Mamba (ours)                d128 × 1层                   155,392  (0.36×)
#   MLP (单帧下界)              256×3                        142,848  (0.33×)
#
# 训练协议(2026-08-26 用户纠正)：**4096 envs × 20M frames** —— MarineGym 基准
# 原始协议(初始提交的 env_base.yaml)。之前沿用论文①②的 64 envs 是错的：
# 比较策略架构时 PPO batch 太小 ⇒ 策略欠训练 ⇒ 架构差异被压缩。
#
# 统计口径：每格训练一次 × **10 个独立评测 seed**（与论文①② 一致），
#   各 arm 用同一组评测 seed（共同随机数）⇒ 面对相同的初始条件与轨迹参数序列。
#   误差棒 = 跨评测 seed 的 std；显著性用 n=10 的 Welch t 检验。
#   ⚠ 这只刻画**评测噪声**，不含训练 run 间方差（RL 里后者通常更大）。
#   所以"显著"只能说"这两个训练出来的策略不同"，不能说"这个方法平均更好"。
#
# 共享 GPU：其他租户常年占满，空闲显存常低于 1.7 GB，sweep 会等待+重试，不是卡死。
set -o pipefail
cd "$(dirname "$0")" || exit 1
OUT=/home/jovyan/MarineGym-mamba/scripts/outputs_p3/main
S=0

wait_free () { while pgrep -f "bash p3_sweep.sh" >/dev/null; do sleep 60; done; }

stage () {
  local desc=$1; shift
  echo "=================================================="
  echo "[chain] $desc  $(date '+%m-%d %H:%M')"
  echo "=================================================="
  # MarineGym 基准原始协议(初始提交 7bb344c)：4096 envs × 20M frames。
  # frames_per_batch = 4096×64 = 262,144 ⇒ 20M/262k ≈ 76 iters。
  # num_minibatches=64 让单个 minibatch 保持 4096 样本(反向传播显存可控)。
  # LOWMEM=0：PhysX 缓冲削减是按 64 envs 算的，4096 envs 必须用原生默认。
  env "$@" OUT="$OUT" NENV=4096 ITERS=76 NMINI=64 LOWMEM=0 \
      EVAL_SEEDS=10 EVAL_EP=2000 EVAL_MAXB=3 STRIDE=4 MAX_PAR=2 \
      MIN_FREE_MB=9000 RETRIES=10 WAIT_MAX=14400 GPUS="0 1 2 3" \
      bash p3_sweep.sh
  wait_free
}

wait_free

# 1. 【头条表】更少参数 vs 相当精度 —— 最先跑，其余实验都排在后面
stage "头条表 transformer/mamba/mlp" ARMS="transformer mamba mlp" CONDS="drift" SEEDS="$S"

# 2. 补齐其余基线
stage "其余基线 gru/stack" ARMS="gru stack" CONDS="drift" SEEDS="$S"

# 3. Mamba 容量扫描：给出"最小到多少还能匹配基线"的曲线
stage "Mamba 容量扫描 d96/d64" ARMS="mamba_d96 mamba_d64" CONDS="drift" SEEDS="$S"

# 4. 机制验证：drift_accel 与 drift 唯一差别是多给 6 维加速度观测
stage "机制验证 drift_accel" ARMS="mlp mamba transformer" CONDS="drift_accel" SEEDS="$S"

# 5. 反向质疑对照
stage "容量对照" ARMS="transformer_small mamba_big mlp_wide" CONDS="drift" SEEDS="$S"

# 6. 负结果：增益可见与否
stage "负结果 drift_oracle" ARMS="mlp mamba transformer" CONDS="drift_oracle" SEEDS="$S"

echo "[chain] 全部完成 $(date '+%m-%d %H:%M')  共 $(ls -1 "$OUT"/*.json | wc -l) 格"
