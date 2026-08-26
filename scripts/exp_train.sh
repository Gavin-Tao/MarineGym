#!/usr/bin/env bash
# 论文② 训练。用法：  bash exp_train.sh <cell> <seed>
#
# ── 实验设计：一条策略 + 多个滤波器 ────────────────────────────────────────
# 主消融矩阵里的所有滤波器格子**共用同一条已训练策略**，只在评测期挂不同的滤波器。
# 这样各格之间的差异纯粹来自滤波逻辑，是比"每格各训一条策略"更受控的比较，
# 而且在 GPU 紧张时可行（论文①那种每格各训一次这里跑不起）。
#
# 前提：`safety.risk.in_obs=false`，即 A 只做门控、不进观测 —— 这样所有格子的
# 观测维度完全一致，同一条策略可以直接加载。risk 进观测是**另一条独立的轴**
# （risk-aware 策略），单独训一条策略来比。
#
# 训练配置（只有这三个需要真训练，其余全是 eval-only）：
#   ppo      纯 PPO baseline（走廊 + 阵风，无安全栈）—— 主策略，所有滤波器格子都用它
#   dr       域随机化 PPO（训练时就见强阵风）—— 最危险的审稿意见对照
#   riskobs  risk 进观测的 PPO —— risk-aware 轴
set -o pipefail
CELL=${1:?用法: exp_train.sh <ppo|dr|riskobs> <seed>}
SEED=${2:-0}
OUT=/home/jovyan/MarineGym-flow/scripts/outputs_flow/train
mkdir -p "$OUT"
R=/home/jovyan/MarineGym-flow/scripts/run_flow.sh

# 冻结的共用参数（改这里 = 改全部格子，别在单个格子里散着改）
COMMON="task=Track algo=ppo task.drone_model.name=BlueROV \
headless=true enable_livestream=false wandb.mode=offline \
task.env.num_envs=${NUM_ENVS:-256} algo.train_every=${TRAIN_EVERY:-32} \
max_iters=${ITERS:-300} save_interval=25 seed=$SEED +train_log=/home/jovyan/MarineGym-flow/scripts/outputs_flow/train/curve_${CELL}_s${SEED}.csv \
task.corridor.enable=true task.gust.enable=true"

case "$CELL" in
  ppo)     EXTRA="task.safety.risk.enable=false task.safety.mppi.enable=false" ;;
  # 域随机化：训练时阵风速度覆盖到评测的 OOD 区间，并把上升沿也随机得更陡
  dr)      EXTRA="task.safety.risk.enable=false task.safety.mppi.enable=false \
                  task.gust.speed=[0.3,2.0] task.gust.ramp=[0.1,0.6]" ;;
  riskobs) EXTRA="task.safety.risk.enable=true task.safety.risk.in_obs=true \
                  task.safety.mppi.enable=false" ;;
  *) echo "未知 cell: $CELL"; exit 2 ;;
esac

LOG="$OUT/${CELL}_s${SEED}.log"
echo "[$(date '+%H:%M:%S')] 训练 $CELL seed=$SEED → $LOG"
# 显式指定 GPU 时用 export，**不能**写成 `${GPU:+CUDA_VISIBLE_DEVICES=$GPU} cmd`：
# 变量赋值前缀必须是字面量，由参数展开产生的会被 bash 当成命令名执行，报
# "CUDA_VISIBLE_DEVICES=2: command not found" 并且**返回 0**，整格静默跳过。
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"
timeout ${TIMEOUT:-36000} bash "$R" train.py $COMMON $EXTRA > "$LOG" 2>&1
echo "[$(date '+%H:%M:%S')] $CELL s$SEED rc=$?"
grep -E "^\s+(wall_violation|tracking_error_ema|episode_len|return):" "$LOG" | tail -8
