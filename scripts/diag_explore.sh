#!/usr/bin/env bash
# 判决：评测取动作的方式（MODE=分布 mode）是否就是训练/评测差 7 倍的原因。
#
#   MODE   : 确定性动作（评测默认）
#   RANDOM : 采样动作（与训练时一致）
#
# 若 RANDOM 明显好于 MODE 且接近训练时报出的 0.158 → 问题在评测取动作的方式，
#   **训练本身没问题、策略是好的、不用重训**，只需改评测。
# 若两者相当（都在 1.2 附近）→ 训练时报出的指标本身乐观，策略实际就这个水平，
#   **需要重训或重新设计**。
set -o pipefail
CKPT=${CKPT:?需要 CKPT}
OUT=/home/jovyan/MarineGym-flow/scripts/outputs_flow/diag
mkdir -p "$OUT"
R=/home/jovyan/MarineGym-flow/scripts/run_flow.sh
BASE="task=Track algo=ppo task.drone_model.name=BlueROV \
headless=true enable_livestream=false wandb.mode=offline \
eval_only=true eval_episodes=500 +eval_max_batches=12 \
task.env.num_envs=${NUM_ENVS:-32} seed=0 load_ckpt=$CKPT \
task.corridor.enable=true task.gust.enable=true \
task.safety.risk.enable=false task.safety.mppi.enable=false"

for ex in mode random; do
  LOG="$OUT/explore_${ex}.log"
  grep -q 'EVAL-ONLY RESULTS' "$LOG" 2>/dev/null && { echo "[skip] $ex"; continue; }
  echo "[$(date '+%H:%M:%S')] eval_explore=$ex"
  timeout 3600 bash "$R" train.py $BASE +eval_explore=$ex > "$LOG" 2>&1
  echo "  rc=$?"
  grep -E "episode_len|tracking_error_ema|min_wall_dist|wall_violation|episodes=" "$LOG" | tail -5
done

echo
echo "训练时报出（iter 491，同配置）: ep_len 594 | tracking 0.158 | min_wall 0.757 | viol 0.015"
echo "随机策略对照            : ep_len 541 | tracking 1.970 | min_wall 0.251 | viol 0.256"
