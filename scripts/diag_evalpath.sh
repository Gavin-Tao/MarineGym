#!/usr/bin/env bash
# 训练/评测差异定位：两个对照，都是 eval_only（纯评测，策略不更新）。
#
#   A) 用**训练时的收集器设置**（num_envs=256, train_every=32）评测
#      → 若回到 ~0.16，说明差异来自 num_envs / train_every，而不是策略
#   B) **不加载 checkpoint**（随机策略）评测
#      → 给出"策略没生效"时的数值下界，用来判断 A/评测里的 checkpoint 是否真的生效
set -o pipefail
CKPT=${CKPT:?需要 CKPT}
OUT=/home/jovyan/MarineGym-flow/scripts/outputs_flow/diag
mkdir -p "$OUT"
R=/home/jovyan/MarineGym-flow/scripts/run_flow.sh

BASE="task=Track algo=ppo task.drone_model.name=BlueROV \
headless=true enable_livestream=false wandb.mode=offline \
eval_only=true eval_episodes=500 seed=0 \
task.corridor.enable=true task.gust.enable=true \
task.safety.risk.enable=false task.safety.mppi.enable=false"

echo "== A) 训练时的收集器设置 + checkpoint =="
timeout 3600 bash "$R" train.py $BASE \
  task.env.num_envs=256 algo.train_every=32 +eval_max_batches=24 \
  load_ckpt="$CKPT" > "$OUT/A_traincollector.log" 2>&1
echo "rc=$?"
grep -E "episode_len|tracking_error_ema|min_wall_dist|wall_violation|episodes=" "$OUT/A_traincollector.log" | tail -5

echo
echo "== B) 随机策略（不加载 checkpoint）=="
timeout 3600 bash "$R" train.py $BASE \
  task.env.num_envs=128 +eval_max_batches=12 \
  > "$OUT/B_random.log" 2>&1
echo "rc=$?"
grep -E "episode_len|tracking_error_ema|min_wall_dist|wall_violation|episodes=" "$OUT/B_random.log" | tail -5
