#!/usr/bin/env bash
# 几何门控两格重跑：risk.enable=true + risk.gate=false
#   → A 照常前滚并把 risk 标量喂进观测(obs 45 维，与本方法一致)，但门控退回当前间隙。
# 这样「门控信号」这个轴才是唯一变量。需先落地 risk.gate 补丁。
OUT=/home/jovyan/MarineGym/scripts/outputs_aei
R=/home/jovyan/MarineGym/scripts/run_aei.sh
LOG=$OUT/geom_progress.log
source /home/jovyan/MarineGym/scripts/aei_fixed_params.sh

GEOM="task.keepout.risk.gate=false"          # 唯一改动：门控信号
train_one(){ local lbl=$1; shift
  grep -q checkpoint_final.pt "$OUT/${lbl}.log" 2>/dev/null && { echo "[$(date '+%H:%M')] SKIP $lbl" >> "$LOG"; return; }
  echo "[$(date '+%H:%M')] 训练 START $lbl" >> "$LOG"
  $R train.py "$@" > "$OUT/${lbl}.log" 2>&1
  echo "[$(date '+%H:%M')] 训练 DONE  $lbl rc=$?" >> "$LOG"; }

train_one r1b_geom_soft_v2   $FIXED_ALL $GEOM task.keepout.mppi.soft_blend=true
train_one r1c_geom_binary_v2 $FIXED_ALL $GEOM task.keepout.mppi.soft_blend=false

EVB="task=Track algo=ppo task.drone_model.name=BlueROV task.traj_scale_mult=2.5 \
headless=true enable_livestream=false wandb.mode=offline eval_only=true \
eval_episodes=100 +eval_max_batches=60 task.env.num_envs=64"
eval_one(){ local lbl=$1; shift
  local run ck
  run=$(grep -oE "wandb/offline-run-[0-9_a-z-]+" "$OUT/${lbl}.log" 2>/dev/null | head -1)
  ck="/home/jovyan/MarineGym/scripts/${run}/files/checkpoint_final.pt"
  [ -f "$ck" ] || { echo "[$(date '+%H:%M')] MISS ckpt $lbl" >> "$LOG"; return; }
  for s in 1 2 3 4 5 6 7 8 9 10; do
    local f="$OUT/eval/${lbl}_ood_s${s}.log"
    grep -q 'EVAL-ONLY RESULTS' "$f" 2>/dev/null && continue
    timeout 1500 $R train.py $EVB seed=$s load_ckpt="$ck" \
      task.keepout.dynamic.speed=[2.2,2.8] "$@" > "$f" 2>&1
    echo "[$(date '+%H:%M')] $lbl ood s$s rc=$?" >> "$LOG"
  done; }

eval_one r1b_geom_soft_v2   $FIXED_ALL $GEOM task.keepout.mppi.soft_blend=true
eval_one r1c_geom_binary_v2 $FIXED_ALL $GEOM task.keepout.mppi.soft_blend=false
echo "[$(date '+%H:%M')] ==== 几何两格完成 ====" >> "$LOG"
