#!/usr/bin/env bash
# AEI 消融矩阵重跑 @ P1 (K=128 / N=20 / risk.H=15，全走代码默认值)
# 串行 + 断点续跑；不指定 CUDA_VISIBLE_DEVICES（共享卡上换卡会让 PhysX 起不来）
OUT=/home/jovyan/MarineGym/scripts/outputs_aei
mkdir -p "$OUT"
R=/home/jovyan/MarineGym/scripts/run_aei.sh

COMMON="task=Track algo=ppo task.drone_model.name=BlueROV task.traj_scale_mult=2.5 \
algo.train_every=16 max_iters=120 save_interval=40 headless=true enable_livestream=false \
wandb.mode=offline seed=0"

ABS="task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true \
task.keepout.mppi.enable=true task.keepout.mppi.exact=true task.keepout.risk.enable=true \
task.keepout.mppi.soft_blend=true"

run_one() {
  local label=$1; shift
  if grep -q "checkpoint_final.pt" "$OUT/${label}.log" 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] SKIP  $label" >> "$OUT/progress.log"; return 0
  fi
  echo "[$(date '+%H:%M:%S')] START $label" >> "$OUT/progress.log"
  $R train.py $COMMON "$@" > "$OUT/${label}.log" 2>&1
  local rc=$?
  echo "[$(date '+%H:%M:%S')] DONE  $label rc=$rc" >> "$OUT/progress.log"
}

run_one r1a_pred_binary  $ABS task.keepout.mppi.soft_blend=false
run_one r1b_geom_soft    $ABS task.keepout.risk.enable=false
run_one r1c_geom_binary  $ABS task.keepout.risk.enable=false task.keepout.mppi.soft_blend=false
run_one r1f_a_only       task.keepout.enable=true task.keepout.radius=0.8 \
                         task.keepout.dynamic.enable=true task.keepout.risk.enable=true \
                         task.keepout.mppi.enable=false
run_one r1d_fixed_C      $ABS task.keepout.internalize_weight=0.5
run_one r1e_adaptive_C   $ABS task.keepout.internalize_adaptive=true task.keepout.internalize_gain=5.0
echo "[$(date '+%H:%M:%S')] ALL TRAINING DONE" >> "$OUT/progress.log"
