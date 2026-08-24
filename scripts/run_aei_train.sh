#!/usr/bin/env bash
# AEI: 消融矩阵重跑（P1 = K=128, N=20, risk.H=15，全部走代码默认值，不显式覆盖）
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
  local gpu=$1 label=$2; shift 2
  echo "[$(date '+%H:%M:%S')] START $label (GPU$gpu)" >> "$OUT/progress.log"
  CUDA_VISIBLE_DEVICES=$gpu $R train.py $COMMON "$@" > "$OUT/${label}.log" 2>&1
  echo "[$(date '+%H:%M:%S')] DONE  $label exit=$?" >> "$OUT/progress.log"
}

# ---- 2×2 消融：门控信号 × 接管律（T2 = 预测+比例，已有，不重跑）----
run_one 0 r1a_pred_binary  $ABS task.keepout.mppi.soft_blend=false &
run_one 1 r1b_geom_soft    $ABS task.keepout.risk.enable=false &
wait
run_one 0 r1c_geom_binary  $ABS task.keepout.risk.enable=false task.keepout.mppi.soft_blend=false &
run_one 1 r1f_a_only       task.keepout.enable=true task.keepout.radius=0.8 \
                           task.keepout.dynamic.enable=true task.keepout.risk.enable=true \
                           task.keepout.mppi.enable=false &
wait
# ---- 内化惩罚 C 的负结果 ----
run_one 0 r1d_fixed_C      $ABS task.keepout.internalize_weight=0.5 &
run_one 1 r1e_adaptive_C   $ABS task.keepout.internalize_adaptive=true task.keepout.internalize_gain=5.0 &
wait
echo "[$(date '+%H:%M:%S')] ALL TRAINING DONE" >> "$OUT/progress.log"
