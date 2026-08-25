#!/usr/bin/env bash
# 给争议最大的两格补训练 seed：预测+比例(T2) vs 预测+二值(r1a)
OUT=/home/jovyan/MarineGym/scripts/outputs_aei
R=/home/jovyan/MarineGym/scripts/run_aei.sh
COMMON="task=Track algo=ppo task.drone_model.name=BlueROV task.traj_scale_mult=2.5 \
algo.train_every=16 max_iters=120 save_interval=40 headless=true enable_livestream=false wandb.mode=offline"
ABS="task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true \
task.keepout.mppi.enable=true task.keepout.mppi.exact=true task.keepout.risk.enable=true \
task.keepout.mppi.soft_blend=true"
# 等 hard 场景评测结束再开始，避免三进程 OOM
while ! grep -q 'HARD 场景完成' "$OUT/eval_progress.log" 2>/dev/null; do sleep 120; done
one(){ local lbl=$1; shift
  grep -q checkpoint_final.pt "$OUT/${lbl}.log" 2>/dev/null && return 0
  echo "[$(date '+%H:%M')] START $lbl" >> "$OUT/seeds_progress.log"
  $R train.py $COMMON "$@" > "$OUT/${lbl}.log" 2>&1
  echo "[$(date '+%H:%M')] DONE  $lbl rc=$?" >> "$OUT/seeds_progress.log"; }
one s1_t2_soft   seed=1 $ABS
one s1_r1a_bin   seed=1 $ABS task.keepout.mppi.soft_blend=false
one s2_t2_soft   seed=2 $ABS
one s2_r1a_bin   seed=2 $ABS task.keepout.mppi.soft_blend=false
echo "[$(date '+%H:%M')] SEEDS DONE" >> "$OUT/seeds_progress.log"
