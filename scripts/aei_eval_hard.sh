#!/usr/bin/env bash
# 有区分度的评测场景：缩短拦截提前量，让障碍在 episode 内真正到达
OUT=/home/jovyan/MarineGym/scripts/outputs_aei
R=/home/jovyan/MarineGym/scripts/run_aei.sh
W=/home/jovyan/MarineGym/scripts/wandb
LOG=$OUT/eval_progress.log
EVB="task=Track algo=ppo task.drone_model.name=BlueROV task.traj_scale_mult=2.5 \
headless=true enable_livestream=false wandb.mode=offline eval_only=true \
eval_episodes=40 +eval_max_batches=25 task.env.num_envs=64"
KO="task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true"
ABS="$KO task.keepout.mppi.enable=true task.keepout.mppi.exact=true \
task.keepout.risk.enable=true task.keepout.mppi.soft_blend=true"
HARD="task.keepout.dynamic.intercept_steps=[25,55] task.keepout.dynamic.re_aim_period=50"

go(){ local tag=$1 ck=$2; shift 2
  for s in 1 2 3 4 5; do
    f="$OUT/eval/${tag}_hard_s${s}.log"
    grep -q 'EVAL-ONLY RESULTS' "$f" 2>/dev/null && continue
    timeout 1500 $R train.py $EVB seed=$s load_ckpt="$ck" $HARD "$@" > "$f" 2>&1
    echo "[$(date '+%H:%M')] $tag hard s$s rc=$?" >> "$LOG"
  done; }
ck(){ local run; run=$(grep -oE "wandb/offline-run-[0-9_a-z-]+" "$OUT/$1.log" 2>/dev/null | head -1)
      [ -n "$run" ] && echo "/home/jovyan/MarineGym/scripts/$run/files/checkpoint_final.pt"; }

echo "[$(date '+%H:%M')] ==== HARD 场景 ====" >> "$LOG"
go t2_absoft $W/offline-run-20260711_165556-y7en5cfg/files/checkpoint_final.pt $ABS
go t1_ppo    $W/offline-run-20260711_094733-ck77bi9a/files/checkpoint_final.pt $KO
echo "[$(date '+%H:%M')] ==== HARD 场景完成 ====" >> "$LOG"
