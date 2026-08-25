#!/usr/bin/env bash
# AEI 评测总控：P1 配置（绝不覆盖 num_samples/horizon/risk.horizon）
# 容错：单个 run 超时或失败即跳过，不阻塞队列；断点续跑
OUT=/home/jovyan/MarineGym/scripts/outputs_aei
EV=$OUT/eval; mkdir -p "$EV" "$OUT/data"
R=/home/jovyan/MarineGym/scripts/run_aei.sh
W=/home/jovyan/MarineGym/scripts/wandb
LOG=$OUT/eval_progress.log
SEEDS="1 2 3 4 5 6 7 8 9 10"
TMO=1500

BASE="task=Track algo=ppo task.drone_model.name=BlueROV task.traj_scale_mult=2.5 \
headless=true enable_livestream=false wandb.mode=offline eval_only=true \
eval_episodes=100 +eval_max_batches=60 task.env.num_envs=64"
KO="task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true"
ABS="$KO task.keepout.mppi.enable=true task.keepout.mppi.exact=true \
task.keepout.risk.enable=true task.keepout.mppi.soft_blend=true"

# 找某个训练 label 最新产生的 checkpoint
ck_of_label() {  # $1=label -> 打印 ckpt 路径
  local run
  run=$(grep -oE "wandb/offline-run-[0-9_a-z-]+" "$OUT/$1.log" 2>/dev/null | head -1)
  [ -n "$run" ] && echo "/home/jovyan/MarineGym/scripts/$run/files/checkpoint_final.pt"
}

do_eval() {  # $1=tag $2=ckpt $3=scene  剩下=模型专属 flags
  local tag=$1 ck=$2 scene=$3; shift 3
  [ -f "$ck" ] || { echo "[$(date '+%H:%M')] MISS ckpt $tag" >> "$LOG"; return; }
  local sc=""; [ "$scene" = ood ] && sc="task.keepout.dynamic.speed=[2.2,2.8]"
  for s in $SEEDS; do
    local f="$EV/${tag}_${scene}_s${s}.log"
    grep -q 'EVAL-ONLY RESULTS' "$f" 2>/dev/null && continue
    timeout $TMO $R train.py $BASE seed=$s load_ckpt="$ck" $sc "$@" > "$f" 2>&1
    echo "[$(date '+%H:%M')] $tag $scene s$s rc=$?" >> "$LOG"
  done
}

T1=$W/offline-run-20260711_094733-ck77bi9a/files/checkpoint_final.pt
T2=$W/offline-run-20260711_165556-y7en5cfg/files/checkpoint_final.pt
T3=$W/offline-run-20260711_175104-vhwms10h/files/checkpoint_final.pt
T4=$W/offline-run-20260711_183826-tcnn2ht7/files/checkpoint_final.pt
T5=$W/offline-run-20260711_194323-eb8iodt1/files/checkpoint_final.pt

echo "[$(date '+%H:%M')] ==== PHASE 1: 主结果 C1 (T1 vs T2) ====" >> "$LOG"
do_eval t1_ppo      "$T1" ood     $KO
do_eval t2_absoft   "$T2" ood     $ABS
do_eval t1_ppo      "$T1" nominal $KO
do_eval t2_absoft   "$T2" nominal $ABS

echo "[$(date '+%H:%M')] ==== PHASE 2: soft_hi 敏感性 C6 ====" >> "$LOG"
do_eval t3_hi0.3    "$T3" ood $ABS task.keepout.mppi.soft_hi=0.3
do_eval t4_hi0.9    "$T4" ood $ABS task.keepout.mppi.soft_hi=0.9
do_eval t5_hi1.2    "$T5" ood $ABS task.keepout.mppi.soft_hi=1.2

echo "[$(date '+%H:%M')] ==== PHASE 3: 等训练结束后评测消融 ====" >> "$LOG"
while ! grep -q "ALL TRAINING DONE" "$OUT/progress.log" 2>/dev/null; do sleep 120; done

do_eval r1a_pred_binary "$(ck_of_label r1a_pred_binary)" ood $ABS task.keepout.mppi.soft_blend=false
do_eval r1b_geom_soft   "$(ck_of_label r1b_geom_soft)"   ood $ABS task.keepout.risk.enable=false
do_eval r1c_geom_binary "$(ck_of_label r1c_geom_binary)" ood $ABS task.keepout.risk.enable=false task.keepout.mppi.soft_blend=false
do_eval r1f_a_only      "$(ck_of_label r1f_a_only)"      ood $KO task.keepout.risk.enable=true task.keepout.mppi.enable=false
do_eval r1d_fixed_C     "$(ck_of_label r1d_fixed_C)"     ood $ABS task.keepout.internalize_weight=0.5
do_eval r1e_adaptive_C  "$(ck_of_label r1e_adaptive_C)"  ood $ABS task.keepout.internalize_adaptive=true task.keepout.internalize_gain=5.0

echo "[$(date '+%H:%M')] ==== ALL EVAL DONE ====" >> "$LOG"
