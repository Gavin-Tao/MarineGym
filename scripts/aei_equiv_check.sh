#!/usr/bin/env bash
# 等价性验证：用本方法的既有 checkpoint(T2)，在【已打补丁】的代码上重跑 ood 10 个 seed，
# 与补丁前的结果逐 seed 对比。一致 → 旧数据在新代码下行为等价，可全部保留。
OUT=/home/jovyan/MarineGym/scripts/outputs_aei
R=/home/jovyan/MarineGym/scripts/run_aei.sh
W=/home/jovyan/MarineGym/scripts/wandb
LOG=$OUT/equiv_progress.log
mkdir -p "$OUT/equiv"
source /home/jovyan/MarineGym/scripts/aei_fixed_params.sh

T2=$W/offline-run-20260711_165556-y7en5cfg/files/checkpoint_final.pt
EVB="task=Track algo=ppo task.drone_model.name=BlueROV task.traj_scale_mult=2.5 \
headless=true enable_livestream=false wandb.mode=offline eval_only=true \
eval_episodes=100 +eval_max_batches=60 task.env.num_envs=64"

echo "[$(date '+%H:%M')] 等价性验证开始（补丁后代码 + 冻结参数集）" >> "$LOG"
for s in 1 2 3 4 5 6 7 8 9 10; do
  f="$OUT/equiv/t2_ood_s${s}.log"
  grep -q 'EVAL-ONLY RESULTS' "$f" 2>/dev/null && continue
  timeout 1500 $R train.py $EVB seed=$s load_ckpt="$T2" \
    task.keepout.dynamic.speed=[2.2,2.8] $OURS > "$f" 2>&1
  echo "[$(date '+%H:%M')] equiv ood s$s rc=$?" >> "$LOG"
done
echo "[$(date '+%H:%M')] ==== 等价性验证跑完 ====" >> "$LOG"
