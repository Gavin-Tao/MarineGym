#!/usr/bin/env bash
# 把 Predictive+binary 的触发边界从 risk.threshold=0.3 对齐到 0.6，
# 与 RP-PSF 的 mppi.soft_hi=0.6 一致 —— 两格从此只差接管律（斜坡 vs 阶跃）。
# 新 label 用 _t06 后缀，不覆盖原始 0.3 的日志，保留可追溯性。
OUT=/home/jovyan/MarineGym/scripts/outputs_aei
R=/home/jovyan/MarineGym/scripts/run_aei.sh
W=/home/jovyan/MarineGym/scripts/wandb
LOG=$OUT/matched_progress.log
LBL=r1a_pred_binary_t06

COMMON="task=Track algo=ppo task.drone_model.name=BlueROV task.traj_scale_mult=2.5 \
algo.train_every=16 max_iters=120 save_interval=40 headless=true enable_livestream=false \
wandb.mode=offline seed=0"
ABS="task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true \
task.keepout.mppi.enable=true task.keepout.mppi.exact=true task.keepout.risk.enable=true \
task.keepout.mppi.soft_blend=true"
# 关键：soft_blend=false 走阶跃接管；risk.threshold=0.6 让触发边界与 soft_hi 对齐
MATCH="task.keepout.mppi.soft_blend=false task.keepout.risk.threshold=0.6"

say(){ echo "[$(date '+%H:%M')] $*" >> "$LOG"; }

# ---- 训练（OOM 重试 3 次）----
if grep -q checkpoint_final.pt "$OUT/${LBL}.log" 2>/dev/null; then
  say "SKIP 训练（已完成）"
else
  for try in 1 2 3; do
    say "训练 START (尝试 $try)"
    $R train.py $COMMON $ABS $MATCH > "$OUT/${LBL}.log" 2>&1
    rc=$?
    if grep -q checkpoint_final.pt "$OUT/${LBL}.log" 2>/dev/null; then say "训练 DONE rc=$rc"; break; fi
    say "训练失败 rc=$rc（可能 OOM），10 分钟后重试"; sleep 600
  done
fi

CK=$(grep -oE "wandb/offline-run-[0-9_a-z-]+" "$OUT/${LBL}.log" 2>/dev/null | head -1)
CK="/home/jovyan/MarineGym/scripts/${CK}/files/checkpoint_final.pt"
[ -f "$CK" ] || { say "找不到 checkpoint，终止"; exit 1; }
say "checkpoint: $CK"

# ---- 评测：ood 10 个 seed，评测时同样带上对齐后的阈值 ----
EVB="task=Track algo=ppo task.drone_model.name=BlueROV task.traj_scale_mult=2.5 \
headless=true enable_livestream=false wandb.mode=offline eval_only=true \
eval_episodes=100 +eval_max_batches=60 task.env.num_envs=64"
for s in 1 2 3 4 5 6 7 8 9 10; do
  f="$OUT/eval/${LBL}_ood_s${s}.log"
  grep -q 'EVAL-ONLY RESULTS' "$f" 2>/dev/null && continue
  timeout 1500 $R train.py $EVB seed=$s load_ckpt="$CK" \
    task.keepout.dynamic.speed=[2.2,2.8] $ABS $MATCH > "$f" 2>&1
  say "eval ood s$s rc=$?"
done
say "==== 全部完成 ===="
