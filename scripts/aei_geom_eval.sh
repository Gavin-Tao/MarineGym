#!/usr/bin/env bash
# r1b_v2 / r1c_v2 的 ood 评测。
# 事故记录：① speed 被冻结集反压 → 跑成 nominal；② seed 被冻结集反压 → 10 次全 seed=0。
# 现在：冻结集不含 seed/场景/消融开关，且每条命令跑前用 check_overrides.py 预检。
OUT=/home/jovyan/MarineGym/scripts/outputs_aei
R=/home/jovyan/MarineGym/scripts/run_aei.sh
LOG=$OUT/geom_progress.log
CHK=/home/jovyan/MarineGym/scripts/check_overrides.py
source /home/jovyan/MarineGym/scripts/aei_fixed_params.sh

EVB="task=Track algo=ppo task.drone_model.name=BlueROV task.traj_scale_mult=2.5 \
headless=true enable_livestream=false wandb.mode=offline eval_only=true \
eval_episodes=100 +eval_max_batches=60 task.env.num_envs=64"
say(){ echo "[$(date '+%H:%M')] $*" >> "$LOG"; }

run(){ local lbl=$1; shift
  local run ck
  run=$(grep -oE "wandb/offline-run-[0-9_a-z-]+" "$OUT/${lbl}.log" 2>/dev/null | head -1)
  ck="/home/jovyan/MarineGym/scripts/${run}/files/checkpoint_final.pt"
  [ -f "$ck" ] || { say "MISS ckpt $lbl"; return 1; }
  for s in 1 2 3 4 5 6 7 8 9 10; do
    local f="$OUT/eval/${lbl}_ood_s${s}.log"
    grep -q 'EVAL-ONLY RESULTS' "$f" 2>/dev/null && continue
    # 变量部分一律最后：消融开关 → 场景 → seed
    local CMD="$EVB $FIXED_ALL $MPPI_ON load_ckpt=$ck $* $SCENE_OOD seed=$s"
    python3 "$CHK" $CMD || { say "!!! $lbl s$s 预检失败，已中止"; return 1; }
    timeout 1500 $R train.py $CMD > "$f" 2>&1
    say "$lbl ood s$s rc=$?"
  done; }

say "==== ood 评测（第 3 次；已加覆盖预检）===="
run r1b_geom_soft_v2   task.keepout.risk.gate=false task.keepout.mppi.soft_blend=true  || exit 1
run r1c_geom_binary_v2 task.keepout.risk.gate=false task.keepout.mppi.soft_blend=false || exit 1
say "==== 几何两格 ood 完成 ===="
