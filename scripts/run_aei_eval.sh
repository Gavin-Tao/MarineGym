#!/usr/bin/env bash
# AEI 评测驱动 —— 一律 P1 配置：不覆盖 num_samples / horizon / risk.horizon，走代码默认 128/20/15
OUT=/home/jovyan/MarineGym/scripts/outputs_aei/eval
mkdir -p "$OUT"
R=/home/jovyan/MarineGym/scripts/run_aei.sh
EPS=${EPS:-100}

BASE="task=Track algo=ppo task.drone_model.name=BlueROV task.traj_scale_mult=2.5 \
headless=true enable_livestream=false wandb.mode=offline eval_only=true eval_episodes=$EPS"

KO="task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true"
ABS="$KO task.keepout.mppi.enable=true task.keepout.mppi.exact=true \
task.keepout.risk.enable=true task.keepout.mppi.soft_blend=true"

# eval_one <gpu> <tag> <ckpt> <seed> <scene> <extra...>
eval_one() {
  local gpu=$1 tag=$2 ckpt=$3 seed=$4 scene=$5; shift 5
  local sc=""
  [ "$scene" = "ood" ] && sc="task.keepout.dynamic.speed=[2.2,2.8]"
  local f="$OUT/${tag}_${scene}_s${seed}.log"
  [ -s "$f" ] && grep -q 'EVAL-ONLY RESULTS' "$f" && return 0   # 断点续跑
  CUDA_VISIBLE_DEVICES=$gpu $R train.py $BASE seed=$seed load_ckpt="$ckpt" $sc "$@" > "$f" 2>&1
  echo "[$(date '+%H:%M:%S')] $tag $scene s$seed exit=$?" >> "$OUT/../eval_progress.log"
}
export -f eval_one
