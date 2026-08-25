#!/usr/bin/env bash
# 走廊 + 阵风的冒烟测试：600 步，只看会不会崩、指标数量级对不对。
OUT=/home/jovyan/MarineGym-flow/scripts/outputs_flow/regress
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=${GPU:-2} timeout 900 bash /home/jovyan/MarineGym-flow/scripts/run_flow.sh train.py \
  task=Track algo=ppo task.drone_model.name=BlueROV \
  headless=true enable_livestream=false wandb.mode=offline seed=0 \
  task.env.num_envs=16 \
  task.corridor.enable=true task.gust.enable=true \
  +save_traj="$OUT/smoke_corridor.npz" +traj_steps=600 \
  > "$OUT/smoke_corridor.log" 2>&1
echo "rc=$?"
