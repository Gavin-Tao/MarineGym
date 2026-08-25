#!/usr/bin/env bash
# 全栈冒烟：走廊 + 阵风 + 扰动观测器 + A(预测风险) + B(MPPI 软接管)。
# 只看会不会崩、λ/d̂/违约的量级对不对，不看性能（策略未训练）。
OUT=/home/jovyan/MarineGym-flow/scripts/outputs_flow/regress
mkdir -p "$OUT"
TAG=${TAG:-full}
${GPU:+CUDA_VISIBLE_DEVICES=$GPU} timeout 1200 bash /home/jovyan/MarineGym-flow/scripts/run_flow.sh train.py \
  task=Track algo=ppo task.drone_model.name=BlueROV \
  headless=true enable_livestream=false wandb.mode=offline seed=0 \
  task.env.num_envs=16 \
  task.corridor.enable=true task.gust.enable=true \
  task.safety.risk.enable=true task.safety.mppi.enable=true \
  ${EXTRA:-} \
  +save_traj="$OUT/smoke_$TAG.npz" +traj_steps=${STEPS:-120} \
  > "$OUT/smoke_$TAG.log" 2>&1
echo "rc=$?"
tail -5 "$OUT/smoke_$TAG.log"
