#!/usr/bin/env bash
# K4：扰动观测器是否让预测门控可信。
#
# 开环诊断，**不需要训练好的策略** —— 测的是名义模型 + d̂ 的预测质量，
# 与策略好坏无关（策略只决定被前滚的那个动作，三种 d̂ 用的是同一个动作）。
# 因此这是四个 kill-switch 里最先该跑的：最便宜，且直接决定本篇一号贡献是否成立。
#
# 关掉 MPPI（mppi.enable=false）：只跑 A，快得多，而且避免滤波器改变动作、
# 让三种预测面对不同的执行序列。
OUT=/home/jovyan/MarineGym-flow/scripts/outputs_flow/k4
mkdir -p "$OUT"
STEPS=${STEPS:-300}
H=${H:-15}
# 显式指定 GPU 时用 export，**不能**写成 `${GPU:+CUDA_VISIBLE_DEVICES=$GPU} cmd`：
# 变量赋值前缀必须是字面量，由参数展开产生的会被 bash 当成命令名执行，报
# "CUDA_VISIBLE_DEVICES=2: command not found" 并且**返回 0**，整格静默跳过。
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"
timeout 1800 bash /home/jovyan/MarineGym-flow/scripts/run_flow.sh train.py \
  task=Track algo=ppo task.drone_model.name=BlueROV \
  headless=true enable_livestream=false wandb.mode=offline seed=0 \
  task.env.num_envs=${K4_ENVS:-32} \
  task.corridor.enable=true task.gust.enable=true \
  task.gust.speed=[${GSPEED:-0.6,1.0}] task.gust.ramp=[${GRAMP:-0.2,0.5}] \
  task.safety.risk.enable=true task.safety.mppi.enable=false \
  task.safety.risk.horizon=$H task.safety.k4=true \
  task.safety.dobs.alpha=${ALPHA:-0.1} \
  +save_traj="$OUT/k4${TAG:+_$TAG}.npz" +traj_steps=$STEPS \
  > "$OUT/k4${TAG:+_$TAG}.log" 2>&1
echo "capture rc=$?"
/home/jovyan/envs/sim/bin/python /home/jovyan/MarineGym-flow/scripts/flow_validate.py \
  k4 --npz "$OUT/k4${TAG:+_$TAG}.npz" --horizon $H
