#!/usr/bin/env bash
# V1/V2/V3 洋流物理体检：六向自由漂移采集。
#
# 设计要点：
#   const_action=0        → T200 在 |throttle|<0.075 时 rpm=0 → **真正零推力**，纯自由漂移
#   task.reset_thres=1e6  → 关掉跟踪误差终止，300 步全是连续动力学（默认 0.5 会让它反复 reset）
#   噪声置零              → underwaterVehicle.py:185 的噪声是 rand_like（均匀[0,1]，均值非零），
#                           开着会给漂移加正偏置，污染 V3 的幅值标定
#
# 注意：set_flow_velocities 采的是 rand_like(...)*max_flow_vel = 均匀[0,max]，
# 不是恒定 max。所以判据对比的是 npz 里**实际记录的** flow_vels，不是配置值。
set -o pipefail
OUT=/home/jovyan/MarineGym-flow/scripts/outputs_flow/flowcheck
mkdir -p "$OUT"
R=/home/jovyan/MarineGym-flow/scripts/run_flow.sh
GPU=${GPU:-2}
SPEED=${SPEED:-0.5}
# 4000 步 = 64 s。实测漂移收敛的时间常数 τ≈25 s（低速段只剩线性阻尼，尾巴极长）：
# 400 步时 |v|/|v_flow| 才 0.73 且仍在上升 —— 那是未收敛，不是 bug。达到 0.95 需 ~3000 步。
STEPS=${STEPS:-4000}
# max_episode_length 必须一起抬高：默认 600 会在 600 步 truncate 并 reset，
# 把长漂移切断（reset_thres 只关掉了误差终止，管不了截断）。
BASE="task=Track algo=ppo task.drone_model.name=BlueROV headless=true enable_livestream=false \
wandb.mode=offline seed=0 task.env.num_envs=16 task.reset_thres=1000000 \
task.env.max_episode_length=$((STEPS + 100)) \
task.disturbances.train.flow.enable_flow=true \
task.disturbances.train.flow.flow_velocity_gaussian_noise=[0,0,0,0,0,0] \
+const_action=0 +traj_steps=$STEPS"

run_dir () {  # $1=tag  $2=六维流速向量
  local tag=$1 vec=$2
  local f="$OUT/drift_${tag}.npz"
  if [ -f "$f" ]; then echo "[skip] $tag 已存在"; return 0; fi
  echo "[$(date '+%H:%M:%S')] 采集 $tag  flow=$vec"
  CUDA_VISIBLE_DEVICES=$GPU timeout 900 bash "$R" train.py $BASE \
    task.disturbances.train.flow.max_flow_velocity="$vec" \
    +save_traj="$f" > "$OUT/drift_${tag}.log" 2>&1
  echo "[$(date '+%H:%M:%S')] $tag rc=$?"
}

S=$SPEED; N=$(python3 -c "print(-$SPEED)")
run_dir px "[$S,0,0,0,0,0]"
run_dir nx "[$N,0,0,0,0,0]"
run_dir py "[0,$S,0,0,0,0]"
run_dir ny "[0,$N,0,0,0,0]"

echo
echo "==== 体检表 ===="
/home/jovyan/envs/sim/bin/python /home/jovyan/MarineGym-flow/scripts/flow_validate.py \
  flow-suite --dir "$OUT" --speed "$SPEED"
