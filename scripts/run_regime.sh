#!/usr/bin/env bash
# 找出安全滤波器**真正能起作用的扰动区间**。
#
# 动机（strong 档实测）：
#   · PPO 违约 0.385，ours 0.390 —— 滤波器 68% 时间在介入，零安全收益
#   · MPC-only 违约 **0.993** —— 全接管在强扰动下彻底崩溃
#   · `sat_frac` = 0.63 —— 六个推进器 63% 的时间饱和
# 推论：strong 档载具已接近控制权限极限，没有多余权限可供滤波器重新分配，
# 所以任何接管都只是把跟踪换掉而换不来安全。方法能起作用的区间应当在
# **违约已经出现、但推力尚未饱和**的中等扰动带。
#
# 本脚本沿扰动强度扫 ppo / ours / mpc_only 三格，同时看违约率与 sat_frac，
# 定位那条带；找不到就是方法的真实适用边界（同样是可发表的结论）。
set -o pipefail
CKPT=${CKPT:?需要 CKPT}
export CKPT MPPI_K=${MPPI_K:-64} MPPI_N=${MPPI_N:-10}
export NUM_ENVS=${NUM_ENVS:-128} EP=${EP:-500} MB=${MB:-12}
export MIN_FREE=${MIN_FREE:-2500} WAIT_MAX=${WAIT_MAX:-7200}
S=/home/jovyan/MarineGym-flow/scripts
OUT=/home/jovyan/MarineGym-flow/scripts/outputs_flow/regime
mkdir -p "$OUT"
R=$S/run_flow.sh

SPEEDS=${SPEEDS:-"0.6,1.0 0.9,1.3 1.2,1.6 1.5,2.2"}
CELLS=${CELLS:-"ppo ours mpc_only"}
G="task.safety.risk.threshold=0.6 task.safety.mppi.soft_hi=0.6 \
task.safety.mppi.num_samples=$MPPI_K task.safety.mppi.horizon=$MPPI_N"

for sp in $SPEEDS; do
  lo=${sp%,*}; hi=${sp#*,}
  for c in $CELLS; do
    LOG="$OUT/${c}__v${lo}_${hi}.log"
    grep -q 'EVAL-ONLY RESULTS' "$LOG" 2>/dev/null && { echo "[skip] $c v$lo-$hi"; continue; }
    case "$c" in
      ppo)      CF="task.safety.risk.enable=false task.safety.mppi.enable=false" ;;
      ours)     CF="task.safety.risk.enable=true task.safety.risk.gate=true task.safety.mppi.enable=true task.safety.mppi.soft=true $G" ;;
      mpc_only) CF="task.safety.risk.enable=false task.safety.mppi.enable=true task.safety.mppi.force_lambda=1.0 task.safety.mppi.w_ref=1.0 task.safety.mppi.w_track=0.0 task.safety.mppi.center=prev $G" ;;
    esac
    echo "[$(date '+%H:%M:%S')] $c  gust=[$lo,$hi]"
    timeout 5400 bash "$R" train.py \
      task=Track algo=ppo task.drone_model.name=BlueROV \
      headless=true enable_livestream=false wandb.mode=offline \
      eval_only=true eval_episodes=$EP +eval_max_batches=$MB \
      task.env.num_envs=$NUM_ENVS seed=${SEED:-0} load_ckpt="$CKPT" \
      task.corridor.enable=true task.gust.enable=true \
      task.gust.speed="[$lo,$hi]" $CF \
      +ep_dump="${LOG%.log}.episodes.csv" > "$LOG" 2>&1
    echo "  rc=$?"
  done
done

echo
/home/jovyan/envs/sim/bin/python "$S/regime_table.py"
