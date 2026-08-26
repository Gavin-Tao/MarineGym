#!/usr/bin/env bash
# 论文② 评测矩阵。用法： CKPT=<path> bash exp_eval.sh <cell> <scene> [seed]
#
# 所有滤波器格子**共用同一条策略**（CKPT），只改滤波器开关 —— 各格差异纯粹来自
# 滤波逻辑。前提是 safety.risk.in_obs=false（默认），观测维度处处一致。
#
# 格子（每格相对 ours 只差一个开关）：
#   ppo          无安全栈（下界）
#   ours         预测门控 + 斜坡 λ + 在线 d̂
#   pred_binary  预测门控 + **阶跃**接管
#   react_soft   **反应式**门控（当前裕度）+ 斜坡
#   react_binary 反应式门控 + 阶跃
#   dhat_zero    ours 但 d̂≡0（退化成论文①的完美模型假设）
#   dhat_oracle  ours 但用真值流速算 d̂（观测器的性能上界）
#   mpc_only     λ≡1 全程 MPPI + 参考跟踪代价（"为什么不直接用 MPC"的对照）
#
# 场景（训练档是 nominal；其余为 OOD）：
#   nominal  speed[0.6,1.0] ramp[0.2,0.5]        —— 训练分布，这里违约率**低**才对
#   strong   speed[1.5,2.2] ramp[0.2,0.5]        —— 幅值 OOD
#   fast     speed[1.5,2.2] ramp[0.08,0.15]      —— 幅值+上升沿 OOD；与 strong 只差上升沿
#   calm     gust 关闭                            —— 无扰动，测滤波器的"代价"(nominal RMSE)
set -o pipefail
CELL=${1:?用法: exp_eval.sh <cell> <scene> [seed]}
SCENE=${2:-nominal}
SEED=${3:-0}
CKPT=${CKPT:?需要 CKPT=<checkpoint_final.pt>}
OUT=/home/jovyan/MarineGym-flow/scripts/outputs_flow/eval
mkdir -p "$OUT"
R=/home/jovyan/MarineGym-flow/scripts/run_flow.sh

BASE="task=Track algo=ppo task.drone_model.name=BlueROV \
headless=true enable_livestream=false wandb.mode=offline \
eval_only=true eval_episodes=${EP:-100} +eval_max_batches=${MB:-40} \
task.env.num_envs=${NUM_ENVS:-64} seed=$SEED load_ckpt=$CKPT \
task.corridor.enable=true"

case "$SCENE" in
  nominal) SC="task.gust.enable=true" ;;
  strong)  SC="task.gust.enable=true task.gust.speed=[1.5,2.2]" ;;
  # fast 保持与 strong 相同的幅值，只把上升沿变陡 —— 这样 fast vs strong 是
  # **单变量对比**，隔离出"反应式门控来不及"这个效应（K3 的关键场景）。
  # 若只改上升沿而保持 nominal 幅值，梯形总冲量 amp×(τ+hold) 反而更小、场景更容易，
  # 没有可分辨空间（实测 PPO 违约率与 nominal 同为 0.016）。
  fast)    SC="task.gust.enable=true task.gust.speed=[1.5,2.2] task.gust.ramp=[0.08,0.15]" ;;
  calm)    SC="task.gust.enable=false" ;;
  *) echo "未知 scene: $SCENE"; exit 2 ;;
esac

# 冻结：门控边界在预测/几何两种门控下必须同值，否则比的是阈值不是门控方式
G="task.safety.risk.threshold=0.6 task.safety.mppi.soft_hi=0.6"
# MPPI 规模可覆盖（K×N 决定评测墙钟：完整 128×20 下单格约 40 min，28 个含 MPPI 的
# 格子要 21 小时）。策略是先用小规模跑通全矩阵，再用完整规模复跑 headline 几格验证
# 结论不随规模翻转。**同一批结果里所有格子必须用同一组 K/N**，否则比的不是方法。
[ -n "${MPPI_K:-}" ] && G="$G task.safety.mppi.num_samples=$MPPI_K"
[ -n "${MPPI_N:-}" ] && G="$G task.safety.mppi.horizon=$MPPI_N"
case "$CELL" in
  ppo)          CF="task.safety.risk.enable=false task.safety.mppi.enable=false" ;;
  ours)         CF="task.safety.risk.enable=true  task.safety.risk.gate=true  task.safety.mppi.enable=true task.safety.mppi.soft=true  $G" ;;
  pred_binary)  CF="task.safety.risk.enable=true  task.safety.risk.gate=true  task.safety.mppi.enable=true task.safety.mppi.soft=false $G" ;;
  react_soft)   CF="task.safety.risk.enable=true  task.safety.risk.gate=false task.safety.mppi.enable=true task.safety.mppi.soft=true  $G" ;;
  react_binary) CF="task.safety.risk.enable=true  task.safety.risk.gate=false task.safety.mppi.enable=true task.safety.mppi.soft=false $G" ;;
  dhat_zero)    CF="task.safety.risk.enable=true  task.safety.risk.gate=true  task.safety.mppi.enable=true task.safety.mppi.soft=true  task.safety.dobs.zero=true $G" ;;
  dhat_oracle)  CF="task.safety.risk.enable=true  task.safety.risk.gate=true  task.safety.mppi.enable=true task.safety.mppi.soft=true  task.safety.dobs.oracle=true task.safety.dobs.enable=false $G" ;;
  mpc_only)     CF="task.safety.risk.enable=false task.safety.mppi.enable=true task.safety.mppi.force_lambda=1.0 task.safety.mppi.w_ref=1.0 task.safety.mppi.w_track=0.0 task.safety.mppi.center=prev $G" ;;
  *) echo "未知 cell: $CELL"; exit 2 ;;
esac

SUF=""
[ -n "${MPPI_K:-}${MPPI_N:-}" ] && SUF="_K${MPPI_K:-def}N${MPPI_N:-def}"
mkdir -p "$OUT$SUF"
LOG="$OUT$SUF/${CELL}__${SCENE}__s${SEED}.log"
if grep -q 'EVAL-ONLY RESULTS' "$LOG" 2>/dev/null; then echo "[skip] $CELL/$SCENE/s$SEED 已完成"; exit 0; fi
echo "[$(date '+%H:%M:%S')] eval $CELL / $SCENE / s$SEED"
# 显式指定 GPU 时用 export，**不能**写成 `${GPU:+CUDA_VISIBLE_DEVICES=$GPU} cmd`：
# 变量赋值前缀必须是字面量，由参数展开产生的会被 bash 当成命令名执行，报
# "CUDA_VISIBLE_DEVICES=2: command not found" 并且**返回 0**，整格静默跳过。
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"
timeout ${TIMEOUT:-5400} bash "$R" train.py $BASE $SC $CF \
  +ep_dump="${LOG%.log}.episodes.csv" > "$LOG" 2>&1
echo "[$(date '+%H:%M:%S')] rc=$?"
sed -n '/EVAL-ONLY RESULTS/,$p' "$LOG" | head -25
