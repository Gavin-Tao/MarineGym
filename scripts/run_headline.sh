#!/usr/bin/env bash
# Headline 对比：ours vs PPO（论文主线），加上判断"必须预测"的反应式对照，
# 以及回答"为什么不直接用 MPC"的 MPC-only。
#
# 场景优先级按信息量排：strong 是 OOD 主战场（PPO 违约 0.385，有充分改进空间），
# fast 隔离上升沿效应，calm 给出滤波器的标称代价（帕累托图的横轴），
# nominal 是训练分布（PPO 已经安全，预期打平 —— 这与 K4 一致，不是坏消息）。
set -o pipefail
CKPT=${CKPT:?需要 CKPT}
export CKPT MPPI_K=${MPPI_K:-64} MPPI_N=${MPPI_N:-10}
export NUM_ENVS=${NUM_ENVS:-128} EP=${EP:-500} MB=${MB:-12}
export MIN_FREE=${MIN_FREE:-2500} WAIT_MAX=${WAIT_MAX:-7200}
S=/home/jovyan/MarineGym-flow/scripts
SUF="_K${MPPI_K}N${MPPI_N}"
EVDIR=/home/jovyan/MarineGym-flow/scripts/outputs_flow/eval$SUF

# ppo 也在本目录内重跑一遍：它不用 MPPI（每格约 5 min），但必须与其它格子处在
# **同一输出目录、同一 MPPI 规模口径**下，否则汇总时会混用两批不同配置的结果。
for spec in "ppo strong" "ours strong" "react_soft strong" "mpc_only strong" \
            "ppo fast" "ours fast" "react_soft fast" "mpc_only fast" \
            "ppo calm" "ours calm" "ppo nominal" "ours nominal" "mpc_only calm" "react_soft calm" \
            "mpc_only nominal" "react_soft nominal"; do
  set -- $spec
  bash "$S/exp_eval.sh" "$1" "$2" "${SEED:-0}" || echo "  ⚠ $1/$2 失败，继续"
  /home/jovyan/envs/sim/bin/python "$S/flow_collect.py" --dir "$EVDIR" >/dev/null 2>&1
  echo "  [已完成] $1/$2"
done
/home/jovyan/envs/sim/bin/python "$S/flow_collect.py" --dir "$EVDIR"
/home/jovyan/envs/sim/bin/python "$S/make_prelim_table.py"
/home/jovyan/envs/sim/bin/python "$S/flow_significance.py" --dir "$EVDIR" || true
