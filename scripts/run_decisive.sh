#!/usr/bin/env bash
# 判断"方向是否可行"的最小决定性集合：strong / fast 两个 OOD 档补齐四个格子。
#
# 为什么是这两档：K4 已经证明扰动观测器的收益**集中在强扰动区**（nominal 档反而
# d̂=0 更好）。calm/nominal 已跑完，ours 相对 PPO 的差异在噪声内 —— 那是预期之中的，
# 因为那里本来就不是方法针对的区域。strong/fast 才是判据。
#
# GPU 三个 session 争抢，守卫用长等待（不 OOM，排队等）。
set -o pipefail
CKPT=${CKPT:?需要 CKPT}
export CKPT MPPI_K=${MPPI_K:-64} MPPI_N=${MPPI_N:-10}
export NUM_ENVS=${NUM_ENVS:-128} EP=${EP:-500} MB=${MB:-12}
export MIN_FREE=${MIN_FREE:-2000} WAIT_MAX=${WAIT_MAX:-7200}
S=/home/jovyan/MarineGym-flow/scripts

# 顺序按信息量：先把 strong 档补全（ppo/mpc_only 已有），再上 fast
for spec in "ours strong" "react_soft strong" "ours fast" "ppo fast" "react_soft fast" "mpc_only fast"; do
  set -- $spec
  bash "$S/exp_eval.sh" "$1" "$2" "${SEED:-0}" || echo "  ⚠ $1/$2 失败，继续"
  /home/jovyan/envs/sim/bin/python "$S/flow_collect.py" \
    --dir /home/jovyan/MarineGym-flow/scripts/outputs_flow/eval_K${MPPI_K}N${MPPI_N} >/dev/null 2>&1
  /home/jovyan/envs/sim/bin/python "$S/make_prelim_table.py" >/dev/null 2>&1
  echo "  [表已更新] $1/$2"
done

/home/jovyan/envs/sim/bin/python "$S/flow_significance.py" \
  --dir /home/jovyan/MarineGym-flow/scripts/outputs_flow/eval_K${MPPI_K}N${MPPI_N} || true
