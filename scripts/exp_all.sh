#!/usr/bin/env bash
# 论文② 评测全流程：评测矩阵（多 seed）→ 汇总 → kill-switch 判决 → 出表出图 → 显著性。
#
#   CKPT=/path/checkpoint.pt MPPI_K=64 MPPI_N=10 SEEDS="0 1 2" bash exp_all.sh
#
# 实验设计（用户 2026-08-26 明确）：**训练只跑一次**，方差来自**不同 seed 的评测**。
#   · 所有格子共用同一条策略 → 各格差异纯粹来自滤波逻辑
#   · 同一 seed 内，各格面对完全相同的轨迹与阵风序列（环境用全局 RNG、阵风与 MPPI
#     各有独立 generator）→ 格子之间是**配对**比较
#   · 跨 seed 改变场景抽样 → 提供误差棒
# 为了不让成本乘以 seed 数，单个 seed 的 batch 数相应减少，总 episode 数不变。
#
# 顺序有意为之：先跑决定生死的四格，立刻出 kill-switch 判决；不过就不必再烧剩下的。
set -o pipefail
CKPT=${CKPT:?需要 CKPT=<checkpoint.pt>}
S=/home/jovyan/MarineGym-flow/scripts
PY=/home/jovyan/envs/sim/bin/python
export CKPT

SUF=""
[ -n "${MPPI_K:-}${MPPI_N:-}" ] && SUF="_K${MPPI_K:-def}N${MPPI_N:-def}"
EVDIR=/home/jovyan/MarineGym-flow/scripts/outputs_flow/eval$SUF

STAGE1_CELLS=${STAGE1_CELLS:-"ppo mpc_only react_soft ours"}
STAGE2_CELLS=${STAGE2_CELLS:-"pred_binary react_binary dhat_zero dhat_oracle"}
SCENES=${SCENES:-"calm nominal strong fast"}
SEEDS=${SEEDS:-"0 1 2"}

run_stage () {
  local cells="$1"
  for sd in $SEEDS; do
    for sc in $SCENES; do
      for c in $cells; do
        bash "$S/exp_eval.sh" "$c" "$sc" "$sd" || echo "  ⚠ $c/$sc/s$sd 失败，继续"
      done
    done
  done
}

echo "==== 第一批（kill-switch 最小集合）seeds=[$SEEDS] ===="
run_stage "$STAGE1_CELLS"
$PY "$S/flow_collect.py" --dir "$EVDIR"
echo
echo "==== kill-switch 判决 ===="
$PY "$S/flow_killswitch.py"

echo
echo "==== 第二批（补完消融矩阵）===="
run_stage "$STAGE2_CELLS"
$PY "$S/flow_collect.py" --dir "$EVDIR"
$PY "$S/flow_report.py"
echo
echo "==== 显著性检验（各格 vs ours）===="
$PY "$S/flow_significance.py" --dir "$EVDIR" || echo "(逐 episode 数据不全，跳过)"
