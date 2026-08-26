#!/usr/bin/env bash
# 论文② 评测全流程：评测矩阵 → 汇总 → kill-switch 判决 → 出表出图。
# 训练由 exp_train.sh 单独跑（GPU 紧张，严格串行），本脚本假定 CKPT 已就绪。
#
#   CKPT=/path/checkpoint.pt MPPI_K=64 MPPI_N=10 bash exp_all.sh
#
# 顺序有意为之：先跑决定生死的四格（ppo / mpc_only / react_soft / ours），
# 立刻出 kill-switch 判决；判决不过就没必要再烧剩下的格子。
set -o pipefail
CKPT=${CKPT:?需要 CKPT=<checkpoint.pt>}
S=/home/jovyan/MarineGym-flow/scripts
PY=/home/jovyan/envs/sim/bin/python
export CKPT
SEED=${SEED:-0}

# 结果目录随 MPPI 规模分开，避免不同规模的结果混在一张表里
SUF=""
[ -n "${MPPI_K:-}${MPPI_N:-}" ] && SUF="_K${MPPI_K:-def}N${MPPI_N:-def}"
EVDIR=/home/jovyan/MarineGym-flow/scripts/outputs_flow/eval$SUF

STAGE1_CELLS="ppo mpc_only react_soft ours"
STAGE2_CELLS="pred_binary react_binary dhat_zero dhat_oracle"
SCENES=${SCENES:-"calm nominal strong fast"}

run_stage () {
  local cells="$1"
  for sc in $SCENES; do
    for c in $cells; do
      bash "$S/exp_eval.sh" "$c" "$sc" "$SEED" || echo "  ⚠ $c/$sc/s$SEED 失败，继续"
    done
  done
}

echo "==== 第一批（kill-switch 最小集合）===="
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
# 统计功效提示：n≈68 时只够分辨 ~0.2 量级的违约率差（strong 档够用）；
# nominal 档若要分辨 ~0.10 的差，每格需要约 250 个 episode，见 RESULTS.md 的说明。
$PY "$S/flow_significance.py" --dir "$EVDIR" || echo "(逐 episode 数据不全，跳过)"
