#!/usr/bin/env bash
# 论文② 全流程：评测矩阵 → 汇总 → kill-switch 判决 → 出表出图。
# 训练由 exp_train.sh 单独跑（GPU 紧张，串行），本脚本假定 CKPT 已就绪。
#
#   CKPT=/path/checkpoint_final.pt bash exp_all.sh
#
# 顺序有意为之：先跑决定生死的几格（ppo / mpc_only / react_soft / ours），
# 跑完立刻出 kill-switch 判决；判决不过就没必要再烧剩下的格子。
set -o pipefail
CKPT=${CKPT:?需要 CKPT=<checkpoint_final.pt>}
S=/home/jovyan/MarineGym-flow/scripts
PY=/home/jovyan/envs/sim/bin/python
export CKPT

# 第一批：kill-switch 需要的最小集合
STAGE1_CELLS="ppo mpc_only react_soft ours"
STAGE1_SCENES="calm nominal strong fast"
# 第二批：补完消融矩阵
STAGE2_CELLS="pred_binary react_binary dhat_zero dhat_oracle"
STAGE2_SCENES="calm nominal strong fast"

run_stage () {
  local cells="$1" scenes="$2" seed="$3"
  for sc in $scenes; do
    for c in $cells; do
      bash "$S/exp_eval.sh" "$c" "$sc" "$seed" || echo "  ⚠ $c/$sc/s$seed 失败，继续"
    done
  done
}

SEED=${SEED:-0}
echo "==== 第一批（kill-switch 最小集合）===="
run_stage "$STAGE1_CELLS" "$STAGE1_SCENES" "$SEED"
$PY "$S/flow_collect.py"
echo
echo "==== kill-switch 判决 ===="
$PY "$S/flow_killswitch.py"

echo
echo "==== 第二批（补完消融矩阵）===="
run_stage "$STAGE2_CELLS" "$STAGE2_SCENES" "$SEED"
$PY "$S/flow_collect.py"
$PY "$S/flow_report.py"
