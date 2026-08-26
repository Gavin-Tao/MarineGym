#!/usr/bin/env bash
# 观测器低通系数 α 的敏感性扫描。
#
# 为什么需要（见 outputs_flow/K4_FINDINGS.md）：风险监视器把策略动作零阶保持前滚，
# 同时注入扰动力，而现实中策略会反应 —— 于是完整扰动被"重复计入"，系统性高估漂移。
# 实测偏置：d̂=0 → +0.077（低估漂移）；d̂=oracle → −0.085（高估漂移）；
# 在线估计 α=0.1 → −0.019。也就是说**α 是在这两个偏差之间的调节旋钮**，
# 在线观测器最优部分来自低通衰减恰好抵消了零阶保持的偏差。
#
# 这个扫描把"幸运标定"变成有据可查的设计参数：给出 MAE / 偏置 / 门控一致率
# 随 α 的变化，并指出偏置过零点。
set -o pipefail
CKPT=${CKPT:?需要 CKPT}
export CKPT
S=/home/jovyan/MarineGym-flow/scripts
ALPHAS=${ALPHAS:-"0.02 0.05 0.10 0.20 0.50"}
GS=${GSPEED:-1.5,2.2}          # 默认在强扰动档扫，那里观测器才有发挥空间
for a in $ALPHAS; do
  TAG="a${a}" ALPHA="$a" GSPEED="$GS" STEPS=${STEPS:-300} bash "$S/k4_observer.sh" \
    2>&1 | sed "s/^/[α=$a] /"
done
