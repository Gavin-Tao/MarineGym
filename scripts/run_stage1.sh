#!/usr/bin/env bash
# 训练结束后的第一阶段：标定扫描 + K4。两者都不含 MPPI，相对快，
# 且都是"决定后续该怎么跑"的前置信息 —— 先拿到它们再投入昂贵的全矩阵。
#
#   1) 标定扫描：扫阵风幅值，找出让 PPO 违约率落在 10–60% 的区间 → 定 OOD 档位
#   2) K4：扰动观测器是否让预测门控可信 → 本篇一号贡献是否成立
set -o pipefail
S=/home/jovyan/MarineGym-flow/scripts

CKPT=${CKPT:-}
if [ -z "$CKPT" ]; then
  # 取最近一次训练目录里帧数最大的 checkpoint（可能没有 final，训练被提前停时用中间点）
  D=$(ls -td "$S"/wandb/offline-run-* 2>/dev/null | head -1)
  CKPT=$(ls "$D"/files/checkpoint_final.pt 2>/dev/null || \
         ls "$D"/files/checkpoint_*.pt 2>/dev/null | sed 's/.*checkpoint_\([0-9]*\)\.pt/\1 &/' | sort -nr | head -1 | cut -d' ' -f2)
fi
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "找不到 checkpoint"; exit 1; }
export CKPT
echo "使用 checkpoint: $CKPT"

echo
echo "======== 1/2 场景标定扫描 ========"
NUM_ENVS=${NUM_ENVS:-32} EP=${EP:-60} MB=${MB:-30} bash "$S/run_calibrate.sh"

echo
echo "======== 2/2 K4 扰动观测器 ========"
STEPS=${K4_STEPS:-300} bash "$S/k4_observer.sh"
