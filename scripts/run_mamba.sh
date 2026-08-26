#!/usr/bin/env bash
# 论文③ (mamba) 实验统一启动器。
#
# 与另外两篇的区别 —— 不要混用：
#   run_aei.sh  硬编码 cd /home/jovyan/MarineGym/scripts       → 论文① 代码 + envs/sim
#   run_flow.sh 指向 /home/jovyan/MarineGym-flow               → 论文② 代码 + envs/sim
#   本脚本      指向 /home/jovyan/MarineGym-mamba              → 论文③ 代码 + envs/sim-mamba
#
# 两个必须压过去的默认值：
#   1) PYTHONPATH：editable 安装 (__editable__.marinegym-1.0.pth) 指向主树
#      /home/jovyan/MarineGym，不压过去会【静默】跑主树的 marinegym，不报错。
#   2) conda env：envs/sim-mamba 是 envs/sim 的克隆 + mamba_ssm/causal_conv1d，
#      论文①② 的 envs/sim 保持原样，绝不在里面装东西。
set -o pipefail
MG_ROOT=/home/jovyan/MarineGym-mamba
MG_ENV=/home/jovyan/envs/sim-mamba
export PYTHONPATH="$MG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /opt/conda/etc/profile.d/conda.sh
conda activate "$MG_ENV"
cd "$MG_ROOT/scripts" || exit 1
export WANDB_MODE=offline
# 共享 GPU 上显存紧张，减少碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 自检 1：marinegym 必须解析到本 worktree，否则直接退出，
# 别浪费一次 Isaac 启动（~2-3 min）
_r=$(python -c "import marinegym;print(marinegym.__file__)" 2>/dev/null)
case "$_r" in
  "$MG_ROOT"/*) : ;;
  *) echo "[run_mamba] FATAL: marinegym 解析到 $_r ，应为 $MG_ROOT/..." >&2; exit 2 ;;
esac

# 自检 2：conda 前缀必须是 sim-mamba（防止误跑进论文①② 的环境）
_p=$(python -c "import sys;print(sys.prefix)" 2>/dev/null)
case "$_p" in
  "$MG_ENV"*) : ;;
  *) echo "[run_mamba] FATAL: sys.prefix=$_p ，应为 $MG_ENV" >&2; exit 2 ;;
esac

exec python "$@"
