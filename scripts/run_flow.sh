#!/usr/bin/env bash
# 论文② (flow-safety) 实验统一启动器。
#
# 与主树的 run_aei.sh 的区别 —— 不要混用：
#   run_aei.sh 硬编码 cd /home/jovyan/MarineGym/scripts，跑的是论文①的代码。
#   本脚本把 PYTHONPATH 指向本 worktree，压过 editable 安装
#   (__editable__.marinegym-1.0.pth → /home/jovyan/MarineGym)，
#   否则在本目录下跑的仍然是主树的 marinegym（静默，不报错）。
set -o pipefail
MG_ROOT=/home/jovyan/MarineGym-flow
export PYTHONPATH="$MG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /opt/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/envs/sim
cd "$MG_ROOT/scripts" || exit 1
export WANDB_MODE=offline

# 自检：解析错了就直接退出，别浪费一次 Isaac 启动（~2-3 min）
_r=$(python -c "import marinegym;print(marinegym.__file__)" 2>/dev/null)
case "$_r" in
  "$MG_ROOT"/*) : ;;
  *) echo "[run_flow] FATAL: marinegym 解析到 $_r ，应为 $MG_ROOT/..." >&2; exit 2 ;;
esac

# ── 显存守卫 ────────────────────────────────────────────────────────────────
# 这台机器上 4 张 L40 被其他租户 + 另外两个 session（论文①、论文③）共享，空闲显存
# 经常只剩 2–3 GB。Isaac 起不来就是一次 CUDA OOM，白等一分钟启动。所以：
#   · 没显式指定卡时，自动选空闲最多的一张
#   · 选定后等到空闲显存达标再启动（超时则放弃，让上层脚本记为失败而不是崩在半路）
MIN_FREE=${MIN_FREE:-2600}          # MiB
WAIT_MAX=${WAIT_MAX:-1800}          # 秒
_free_of () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null; }
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                         | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
  export CUDA_VISIBLE_DEVICES
  echo "[run_flow] 自动选卡 GPU$CUDA_VISIBLE_DEVICES"
fi
_t=0
while [ "$(_free_of "$CUDA_VISIBLE_DEVICES")" -lt "$MIN_FREE" ] 2>/dev/null; do
  if [ "$_t" -ge "$WAIT_MAX" ]; then
    echo "[run_flow] FATAL: GPU$CUDA_VISIBLE_DEVICES 空闲显存 $(_free_of "$CUDA_VISIBLE_DEVICES") MiB < $MIN_FREE，等待 ${WAIT_MAX}s 超时" >&2
    exit 3
  fi
  [ $((_t % 120)) -eq 0 ] && echo "[run_flow] 等 GPU$CUDA_VISIBLE_DEVICES 显存：$(_free_of "$CUDA_VISIBLE_DEVICES") / $MIN_FREE MiB"
  sleep 15; _t=$((_t + 15))
done

exec python "$@"
