#!/usr/bin/env bash
# 场景标定扫描：固定策略，扫阵风幅值，找出 PPO 违约率落在 10–60% 的区间。
#
# 为什么需要：违约率太低 → 没有改进空间，任何滤波器都看不出差别（论文①的静态障碍
# 就是被 PPO 一轮打穿）；太高 → 场景不可救，同样没有区分度。这个区间必须先量出来
# 再定 nominal / strong / fast 三档的具体数值，而不是拍脑袋。
#
# 扫描结果本身也是论文的一张图：违约率 vs 扰动强度，能直接说明各方法的失效点在哪。
set -o pipefail
CKPT=${CKPT:?需要 CKPT}
export CKPT
S=/home/jovyan/MarineGym-flow/scripts
OUT=/home/jovyan/MarineGym-flow/scripts/outputs_flow/calib
mkdir -p "$OUT"
R=/home/jovyan/MarineGym-flow/scripts/run_flow.sh

SPEEDS=${SPEEDS:-"0.6,1.0 1.0,1.5 1.5,2.2 2.2,3.0 3.0,4.0"}
for sp in $SPEEDS; do
  lo=${sp%,*}; hi=${sp#*,}
  tag="v${lo}_${hi}"
  LOG="$OUT/ppo__$tag.log"
  if grep -q 'EVAL-ONLY RESULTS' "$LOG" 2>/dev/null; then echo "[skip] $tag"; continue; fi
  echo "[$(date '+%H:%M:%S')] 标定 gust.speed=[$lo,$hi]"
  timeout 3600 bash "$R" train.py \
    task=Track algo=ppo task.drone_model.name=BlueROV \
    headless=true enable_livestream=false wandb.mode=offline \
    eval_only=true eval_episodes=${EP:-60} +eval_max_batches=${MB:-30} \
    task.env.num_envs=${NUM_ENVS:-32} seed=0 load_ckpt="$CKPT" \
    task.corridor.enable=true task.gust.enable=true \
    task.gust.speed="[$lo,$hi]" \
    task.safety.risk.enable=false task.safety.mppi.enable=false \
    > "$LOG" 2>&1
  echo "  rc=$?"
done

echo
echo "==== PPO 违约率 vs 阵风幅值 ===="
/home/jovyan/envs/sim/bin/python - <<'PY'
import re, glob, os
rows = []
for f in sorted(glob.glob('/home/jovyan/MarineGym-flow/scripts/outputs_flow/calib/ppo__v*.log')):
    txt = open(f, errors='replace').read()
    i = txt.rfind('=== EVAL-ONLY RESULTS')
    if i < 0:
        rows.append((os.path.basename(f), None, None, None)); continue
    tail = txt[i:]
    g = lambda k: (float(m.group(1)) if (m := re.search(rf'{re.escape(k)}:\s*(-?[\d.]+)', tail)) else float('nan'))
    tag = os.path.basename(f)[5:-4]
    rows.append((tag, g('stats.wall_violation'), g('stats.min_wall_dist'), g('stats.tracking_error_ema')))
print(f"{'gust':<12}{'violation':>11}{'min_wall':>11}{'rmse':>9}   判定")
for tag, v, mw, rm in rows:
    if v is None:
        print(f"{tag:<12}{'(未完成)':>11}"); continue
    verdict = '可用' if 0.10 <= v <= 0.60 else ('太易' if v < 0.10 else '太难')
    print(f"{tag:<12}{v:>11.3f}{mw:>11.3f}{rm:>9.3f}   {verdict}")
PY
