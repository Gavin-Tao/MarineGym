#!/usr/bin/env bash
# 沿训练轨迹评测多个 checkpoint，与训练当时报出的指标逐点对照。
#
# 训练曲线（nominal 阵风，train/stats.tracking_error_ema）：
#   iter  93 → 1.587    iter 207 → 0.415    iter 415 → 0.136    iter 510 → 0.224
# 若评测在早期点能对上、后期对不上 → 差距随训练增长，指向训练指标在后期变得乐观；
# 若各点都差一个相近的倍数 → 评测路径存在系统性偏差。
set -o pipefail
D=${D:-/home/jovyan/MarineGym-flow/scripts/wandb/offline-run-20260825_232340-sdpmr5h2/files}
OUT=/home/jovyan/MarineGym-flow/scripts/outputs_flow/diag
mkdir -p "$OUT"
R=/home/jovyan/MarineGym-flow/scripts/run_flow.sh

for fr in ${FRAMES:-"827392 1703936 3407872 4104192"}; do
  CK="$D/checkpoint_${fr}.pt"
  [ -f "$CK" ] || { echo "缺 $CK"; continue; }
  LOG="$OUT/ck_${fr}.log"
  if grep -q 'EVAL-ONLY RESULTS' "$LOG" 2>/dev/null; then echo "[skip] $fr"; continue; fi
  echo "[$(date '+%H:%M:%S')] eval checkpoint_${fr}  (iter≈$((fr/8192)))"
  timeout 3600 bash "$R" train.py \
    task=Track algo=ppo task.drone_model.name=BlueROV \
    headless=true enable_livestream=false wandb.mode=offline \
    eval_only=true eval_episodes=500 +eval_max_batches=12 \
    task.env.num_envs=128 seed=0 load_ckpt="$CK" \
    task.corridor.enable=true task.gust.enable=true \
    task.safety.risk.enable=false task.safety.mppi.enable=false \
    > "$LOG" 2>&1
  echo "  rc=$?"
done

echo
echo "==== checkpoint 沿训练轨迹的评测值 vs 训练当时报出的值 ===="
/home/jovyan/envs/sim/bin/python - <<'PY'
import re, glob, os
import pandas as pd
cur = pd.read_csv('/home/jovyan/MarineGym-flow/scripts/outputs_flow/train/curve_ppo_s0.csv')
print(f"{'iter':>6}{'eval_rmse':>11}{'train_rmse':>12}{'eval_viol':>11}{'train_viol':>12}{'eval_eplen':>12}{'train_eplen':>12}")
for f in sorted(glob.glob('/home/jovyan/MarineGym-flow/scripts/outputs_flow/diag/ck_*.log'),
                key=lambda p: int(os.path.basename(p)[3:-4])):
    fr = int(os.path.basename(f)[3:-4]); it = fr // 8192
    txt = open(f, errors='replace').read()
    i = txt.rfind('=== EVAL-ONLY RESULTS')
    if i < 0:
        print(f"{it:>6}{'(未完成)':>11}"); continue
    tail = txt[i:]
    g = lambda k: (float(m.group(1)) if (m := re.search(rf'{re.escape(k)}:\s*(-?[\d.]+)', tail)) else float('nan'))
    j = (cur['iter'] - it).abs().idxmin()          # 训练曲线上最接近的一点
    tr = cur.loc[j]
    print(f"{it:>6}{g('stats.tracking_error_ema'):>11.3f}"
          f"{tr['train/stats.tracking_error_ema']:>12.3f}"
          f"{g('stats.wall_violation'):>11.3f}{tr['train/stats.wall_violation']:>12.3f}"
          f"{g('stats.episode_len'):>12.1f}{tr['train/stats.episode_len']:>12.1f}")
PY
