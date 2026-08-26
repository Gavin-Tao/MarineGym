#!/usr/bin/env bash
# 门控边界 soft_hi 的敏感性扫描（ours 格，跨场景）。
#
# 为什么需要：走廊 margin=0.8，而策略的跟踪 RMSE≈0.34 m，所以**普通跟踪误差本身**
# 就能把裕度压到 0.46 < soft_hi=0.6 —— 门控在无扰动时也开了 6.5%，带来约 +14% 的
# 标称跟踪代价。边界定得太靠外，介入预算花在了不需要的时刻。
#
# 这个扫描给出三条曲线随 soft_hi 的变化：违约率 ↓ / 标称跟踪代价 ↑ / 接管占空比，
# 用来在证据上选工作点，而不是沿用论文①的 0.6。
#
#   CKPT=... bash run_gate_sweep.sh
set -o pipefail
CKPT=${CKPT:?需要 CKPT}
export CKPT
S=/home/jovyan/MarineGym-flow/scripts
R=$S/run_flow.sh
OUT=/home/jovyan/MarineGym-flow/scripts/outputs_flow/gate
mkdir -p "$OUT"
HIS=${HIS:-"0.20 0.30 0.45 0.60 0.80"}
SCENES=${SCENES:-"calm strong"}

for sc in $SCENES; do
  case "$sc" in
    calm)   SC="task.gust.enable=false" ;;
    nominal) SC="task.gust.enable=true" ;;
    strong) SC="task.gust.enable=true task.gust.speed=[1.5,2.2]" ;;
    fast)   SC="task.gust.enable=true task.gust.speed=[1.5,2.2] task.gust.ramp=[0.08,0.15]" ;;
  esac
  for hi in $HIS; do
    LOG="$OUT/ours__${sc}__hi${hi}.log"
    grep -q 'EVAL-ONLY RESULTS' "$LOG" 2>/dev/null && { echo "[skip] $sc hi=$hi"; continue; }
    echo "[$(date '+%H:%M:%S')] gate sweep  $sc  soft_hi=$hi"
    timeout 5400 bash "$R" train.py \
      task=Track algo=ppo task.drone_model.name=BlueROV \
      headless=true enable_livestream=false wandb.mode=offline \
      eval_only=true eval_episodes=500 +eval_max_batches=${MB:-12} \
      task.env.num_envs=${NUM_ENVS:-128} seed=0 load_ckpt="$CKPT" \
      task.corridor.enable=true $SC \
      task.safety.risk.enable=true task.safety.risk.gate=true \
      task.safety.mppi.enable=true task.safety.mppi.soft=true \
      task.safety.mppi.num_samples=${MPPI_K:-64} task.safety.mppi.horizon=${MPPI_N:-10} \
      task.safety.risk.threshold=$hi task.safety.mppi.soft_hi=$hi \
      +ep_dump="${LOG%.log}.episodes.csv" > "$LOG" 2>&1
    echo "  rc=$?"
  done
done

echo
echo "==== soft_hi 敏感性 ===="
/home/jovyan/envs/sim/bin/python - <<'PY'
import re, glob, os
rows = []
for f in sorted(glob.glob('/home/jovyan/MarineGym-flow/scripts/outputs_flow/gate/ours__*.log')):
    txt = open(f, errors='replace').read()
    i = txt.rfind('=== EVAL-ONLY RESULTS')
    base = os.path.basename(f)[:-4]
    if i < 0:
        rows.append((base, None, None, None)); continue
    tail = txt[i:]
    g = lambda k: (float(m.group(1)) if (m := re.search(rf'{re.escape(k)}:\s*(-?[\d.]+)', tail)) else float('nan'))
    rows.append((base, g('stats.wall_violation'), g('stats.tracking_error_ema'), g('stats.filter_lambda')))
print(f"{'run':<28}{'violation':>11}{'rmse':>9}{'lambda':>9}")
for b, v, r, l in rows:
    if v is None:
        print(f"{b:<28}{'(未完成)':>11}"); continue
    print(f"{b:<28}{v:>11.4f}{r:>9.4f}{l:>9.4f}")
PY
