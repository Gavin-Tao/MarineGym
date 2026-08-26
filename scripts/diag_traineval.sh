#!/usr/bin/env bash
# 判决性诊断：训练路径 vs 评测路径，同一份 checkpoint、同一套配置。
#
# 现象：训练日志报 tracking_error_ema≈0.16 / ep_len≈594 / 违约 0.015，
#       而同配置的 eval_only 报 1.18 / 500 / 0.213 —— 差 7～14 倍。
#       连无扰动的 calm 档评测(0.341)都比"训练时开着阵风"(0.158)差一倍，
#       说明问题指向策略在评测路径下的行为，而不是环境难度。
#
# 做法：用**训练代码路径**从该 checkpoint 续训 25 个迭代（学习率照常，
# 25 迭代不足以显著改变策略），读 train/stats.*：
#   · 若 ≈0.16  → 环境与策略都正常，差异在 eval_only 路径
#   · 若 ≈1.18  → 当前代码的环境比原训练时更难（唯一环境改动是阵风 RNG）
set -o pipefail
CKPT=${CKPT:?需要 CKPT}
OUT=/home/jovyan/MarineGym-flow/scripts/outputs_flow/diag
mkdir -p "$OUT"
rm -f "$OUT/curve_diag.csv"
timeout 3600 bash /home/jovyan/MarineGym-flow/scripts/run_flow.sh train.py \
  task=Track algo=ppo task.drone_model.name=BlueROV \
  headless=true enable_livestream=false wandb.mode=offline \
  task.env.num_envs=${NUM_ENVS:-128} algo.train_every=${TRAIN_EVERY:-32} \
  max_iters=${ITERS:-25} save_interval=-1 seed=0 \
  task.corridor.enable=true task.gust.enable=true \
  task.safety.risk.enable=false task.safety.mppi.enable=false \
  algo.checkpoint_path="$CKPT" \
  +train_log="$OUT/curve_diag.csv" \
  > "$OUT/diag.log" 2>&1
echo "rc=$?"
/home/jovyan/envs/sim/bin/python - <<'PY'
import pandas as pd
try:
    d = pd.read_csv('/home/jovyan/MarineGym-flow/scripts/outputs_flow/diag/curve_diag.csv')
except Exception as e:
    print("没有曲线数据:", e); raise SystemExit
k = [c for c in d.columns if any(s in c for s in
     ('episode_len', 'tracking_error_ema', 'wall_violation', 'min_wall_dist'))]
print(d[['iter'] + k].to_string(index=False))
PY
