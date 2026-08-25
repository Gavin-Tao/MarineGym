#!/usr/bin/env bash
# K2 场景标定：纯 PPO 在各场景下的违约率是否落在可区分区间。
# 严格串行 —— 这台机器空闲显存只剩 ~2.5 GB，并发必 OOM。
set -o pipefail
export CKPT=${CKPT:?需要 CKPT}
S=/home/jovyan/MarineGym-flow/scripts
for sc in calm nominal strong fast; do
  NUM_ENVS=${NUM_ENVS:-32} EP=${EP:-60} MB=${MB:-30} bash "$S/exp_eval.sh" ppo "$sc" 0
done
/home/jovyan/envs/sim/bin/python "$S/flow_collect.py"
echo
echo "==== PPO 各场景违约率（K2 判据：落在 0.10–0.60 才有区分空间）===="
/home/jovyan/envs/sim/bin/python - <<'PY'
import pandas as pd
d = pd.read_csv('/home/jovyan/MarineGym-flow/scripts/outputs_flow/data/eval_raw.csv')
d = d[d.cell == 'ppo']
cols = [c for c in ('stats.wall_violation','stats.wall_viol_frac','stats.min_wall_dist',
                    'stats.tracking_error_ema','stats.episode_len') if c in d.columns]
print(d[['scene'] + cols].to_string(index=False))
PY
