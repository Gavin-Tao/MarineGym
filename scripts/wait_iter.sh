#!/usr/bin/env bash
# 等训练曲线到达指定迭代数后返回。用法: wait_iter.sh <csv> <target>
CSV=${1:?csv}
TGT=${2:?target}
PY=/home/jovyan/envs/sim/bin/python
while :; do
  cur=$("$PY" - "$CSV" <<'PY'
import sys, pandas as pd
try:
    print(int(pd.read_csv(sys.argv[1])["iter"].iloc[-1]))
except Exception:
    print(0)
PY
)
  [ "${cur:-0}" -ge "$TGT" ] && { echo "达到 iter=$cur"; exit 0; }
  # 训练进程没了就别再等
  pgrep -f "outputs_flow/train" >/dev/null 2>&1 || { echo "训练进程已结束 (iter=$cur)"; exit 0; }
  sleep 60
done
