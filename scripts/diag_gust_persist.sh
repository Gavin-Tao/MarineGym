#!/usr/bin/env bash
# 验证阵风 reset 计时 bug 是否修好。
#
# bug：IsaacEnv 先 `_reset_idx()` 再把 progress_buf 清零(isaac_env.py:275/277)，
# 所以 reset 时发射阵风读到的是上一 episode 的末值(≈599)，_gust_t0/_gust_next 被排到
# 新 episode 永远追不上的未来 → **第一个 episode 之后再也不发阵风**。
#
# 判据：采 1500 步（跨 ≥2 次 reset），把每步的流速按 episode 序号分组统计。
# 修好之后，第 2、3 个 episode 的阵风活跃度应与第 1 个相当；没修则后面全是 0。
set -o pipefail
OUT=/home/jovyan/MarineGym-flow/scripts/outputs_flow/diag
mkdir -p "$OUT"
timeout 3600 bash /home/jovyan/MarineGym-flow/scripts/run_flow.sh train.py \
  task=Track algo=ppo task.drone_model.name=BlueROV \
  headless=true enable_livestream=false wandb.mode=offline seed=0 \
  task.env.num_envs=${NUM_ENVS:-16} \
  task.corridor.enable=true task.gust.enable=true \
  task.safety.risk.enable=false task.safety.mppi.enable=false \
  +save_traj="$OUT/gust_persist.npz" +traj_steps=${STEPS:-1500} \
  > "$OUT/gust_persist.log" 2>&1
echo "rc=$?"
/home/jovyan/envs/sim/bin/python - <<'PY'
import numpy as np
d = np.load('/home/jovyan/MarineGym-flow/scripts/outputs_flow/diag/gust_persist.npz')
flow, done = d['flow'][:, :, 1], d['done']          # [T,E] 受限轴流速 / done
T, E = flow.shape
ep = np.cumsum(np.vstack([np.zeros((1, E), bool), done[:-1]]), axis=0)   # 每步属于第几个 episode
print(f"{'episode':>8}{'步数':>8}{'阵风活跃占比':>14}{'|流速|均值':>12}{'|流速|max':>11}")
for k in range(int(ep.max()) + 1):
    m = ep == k
    if m.sum() < 50:
        continue
    a = np.abs(flow[m])
    print(f"{k:>8}{m.sum():>8}{(a > 1e-6).mean():>14.3f}{a.mean():>12.3f}{a.max():>11.3f}")
print("\n判据：第 1 个之后的 episode 若活跃占比仍与第 0 个相当 → 已修；若掉到 0 → 未修。")
PY
