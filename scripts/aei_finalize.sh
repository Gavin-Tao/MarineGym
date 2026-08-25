#!/usr/bin/env bash
# AEI 收尾：等评测结束 → N1 模型精度 → N2 延迟 → C3 失配 → 汇总出图 → 提交
OUT=/home/jovyan/MarineGym/scripts/outputs_aei
R=/home/jovyan/MarineGym/scripts/run_aei.sh
W=/home/jovyan/MarineGym/scripts/wandb
LOG=$OUT/finalize.log
PY=/home/jovyan/envs/sim/bin/python
mkdir -p "$OUT/data" "$OUT/extra"
T2=$W/offline-run-20260711_165556-y7en5cfg/files/checkpoint_final.pt

BASE="task=Track algo=ppo task.drone_model.name=BlueROV task.traj_scale_mult=2.5 \
headless=true enable_livestream=false wandb.mode=offline"
EVB="$BASE eval_only=true eval_episodes=60 +eval_max_batches=40 task.env.num_envs=64"
KO="task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true"
ABS="$KO task.keepout.mppi.enable=true task.keepout.mppi.exact=true \
task.keepout.risk.enable=true task.keepout.mppi.soft_blend=true"

say(){ echo "[$(date '+%H:%M')] $*" >> "$LOG"; }

say "等待评测结束"
while ! grep -q "ALL EVAL DONE" "$OUT/eval_progress.log" 2>/dev/null; do sleep 120; done
say "评测结束，开始收尾"

# ---- N1 名义模型单步误差 ----
say "N1 开始"
timeout 1500 $R train.py $EVB seed=1 load_ckpt="$T2" $ABS \
  task.keepout.validate_full=true > "$OUT/extra/n1.log" 2>&1
say "N1 rc=$? (结果见 data/nominal_onestep.json)"

# ---- N2 计算延迟：K×H 扫描（短训练，读 rollout_fps）----
say "N2 开始"
for K in 32 64 128 256; do for H in 10 20; do
  f="$OUT/extra/lat_K${K}_N${H}.log"
  grep -q rollout_fps "$f" 2>/dev/null && continue
  timeout 900 $R train.py $BASE algo.train_every=16 max_iters=3 seed=0 $ABS \
    task.keepout.mppi.num_samples=$K task.keepout.mppi.horizon=$H > "$f" 2>&1
done; done
$PY - <<'PYEOF'
import glob,re,json
out={}
for f in glob.glob('/home/jovyan/MarineGym/scripts/outputs_aei/extra/lat_K*_N*.log'):
    m=re.search(r'lat_K(\d+)_N(\d+)',f); 
    if not m: continue
    v=re.findall(r'rollout_fps\s+([0-9.]+)', open(f,errors='replace').read())
    if not v: continue
    fps=float(v[-1]); E=16
    out[f"{m.group(1)}_{m.group(2)}"]=dict(rollout_fps=fps, num_envs=E,
                                           ms_per_step=1000.0*E/fps if fps>0 else None)
json.dump(out, open('/home/jovyan/MarineGym/scripts/outputs_aei/data/latency.json','w'), indent=2)
print('latency.json:', len(out), 'points')
PYEOF
say "N2 完成"

# ---- C3 模型失配敏感性 ----
say "C3 开始"
c3(){ # $1=tag  $2..=覆盖项
  local tag=$1; shift
  for s in 1 2 3; do
    f="$OUT/eval/c3-${tag}_ood_s${s}.log"
    grep -q 'EVAL-ONLY RESULTS' "$f" 2>/dev/null && continue
    timeout 1200 $R train.py $EVB seed=$s load_ckpt="$T2" $ABS \
      task.keepout.dynamic.speed=[2.2,2.8] "$@" > "$f" 2>&1
  done
}
c3 mass0.8   task.keepout.nom_scale_mass=0.8
c3 mass1.2   task.keepout.nom_scale_mass=1.2
c3 drag0.7   task.keepout.nom_scale_drag=0.7
c3 drag1.3   task.keepout.nom_scale_drag=1.3
c3 thrust0.8 task.keepout.nom_scale_thrust=0.8
c3 thrust1.2 task.keepout.nom_scale_thrust=1.2
say "C3 完成"

# ---- 汇总出图 ----
$PY /home/jovyan/MarineGym/scripts/aei_report.py >> "$LOG" 2>&1
say "报告与图表已生成"

cd /home/jovyan/MarineGym
git add -A scripts/outputs_aei/data scripts/outputs_aei/figures scripts/outputs_aei/RESULTS.md \
        scripts/aei_*.sh scripts/aei_*.py marinegym/envs/single/track.py 2>/dev/null
git -c user.email="taox@tcd.ie" -c user.name="taox" commit -q -m "AEI results: eval summary, significance tests, figures

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AJwybn9tFue7AS29GBDvhk" 2>>"$LOG"
git push origin main >>"$LOG" 2>&1
say "ALL DONE"
