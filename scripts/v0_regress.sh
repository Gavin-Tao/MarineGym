#!/usr/bin/env bash
# V0 回归：默认配置（corridor/gust/keepout 全关）下，当前代码是否与参考 rollout 逐位一致。
# 参考 outputs_flow/regress/v0_ref.npz 是在动洋流代码**之前**采的，实测复现噪声底为 0，
# 所以判据是严格相等。任何影响动力学或奖励的改动都会在这里以非零差值暴露。
OUT=/home/jovyan/MarineGym-flow/scripts/outputs_flow/regress
NEW="$OUT/v0_new.npz"
rm -f "$NEW"
CUDA_VISIBLE_DEVICES=${GPU:-2} timeout 900 bash /home/jovyan/MarineGym-flow/scripts/run_flow.sh train.py \
  task=Track algo=ppo task.drone_model.name=BlueROV \
  headless=true enable_livestream=false wandb.mode=offline seed=0 \
  task.env.num_envs=16 \
  +save_traj="$NEW" +traj_steps=300 +const_action=0.3 \
  > "$OUT/v0_new.log" 2>&1
echo "capture rc=$?"
/home/jovyan/envs/sim/bin/python /home/jovyan/MarineGym-flow/scripts/flow_validate.py \
  v0 --ref "$OUT/v0_ref.npz" --new "$NEW"
