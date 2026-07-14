#!/bin/bash
# E1: 多 seed 统计验证 (10 test seeds × 2 模型)
set +e
PYTHON=/home/jovyan/envs/sim/bin/python
SCRIPT_DIR=/home/jovyan/MarineGym/scripts
OUTDIR=/home/jovyan/MarineGym/scripts/outputs_e1
mkdir -p "$OUTDIR"

T1_CKPT=/home/jovyan/MarineGym/scripts/wandb/offline-run-20260711_094733-ck77bi9a/files/checkpoint_final.pt
T2_CKPT=/home/jovyan/MarineGym/scripts/wandb/offline-run-20260711_104822-0hsw6qx5/files/checkpoint_final.pt

BASE_COMMON="task=Track algo=ppo task.drone_model.name=BlueROV task.traj_scale_mult=2.5 headless=true enable_livestream=false wandb.mode=offline eval_only=true eval_episodes=32"
T1_EXTRA="task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true"
T2_EXTRA="task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true task.keepout.mppi.enable=true task.keepout.mppi.exact=true task.keepout.mppi.num_samples=8 task.keepout.mppi.horizon=6 task.keepout.risk.enable=true task.keepout.risk.horizon=20 task.keepout.mppi.soft_blend=true task.keepout.mppi.soft_hi=0.6"

for seed in 1 2 3 4 5 6 7 8 9 10; do
    echo "========== $(date '+%H:%M:%S') E1-PPO seed=$seed =========="
    cd "$SCRIPT_DIR"
    WANDB_MODE=offline $PYTHON train.py \
        $BASE_COMMON $T1_EXTRA seed=$seed load_ckpt="$T1_CKPT" \
        > "$OUTDIR/e1_ppo_seed${seed}.log" 2>&1
    echo "PPO seed=$seed DONE (exit=$?)"
    
    echo "========== $(date '+%H:%M:%S') E1-ABsoft seed=$seed =========="
    cd "$SCRIPT_DIR"
    WANDB_MODE=offline $PYTHON train.py \
        $BASE_COMMON $T2_EXTRA seed=$seed load_ckpt="$T2_CKPT" \
        > "$OUTDIR/e1_absoft_seed${seed}.log" 2>&1
    echo "ABsoft seed=$seed DONE (exit=$?)"
done

echo ""
echo "========== $(date '+%H:%M:%S') E1 ALL DONE =========="
