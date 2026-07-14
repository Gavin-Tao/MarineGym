#!/bin/bash
# TTE实验计划 Step 0: 训练5个模型
# T1: PPO baseline | T2: A+B-soft (soft_hi=0.6) | T3-T5: soft_hi variants

set -e
PYTHON=/home/jovyan/envs/sim/bin/python
SCRIPT_DIR=/home/jovyan/MarineGym/scripts
OUTPUT_BASE=/home/jovyan/MarineGym/scripts/outputs_tte
mkdir -p "$OUTPUT_BASE"

COMMON="task=Track algo=ppo task.drone_model.name=BlueROV task.traj_scale_mult=2.5 algo.train_every=16 max_iters=120 save_interval=40 headless=true enable_livestream=false wandb.mode=offline seed=0"

# T1: PPO baseline (obstacles present, but NO filter — policy must handle alone)
echo "========== T1: PPO baseline =========="
cd "$SCRIPT_DIR"
WANDB_MODE=offline $PYTHON train.py \
    $COMMON task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true \
    > "$OUTPUT_BASE/t1_ppo_baseline.log" 2>&1
echo "T1 DONE (exit=$?)"

# Common flags for A+B-soft
ABSOFT="task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true task.keepout.mppi.enable=true task.keepout.mppi.exact=true task.keepout.mppi.num_samples=24 task.keepout.mppi.horizon=15 task.keepout.risk.enable=true task.keepout.risk.horizon=50 task.keepout.mppi.soft_blend=true"

# T2: A+B-soft (default soft_hi=0.6)
echo "========== T2: A+B-soft (soft_hi=0.6) =========="
cd "$SCRIPT_DIR"
WANDB_MODE=offline $PYTHON train.py \
    $COMMON $ABSOFT \
    > "$OUTPUT_BASE/t2_absoft_hi0.6.log" 2>&1
echo "T2 DONE (exit=$?)"

# T3: A+B-soft (soft_hi=0.3)
echo "========== T3: A+B-soft (soft_hi=0.3) =========="
cd "$SCRIPT_DIR"
WANDB_MODE=offline $PYTHON train.py \
    $COMMON $ABSOFT task.keepout.mppi.soft_hi=0.3 \
    > "$OUTPUT_BASE/t3_absoft_hi0.3.log" 2>&1
echo "T3 DONE (exit=$?)"

# T4: A+B-soft (soft_hi=0.9)
echo "========== T4: A+B-soft (soft_hi=0.9) =========="
cd "$SCRIPT_DIR"
WANDB_MODE=offline $PYTHON train.py \
    $COMMON $ABSOFT task.keepout.mppi.soft_hi=0.9 \
    > "$OUTPUT_BASE/t4_absoft_hi0.9.log" 2>&1
echo "T4 DONE (exit=$?)"

# T5: A+B-soft (soft_hi=1.2)
echo "========== T5: A+B-soft (soft_hi=1.2) =========="
cd "$SCRIPT_DIR"
WANDB_MODE=offline $PYTHON train.py \
    $COMMON $ABSOFT task.keepout.mppi.soft_hi=1.2 \
    > "$OUTPUT_BASE/t5_absoft_hi1.2.log" 2>&1
echo "T5 DONE (exit=$?)"

echo ""
echo "========== ALL TRAININGS DONE =========="
echo "Checkpoints in $SCRIPT_DIR/wandb/"
echo "Logs in $OUTPUT_BASE/"
