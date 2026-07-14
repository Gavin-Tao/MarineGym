#!/bin/bash
# TTE实验计划 Step 0: 训练5个模型 (GPU优化版 - 减少MPPI计算量)
set +e
PYTHON=/home/jovyan/envs/sim/bin/python
SCRIPT_DIR=/home/jovyan/MarineGym/scripts
OUTPUT_BASE=/home/jovyan/MarineGym/scripts/outputs_tte
mkdir -p "$OUTPUT_BASE"

# 公共参数 (max_iters=120 → ~30K frames, ~13min @60FPS)
COMMON="task=Track algo=ppo task.drone_model.name=BlueROV task.traj_scale_mult=2.5 algo.train_every=16 max_iters=120 save_interval=40 headless=true enable_livestream=false wandb.mode=offline seed=0"

run_one() {
    local label=$1; shift
    echo "========== $(date '+%H:%M:%S') $label START =========="
    cd "$SCRIPT_DIR"
    WANDB_MODE=offline $PYTHON train.py $COMMON "$@" > "$OUTPUT_BASE/${label}.log" 2>&1
    local rc=$?
    echo "========== $(date '+%H:%M:%S') $label DONE (exit=$rc) =========="
    return $rc
}

# T1: PPO baseline (不需要MPPI, 直接跑)
run_one "t1_ppo_baseline" \
    task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true

# MPPI参数 (轻量: num_samples=8, horizon=6, risk.horizon=20)
MPPI_LITE="task.keepout.mppi.num_samples=8 task.keepout.mppi.horizon=6 task.keepout.risk.horizon=20"
ABSOFT="task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true task.keepout.mppi.enable=true task.keepout.mppi.exact=true task.keepout.risk.enable=true task.keepout.mppi.soft_blend=true $MPPI_LITE"

# T2: A+B-soft (soft_hi=0.6)
run_one "t2_absoft_hi0.6" $ABSOFT

# T3: A+B-soft (soft_hi=0.3)
run_one "t3_absoft_hi0.3" $ABSOFT task.keepout.mppi.soft_hi=0.3

# T4: A+B-soft (soft_hi=0.9)
run_one "t4_absoft_hi0.9" $ABSOFT task.keepout.mppi.soft_hi=0.9

# T5: A+B-soft (soft_hi=1.2)
run_one "t5_absoft_hi1.2" $ABSOFT task.keepout.mppi.soft_hi=1.2

echo ""
echo "========== $(date '+%H:%M:%S') ALL TRAININGS DONE =========="
