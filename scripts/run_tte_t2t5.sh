#!/bin/bash
# TTE实验: T2-T5 (A+B-soft, 不同soft_hi)
# T1已完成, 跳过
# MPPI轻量配置: num_samples=8, horizon=6, risk.horizon=20
set +e
PYTHON=/home/jovyan/envs/sim/bin/python
SCRIPT_DIR=/home/jovyan/MarineGym/scripts
OUTPUT_BASE=/home/jovyan/MarineGym/scripts/outputs_tte
mkdir -p "$OUTPUT_BASE"

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

# MPPI参数: 轻量但保留exact精度
MPPI_LITE="task.keepout.mppi.num_samples=8 task.keepout.mppi.horizon=6 task.keepout.risk.horizon=20"
ABSOFT="task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true task.keepout.mppi.enable=true task.keepout.mppi.exact=true task.keepout.risk.enable=true task.keepout.mppi.soft_blend=true $MPPI_LITE"

# T2: A+B-soft (soft_hi=0.6) — 默认
run_one "t2_absoft_hi0.6" $ABSOFT

# T3: A+B-soft (soft_hi=0.3) — 保守
run_one "t3_absoft_hi0.3" $ABSOFT task.keepout.mppi.soft_hi=0.3

# T4: A+B-soft (soft_hi=0.9) — 激进
run_one "t4_absoft_hi0.9" $ABSOFT task.keepout.mppi.soft_hi=0.9

# T5: A+B-soft (soft_hi=1.2) — 几乎不接管
run_one "t5_absoft_hi1.2" $ABSOFT task.keepout.mppi.soft_hi=1.2

echo ""
echo "========== $(date '+%H:%M:%S') T2-T5 ALL DONE =========="
