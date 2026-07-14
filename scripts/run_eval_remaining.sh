#!/bin/bash
# E2: cross-trajectory + E3b: risk.threshold scan + soft_lo scan + E6: perturbation
set +e
PYTHON=/home/jovyan/envs/sim/bin/python
SCRIPT_DIR=/home/jovyan/MarineGym/scripts
OUTDIR=/home/jovyan/MarineGym/scripts/outputs_eval
mkdir -p "$OUTDIR"

T2_CKPT=/home/jovyan/MarineGym/scripts/wandb/offline-run-20260711_104822-0hsw6qx5/files/checkpoint_final.pt

BASE="task=Track algo=ppo task.drone_model.name=BlueROV task.traj_scale_mult=2.5 headless=true enable_livestream=false wandb.mode=offline eval_only=true eval_episodes=32 seed=1"
T2_CFG="task.keepout.enable=true task.keepout.radius=0.8 task.keepout.dynamic.enable=true task.keepout.mppi.enable=true task.keepout.mppi.exact=true task.keepout.mppi.num_samples=8 task.keepout.mppi.horizon=6 task.keepout.risk.enable=true task.keepout.risk.horizon=20 task.keepout.mppi.soft_blend=true task.keepout.mppi.soft_hi=0.6"

run_eval() {
    local label=$1; shift
    echo "========== $(date '+%H:%M:%S') $label =========="
    cd "$SCRIPT_DIR"
    WANDB_MODE=offline $PYTHON train.py $BASE $T2_CFG "$@" load_ckpt="$T2_CKPT" > "$OUTDIR/${label}.log" 2>&1
    echo "$label DONE (exit=$?)"
}

# E3b: risk.threshold scan
for rt in 0.1 0.5; do
    run_eval "e3b_risk_thresh_${rt}" task.keepout.risk.threshold=$rt
done

# E3b: soft_lo scan
for sl in 0.1 0.2; do
    run_eval "e3b_soft_lo_${sl}" task.keepout.mppi.soft_lo=$sl
done

# E6: payload disturbance
run_eval "e6_payload" \
    task.disturbances.train.payload.enable_payload=true

# E6: flow disturbance
run_eval "e6_flow" \
    task.disturbances.train.flow.enable_flow=true

echo ""
echo "========== $(date '+%H:%M:%S') REMAINING EVAL DONE =========="
