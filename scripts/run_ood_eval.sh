#!/bin/bash
# OOD evaluation batch runner.
# After baseline training completes, run this to evaluate the policy
# on all OOD sets (train/b1/b2/b3/b4) with MPPI on and off.
# Usage: bash scripts/run_ood_eval.sh <checkpoint_path> [num_seeds]

CKPT=${1:-""}
SEEDS=${2:-20}  # default 20 seeds for quick eval; increase to 100 for final

if [ -z "$CKPT" ]; then
    # Try to find the latest checkpoint
    CKPT=$(ls -t wandb/offline-run-*/files/checkpoint_final.pt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "ERROR: No checkpoint found. Provide path as argument."
        echo "Usage: bash scripts/run_ood_eval.sh <checkpoint.pt> [seeds]"
        exit 1
    fi
fi

echo "=== OOD Evaluation Batch ==="
echo "Checkpoint: $CKPT"
echo "Seeds per config: $SEEDS"
echo ""

PYTHON=/home/jovyan/envs/sim/bin/python
SCRIPT=/home/jovyan/MarineGym/scripts/eval_ood.py
RESULTS_DIR=/tmp/ood_results
mkdir -p $RESULTS_DIR

for OOD in train b1 b2 b3 b4; do
    for MPPI in off coarse; do
        NAME="${OOD}_mppi-${MPPI}"
        LOG="$RESULTS_DIR/${NAME}.log"
        echo "[$(date +%H:%M:%S)] Starting $NAME ..."
        $PYTHON $SCRIPT \
            --ckpt "$CKPT" \
            --ood "$OOD" \
            --seeds "$SEEDS" \
            --mppi "$MPPI" \
            --output "$RESULTS_DIR/${NAME}.json" \
            > "$LOG" 2>&1
        RC=$?
        if [ $RC -ne 0 ]; then
            echo "  FAILED (exit $RC) — see $LOG"
        else
            echo "  DONE"
        fi
    done
done

echo ""
echo "=== Summary ==="
echo "Results in: $RESULTS_DIR/"
for f in $RESULTS_DIR/*.json; do
    if [ -f "$f" ]; then
        python3 -c "
import json
with open('$f') as fp:
    d = json.load(fp)
r = d['results']
print(f\"{d['ood']:5s} | mppi={d['mppi']:6s} | coll={r['collision']['mean']:.4f} | succ={r['success']['mean']:.4f} | min_d={r['min_obstacle_dist']['mean']:.3f} | track_e={r['tracking_error']['mean']:.4f}\")
"
    fi
done
