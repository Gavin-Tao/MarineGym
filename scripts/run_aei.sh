#!/usr/bin/env bash
# AEI 实验统一启动器
set -o pipefail
export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /opt/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/envs/sim
cd /home/jovyan/MarineGym/scripts
export WANDB_MODE=offline
exec python "$@"
