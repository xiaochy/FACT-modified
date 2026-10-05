#!/bin/bash
#SBATCH --job-name=fact-train
#SBATCH --partition=dheld
#SBATCH --nodelist=grogu-4-23
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=/project_data/held/vxiao/Bimanual_Manipulation_Project/FACT/eval_runs/fact_train-%j.out

# Generic single-GPU FACT training from the Wan2.2 base. Parameters (env):
#   WORKDIR  code dir (FACT or the FACT-track3d worktree)    CONFIG  config module path (.config)
#   ROOT     dataset root relative to FACT/                  IDS     one dataset id (no commas, sbatch --export limitation)
#   NORM     norm stats json relative to FACT/               NAME    experiment name (experiments/<NAME>)
#   PBS_MAX_STEPS / PBS_BATCH / PBS_CKPT_EVERY               FACT_TRACK_STATS (track config only)
set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate fact
FACT=/project_data/held/vxiao/Bimanual_Manipulation_Project/FACT
cd ${WORKDIR:-$FACT}
export ROBOTWIN_ROOT=$FACT/${ROOT:?ROOT required}
export ROBOTWIN_DATASET_IDS=${IDS:?IDS required}
export ROBOTWIN_NORM_PATH=$FACT/${NORM:?NORM required}
export FACT_PROJECT_DIR=./experiments/${NAME:?NAME required}
export TRITON_CACHE_DIR=/scratch/vxiao/tmp/triton
[ -n "${FACT_TRACK_STATS:-}" ] && export FACT_TRACK_STATS=$FACT/$FACT_TRACK_STATS
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
echo "workdir=$PWD config=${CONFIG:-world_action_model.configs.robotwin_pbs.config} root=$ROBOTWIN_ROOT ids=$ROBOTWIN_DATASET_IDS name=$NAME steps=${PBS_MAX_STEPS:-} batch=${PBS_BATCH:-}"
python -m scripts.train --config ${CONFIG:-world_action_model.configs.robotwin_pbs.config}
echo "=== train rc=$? ==="
