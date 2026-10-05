#!/bin/bash
#SBATCH --job-name=fact-ft
#SBATCH --partition=dheld
#SBATCH --nodelist=grogu-4-23
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/project_data/held/vxiao/Bimanual_Manipulation_Project/FACT/eval_runs/fact_ft-%j.out

# Single-GPU fine-tune of the released fact-wam ckpt on beat_block_hammer.
#   FT_NAME=expert|expert_fail   ROBOTWIN_DATASET_IDS=...   FT_FAIL_MULT=...   FT_MAX_STEPS=...
set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate fact
cd /project_data/held/vxiao/Bimanual_Manipulation_Project/FACT
export FACT_PROJECT_DIR=./experiments/ft_${FT_NAME}
export ROBOTWIN_ROOT=$PWD/datasets/RoboTwin
# dataset ids contain a comma, so they cannot be passed through sbatch --export; derive from FT_NAME
case "$FT_NAME" in
  expert)      export ROBOTWIN_DATASET_IDS="Clean/beat_block_hammer" ;;
  expert_fail) export ROBOTWIN_DATASET_IDS="Clean/beat_block_hammer,Fail/beat_block_hammer" ;;
  *) echo "unknown FT_NAME=$FT_NAME"; exit 2 ;;
esac
export ROBOTWIN_NORM_PATH=$PWD/models/fact-wam/norm_stats_delta.json   # must match the released ckpt (value range up to 2.0)
export FT_INIT_CKPT=$PWD/models/fact-wam/transformer
export TRITON_CACHE_DIR=/scratch/vxiao/tmp/triton
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
echo "FT_NAME=$FT_NAME ids=$ROBOTWIN_DATASET_IDS fail_mult=${FT_FAIL_MULT:-} steps=${FT_MAX_STEPS:-} batch=${FT_BATCH:-}"
python -m scripts.train --config world_action_model.configs.robotwin_ft.config
echo "=== train rc=$? ==="
