#!/bin/bash
#SBATCH --job-name=fact-pbs-video
#SBATCH --partition=dheld
#SBATCH --nodelist=grogu-4-23
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=/project_data/held/vxiao/Bimanual_Manipulation_Project/FACT/eval_runs/fact_train_pbs-%j.out

# FACT trained from the Wan2.2 base on place_bread_skillet (Clean 50 + Randomized 500 demos), single GPU.
# Waits for the VAE latent cache (compute_vae_latents) to finish, then trains.
set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
conda activate fact
cd /project_data/held/vxiao/Bimanual_Manipulation_Project/FACT
D=datasets/RoboTwin
echo "waiting for VAE latents (Clean 50, Randomized 500)..."
for i in $(seq 1 360); do
  c=$(ls $D/Clean/place_bread_skillet/vae_latents 2>/dev/null | grep -c '^episode_')
  r=$(ls $D/Randomized/place_bread_skillet/vae_latents 2>/dev/null | grep -c '^episode_')
  [ "$c" -ge 50 ] && [ "$r" -ge 500 ] && break
  sleep 30
done
echo "latents: Clean=$c Randomized=$r"; [ "$c" -ge 50 ] && [ "$r" -ge 500 ] || { echo "latents incomplete, abort"; exit 3; }
sleep 60   # let the last files finish writing
export ROBOTWIN_ROOT=$PWD/$D
export ROBOTWIN_DATASET_IDS="Clean/place_bread_skillet,Randomized/place_bread_skillet"
export ROBOTWIN_NORM_PATH=$PWD/artifacts/pbs/norm_stats_delta.json
export FACT_PROJECT_DIR=./experiments/${PBS_NAME:-pbs_video}
export TRITON_CACHE_DIR=/scratch/vxiao/tmp/triton
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
echo "steps=${PBS_MAX_STEPS:-} batch=${PBS_BATCH:-} project=$FACT_PROJECT_DIR"
python -m scripts.train --config world_action_model.configs.robotwin_pbs.config
echo "=== train rc=$? ==="
