#!/bin/bash
#SBATCH --job-name=fact-eval-ckpt
#SBATCH --partition=dheld
#SBATCH --nodelist=grogu-4-23
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/project_data/held/vxiao/Bimanual_Manipulation_Project/FACT/eval_runs/fact_evalckpt-%j.out

# Evaluate a checkpoint produced by fact_train_pbs.sbatch: PBS_NAME (project dir name), STEP (e.g. 24000),
# plus the usual TASK_NAME / TEST_NUM / SEED. Resolves the checkpoint dir by glob, then runs slurm/fact_eval.sbatch.
cd /project_data/held/vxiao/Bimanual_Manipulation_Project/FACT
CK=$(ls -d experiments/${PBS_NAME}/models/checkpoint_*_step_${STEP}/transformer 2>/dev/null | head -1)
[ -n "$CK" ] || { echo "no checkpoint for step ${STEP} under experiments/${PBS_NAME}"; exit 4; }
export TRANSFORMER_PATH=$PWD/$CK STATS_PATH=$PWD/${NORM_STATS:-artifacts/pbs/norm_stats_delta.json}
echo "checkpoint: $TRANSFORMER_PATH"
bash slurm/fact_eval.sbatch
