#!/bin/bash
#SBATCH --job-name=fact-eval-pilot
#SBATCH --partition=dheld

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=/project_data/held/vxiao/Bimanual_Manipulation_Project/FACT/eval_runs/fact_eval_pilot-%j.out

# FACT released ckpt (Bariona/fact-wam) closed-loop eval on RoboTwin, server + client on one GPU.
set -o pipefail
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
cd /project_data/held/vxiao/Bimanual_Manipulation_Project/FACT
export ROBOTWIN_PATH=/project_data/held/vxiao/Bimanual_Manipulation_Project/RoboTwin
export TASK_NAME=${TASK_NAME:-beat_block_hammer} TASK_CONFIG=${TASK_CONFIG:-demo_clean} TEST_NUM=${TEST_NUM:-5}
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ${GPU_MEM} MiB on $(hostname)"
if [ "${GPU_MEM:-0}" -lt 20000 ]; then echo "GPU too small (<20GB) for the 5B model, abort"; exit 99; fi
if [ -z "${TEXT_ENCODER_DEVICE:-}" ]; then
  if [ "${GPU_MEM:-0}" -gt 40000 ]; then TEXT_ENCODER_DEVICE=cuda:0; else TEXT_ENCODER_DEVICE=cpu; fi   # 24GB cards cannot hold 5B model + T5
fi
export PYTHONUNBUFFERED=1 TEXT_ENCODER_DEVICE ENABLE_PREFIX_CACHE=${ENABLE_PREFIX_CACHE:-1} SKIP_FUTURE_STATE_VALUE=${SKIP_FUTURE_STATE_VALUE:-1} NUM_INFERENCE_STEPS=20
# unique port per job: jobs on one node share 127.0.0.1, a fixed port made clients talk to the wrong server
export PORT=${PORT:-$((20000 + SLURM_JOB_ID % 20000))}
export SERVER_WAIT_SECONDS=900 SERVER_TIMEOUT_MS=${SERVER_TIMEOUT_MS:-600000} TRITON_CACHE_DIR=/scratch/vxiao/tmp/triton
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
bash evaluation/robotwin/launch_server.sh > eval_runs/server-${SLURM_JOB_ID}.log 2>&1 &
SERVER_PID=$!
bash evaluation/robotwin/launch_client.sh "$TASK_NAME" "$TASK_CONFIG"
rc=$?
echo "=== client rc=$rc ==="
kill $SERVER_PID 2>/dev/null; pkill -P $SERVER_PID 2>/dev/null
exit $rc
