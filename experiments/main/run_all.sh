#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=run_all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=144:00:00
#SBATCH --array=0-8
#SBATCH --output=/dev/null

# Usage:
#   sbatch experiments/main/run_all.sh       # submit array
#   bash   experiments/main/run_all.sh       # run task 0 locally (set SLURM_ARRAY_TASK_ID to test others)

set -eo pipefail

MODELS=(euclidean hyperbolic hypercore)
DATASETS=(cifar10 cifar100 tiny_imagenet)

SIZE=base
AUGMENTATION=advanced
EPOCHS=100
BATCH_SIZE=128
CURVATURE=1.0

# Map task ID → model / dataset  (model 0-2, dataset 0-2)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$((TASK_ID / 3))]}
DATASET=${DATASETS[$((TASK_ID % 3))]}

if [[ "$DATASET" == "tiny_imagenet" ]]; then PATCH_SIZE=8; else PATCH_SIZE=4; fi

declare -A CONFIGS
CONFIGS[base]="--num_hidden_layers 12 --hidden_size 768 --num_attention_heads 12 --mlp_dim 3072"

declare -A HYPERCORE_CONFIGS
HYPERCORE_CONFIGS[base]="--num_hidden_layers 12 --hidden_size 780 --num_attention_heads 12 --mlp_dim 3120"

if [[ "$MODEL" == "hypercore" ]]; then
    MODEL_CONFIG="${HYPERCORE_CONFIGS[$SIZE]}"
else
    MODEL_CONFIG="${CONFIGS[$SIZE]}"
fi

CHECKPOINT_DIR="checkpoints/${AUGMENTATION}/${DATASET}/${MODEL}_${SIZE}"
mkdir -p "$CHECKPOINT_DIR"

# Redirect all output to the checkpoint dir so logs live next to weights
exec > "${CHECKPOINT_DIR}/slurm_${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}.out" 2>&1

module purge
module load 2024
module load Anaconda3/2024.06-1

source activate hvit
export PYTHONPATH="$(pwd):$(pwd)/experiments/main"

echo "MODEL=${MODEL} SIZE=${SIZE} DATASET=${DATASET} AUGMENTATION=${AUGMENTATION}"
srun python -u experiments/main/train.py \
  --model "${MODEL}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --dataset "${DATASET}" \
  --patch_size "${PATCH_SIZE}" \
  --curvature "${CURVATURE}" \
  --checkpoint_dir "${CHECKPOINT_DIR}/" \
  --augmentation "${AUGMENTATION}" \
  ${MODEL_CONFIG}
