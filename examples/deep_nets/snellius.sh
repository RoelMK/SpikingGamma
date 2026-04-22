#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=sgamma
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=./runs/slurm_output_%A.out

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

mkdir -p /scratch-shared/$USER/sgamma-data/data
mkdir -p /scratch-shared/$USER/sgamma-data/cache

# Start Python
cd $HOME/SpikingGamma
source venv/bin/activate
cd $HOME/SpikingGamma/examples/deep_nets
srun python run_dvsgesture.py \
  --data-path /scratch-shared/$USER/sgamma-data/data \
  --cache-path /scratch-shared/$USER/sgamma-data/cache \
  --epochs 1