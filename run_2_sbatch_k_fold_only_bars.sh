#!/bin/bash
#SBATCH -p gpu_requeue
#SBATCH -c 4
#SBATCH --gres=gpu:1,vram:40GB
#SBATCH --mem=32G
#SBATCH -t 5:00:00
#SBATCH --requeue
#SBATCH --job-name="k-cv-only-bars"
#SBATCH --output="logs/k_fold/%j_%x.out"
#SBATCH --signal=SIGUSR1@90

module load python/3.10.11 gcc/9.2.0 cuda/12.1
source venv/bin/activate
srun python mimic_k_fold.py --num_workers=3 --idp --only-bars
