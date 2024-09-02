#!/bin/bash
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 6:00:00
#SBATCH --job-name="add-bars"
#SBATCH --output="logs/add_bars/%j_%x.out"

module load python/3.10.11 gcc/9.2.0
source venv/bin/activate
python mimic_add_bar_preprocess.py --img_type=$experiment

#python mimic_add_bar_preprocess.py --img_type=$experiment --order=$order --start=$start --end=$end