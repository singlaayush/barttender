#!/bin/bash
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 6:00:00
#SBATCH --output=logs/mask_extraction/ig/%j_%x.out

source venv/bin/activate
python mimic_mask_extraction.py --image_type "$experiment" --run_id "$run_id" --batch_size=1 --mask_type='ig' --split='test'
