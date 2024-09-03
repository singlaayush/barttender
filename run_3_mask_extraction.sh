#!/bin/bash

# Define the strings for the experiments and order
#experiments=("xray" "noise" "blank")
#run_ids=("wqzq428c" "jfpf2s6m" "yxa5wmrg")

experiments=("noise" "blank")
run_ids=("jfpf2s6m" "yxa5wmrg")

counter=0
for experiment in "${experiments[@]}"; do
    echo "Running sbatch script with image_type: $experiment"
    run_id=${run_ids[$counter]}
    #sbatch --job-name="sl-$experiment-mimic" --export=experiment="$experiment",run_id="$run_id",ALL sbatch_mask_extraction_saliency.sh
    #sbatch --job-name="ig-$experiment-mimic" --export=experiment="$experiment",run_id="$run_id",ALL sbatch_mask_extraction_ig.sh
    python mimic_mask_extraction.py --image_type="$experiment"  --run_id="$run_id" --batch_size=1 --mask_type='saliency' --idp
    python mimic_mask_extraction.py --image_type="$experiment"  --run_id="$run_id" --batch_size=1 --mask_type='ig' --idp
    ((counter++))
done
