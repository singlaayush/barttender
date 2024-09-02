#!/bin/bash

orders=('"[0,1,2,3,4,5,6,7]"')

for order in "${orders[@]}"; do
    echo "Running sbatch script with order: $order"
    sbatch --export=experiment="xray",start="0",end="11423",order="$order",ALL sbatch_add_bar_mimic.sh
    sbatch --export=experiment="noise",start="0",end="11423",order="$order",ALL sbatch_add_bar_mimic.sh
    sbatch --export=experiment="blank",start="0",end="11423",order="$order",ALL sbatch_add_bar_mimic.sh     
done
