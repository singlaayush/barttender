#!/bin/bash

experiments=()
orders=()

# Define the strings for the experiments and order
#experiments=("xray" "noise" "blank")
#orders=('"[0,1,2]"' '"[0,2,1]"' '"[1,0,2]"' '"[1,2,0]"' '"[2,1,0]"' '"[2,0,1]"')

for experiment in "${experiments[@]}"; do
    for order in "${orders[@]}"; do
        echo "Running box plot script with image_type: $experiment"
        python chexpert_plot.py --image_type=$experiment --order=$order
    done
done
