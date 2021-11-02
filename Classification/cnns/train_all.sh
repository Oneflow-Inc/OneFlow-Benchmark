#!/bin/bash

for i in $(seq $2 $3); do
    for j in $(seq $5 $6); do
        if [ $i -eq 0 ]; then
            wait_time="0.0"
        else
            wait_time="$1e$i"
        fi
        if [ $j -eq 0 ]; then
            transfer_cost="0.0"
        else
            transfer_cost="$4e$j"
        fi
        echo "0.25" > CostRatioFile.txt
        echo "${wait_time}" >> CostRatioFile.txt
        echo "${transfer_cost}" >> CostRatioFile.txt
        ./train.sh ${7} > "${7}txt/${7}_${i}_${j}.txt"
    done
done
