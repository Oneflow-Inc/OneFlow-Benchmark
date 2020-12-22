#!/bin/bash

#for RatioValue in 1 1e-2 1e2 1e-3 1e-1 1e1 ; do
for RatioValue in 1e-1 ; do
    echo "${RatioValue}" > CostRatioFile.txt
    ./train.sh > WithAutoParallel2AddComputationCost${RatioValue}.txt
done 