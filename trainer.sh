#!/bin/zsh
for weight in 0.0001 0.001 0.01 0.1 1 10
do
    sbatch launch_jobs.sh ${weight}
done