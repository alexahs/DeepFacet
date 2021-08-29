#!/bin/bash
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=sphere
#SBATCH --cpus-per-task=2

echo $CUDA_VISIBLE_DEVICES
mpirun -n 1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in run.lmp

# Replace 1 everywhere with the number of GPUs.
# binsize 7.5 is for SiO2 Vashishta, remove or adjust otherwise.
