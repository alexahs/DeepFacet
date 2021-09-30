#!/bin/bash
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=9_157905
#SBATCH --cpus-per-task=2

echo $CUDA_VISIBLE_DEVICES
mpirun -n 1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in run.lmp

# Replace 1 everywhere with the number of GPUs.
# binsize 7.5 is for SiO2 Vashishta, remove or adjust otherwise.

#NUM_RERUNS DEF
num_reruns=4

python3 copy.py $num_reruns
restart_file="pre_yield.restart"

i=1
while [ $i -le $num_reruns ]
do
    cd "rerun_$i"
    mpirun -n 1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in run.lmp
    let i++
    cd ../
done


# sim_dir=${PWD##*/}
# compl_dir="../completed_runs"
# if [[ ! -d "$compl_dir" ]]
# then
#     mkdir "$compl_dir"
# fi
# mv ../"$sim_dir" "$compl_dir/$sim_dir"
