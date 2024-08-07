#!/bin/bash
#SBATCH -A m4577_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 23:59:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"

python gentraj.py --gpu 3 --mol c6h6n2o2 --basis sto-3g --field off --theta true --dt 0.0008268 --nsteps 200000 --outfname ./trajdata100/train_c6h6n2o2_sto-3g.npy 

python gentraj.py --gpu 3 --mol c6h6n2o2 --basis sto-3g --field on --theta true --dt 0.0008268 --nsteps 200000 --outfname ./trajdata100WF/train_c6h6n2o2_sto-3g.npy 

