#!/bin/bash
#SBATCH -A m4577_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 23:59:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"

python gentraj.py --gpu 2 --mol c2h4 --basis 6-31pgs --field off --theta true --dt 0.0008268 --nsteps 200000 --outfname ./newtrajdata100/train_c2h4_6-31pgs.npy

python gentraj.py --gpu 2 --mol c2h4 --basis 6-31pgs --field on --theta true --dt 0.0008268 --nsteps 200000 --outfname ./newtrajdata100WF/train_c2h4_6-31pgs.npy
