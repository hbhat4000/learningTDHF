#!/bin/bash
#SBATCH -A m4577_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 23:59:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

module load cudnn/8.9.3_cuda12

python fitHB.py --gpu 0 --mol c2h4 --basis sto-3g --train ./trajdata100/train_c2h4_sto-3g.npy --ntrain 200000 --dt 0.0008268 --tol 1e-16 --maxiter 100000 --outfname ./HBthetas100/theta_lowtol_c2h4_sto-3g.npz > ./logsHB/c2h4_lowtol_sto-3g_fit2.out
