#!/bin/bash
#SBATCH -A m4577_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 23:59:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

module load cudnn/8.9.3_cuda12

python ./fitPG.py --gpu 0 --mol heh+ --basis 6-31g --train ./trajdata100/train_heh+_6-31g.npy --ntrain 200000 --dt 0.0008268 --tol 1e-16 --maxiter 100000 --outfname ./PGthetas100/theta_lowtol_heh+_6-31g.npz > ./logsPG/heh+_lowtol_6-31g_fit2.out
