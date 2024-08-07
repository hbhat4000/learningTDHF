#!/bin/bash
#SBATCH -A m4577_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 23:59:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

python gentrajHB.py --gpu 1 --mol heh+ --basis 6-311G --field off --theta custom --custom ./HBthetas100/theta_lowtol_heh+_6-311G.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100/testHB_lowtol_heh+_6-311G.npy

python gentrajHB.py --gpu 1 --mol heh+ --basis 6-311G --field off --theta custom --custom ./HBthetas100/thetaR_lowtol_heh+_6-311G.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100/testHBR_lowtol_heh+_6-311G.npy

python gentrajHB.py --gpu 1 --mol heh+ --basis 6-311G --field on --theta custom --custom ./HBthetas100/theta_lowtol_heh+_6-311G.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100WF/testHB_lowtol_heh+_6-311G.npy

python gentrajHB.py --gpu 1 --mol heh+ --basis 6-311G --field on --theta custom --custom ./HBthetas100/thetaR_lowtol_heh+_6-311G.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100WF/testHBR_lowtol_heh+_6-311G.npy
