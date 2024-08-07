#!/bin/bash
#SBATCH -A m4577_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 23:59:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

python gentrajMO.py --gpu 0 --mol heh+ --basis 6-311G --field off --theta custom --custom ./MOthetas100/theta_heh+_6-311G.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100/testMO_heh+_6-311G.npy

python gentrajMO.py --gpu 0 --mol heh+ --basis 6-311G --field on --theta custom --custom ./MOthetas100/theta_heh+_6-311G.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100WF/testMO_heh+_6-311G.npy

python gentrajMO.py --gpu 0 --mol heh+ --basis 6-311G --field off --theta custom --custom ./MOthetas100/thetaR_heh+_6-311G.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100/testMOR_heh+_6-311G.npy

python gentrajMO.py --gpu 0 --mol heh+ --basis 6-311G --field on --theta custom --custom ./MOthetas100/thetaR_heh+_6-311G.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100WF/testMOR_heh+_6-311G.npy


