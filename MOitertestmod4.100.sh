#!/bin/bash
#SBATCH -A m4577_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 23:59:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

python gentrajMO.py --gpu 0 --mol c2h4 --basis sto-3g --field off --theta custom --custom ./MOiterthetas100/theta_c2h4_sto-3g.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100/testMOiter_c2h4_sto-3g.npy

python gentrajMO.py --gpu 0 --mol c2h4 --basis sto-3g --field on --theta custom --custom ./MOiterthetas100/theta_c2h4_sto-3g.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100WF/testMOiter_c2h4_sto-3g.npy

python gentrajMO.py --gpu 0 --mol c2h4 --basis sto-3g --field off --theta custom --custom ./MOiterthetas100/thetaR_c2h4_sto-3g.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100/testMOiterR_c2h4_sto-3g.npy

python gentrajMO.py --gpu 0 --mol c2h4 --basis sto-3g --field on --theta custom --custom ./MOiterthetas100/thetaR_c2h4_sto-3g.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100WF/testMOiterR_c2h4_sto-3g.npy


