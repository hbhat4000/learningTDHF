#!/bin/bash
#SBATCH -A m4577_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 23:59:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

python gentraj.py --gpu 0 --mol heh+ --basis 6-311G --field off --theta custom --custom ./EHthetas100/theta_heh+_6-311G.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100/testEH_heh+_6-311G.npy

python gentraj.py --gpu 0 --mol heh+ --basis 6-311G --field off --theta custom --custom ./EHthetas100/thetaR_heh+_6-311G.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100/testEHR_heh+_6-311G.npy

python gentraj.py --gpu 0 --mol heh+ --basis 6-311G --field on --theta custom --custom ./EHthetas100/theta_heh+_6-311G.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100WF/testEH_heh+_6-311G.npy

python gentraj.py --gpu 0 --mol heh+ --basis 6-311G --field on --theta custom --custom ./EHthetas100/thetaR_heh+_6-311G.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100WF/testEHR_heh+_6-311G.npy
