#!/bin/bash
#SBATCH -A m4577_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 23:59:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"

python gentrajPG.py --gpu 2 --mol lih --basis 6-31g --field off --theta custom --custom ./PGthetas100/theta_lowtol_lih_6-31g.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100/testPG_lowtol_lih_6-31g.npy

python gentrajPG.py --gpu 2 --mol lih --basis 6-31g --field on --theta custom --custom ./PGthetas100/theta_lowtol_lih_6-31g.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100WF/testPG_lowtol_lih_6-31g.npy

python gentrajPG.py --gpu 2 --mol lih --basis 6-31g --field off --theta custom --custom ./PGthetas100/thetaR_lowtol_lih_6-31g.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100/testPGR_lowtol_lih_6-31g.npy

python gentrajPG.py --gpu 2 --mol lih --basis 6-31g --field on --theta custom --custom ./PGthetas100/thetaR_lowtol_lih_6-31g.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100WF/testPGR_lowtol_lih_6-31g.npy

