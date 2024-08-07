#!/bin/bash
#SBATCH -A m2530_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 23:59:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"

python gentrajHB.py --gpu 0 --mol lih --basis 6-311ppgss --field off --theta custom --custom ./HBthetas100/theta_lowtol_lih_6-311ppgss.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100/testHB_lowtol_lih_6-311ppgss.npy

python gentrajHB.py --gpu 0 --mol lih --basis 6-311ppgss --field off --theta custom --custom ./HBthetas100/thetaR_lowtol_lih_6-311ppgss.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100/testHBR_lowtol_lih_6-311ppgss.npy

python gentrajHB.py --gpu 0 --mol lih --basis 6-311ppgss --field on --theta custom --custom ./HBthetas100/theta_lowtol_lih_6-311ppgss.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100WF/testHB_lowtol_lih_6-311ppgss.npy

python gentrajHB.py --gpu 0 --mol lih --basis 6-311ppgss --field on --theta custom --custom ./HBthetas100/thetaR_lowtol_lih_6-311ppgss.npz --npzkey x --dt 0.0008268 --nsteps 200000 --outfname ./testdata100WF/testHBR_lowtol_lih_6-311ppgss.npy

