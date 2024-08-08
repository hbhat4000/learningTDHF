#!/bin/bash
#SBATCH -A m4577_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 23:59:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

module load cudnn/8.9.3_cuda12

mols=("c2h4")
bases=("6-31pgs")
for i in ${!mols[@]}; do
  baseoutfname="theta_lowtol_${mols[$i]}_${bases[$i]}.npz"
  inpart="${mols[$i]}_${bases[$i]}.npy"
  # regular fits
  outfname="./HBthetas100/${baseoutfname}"
  logfname="./logsHB/${mols[$i]}_lowtol_${bases[$i]}_fit2.out"
  trainfname="./trajdata100/train_${mols[$i]}_${bases[$i]}.npy"
  python fitHB.py --gpu 0 --mol ${mols[$i]} --basis ${bases[$i]} --train ${trainfname} --ntrain 200000 --stride 10 --dt 0.0008268 --tol 1e-16 --maxiter 100000 --outfname ${outfname} > ${logfname}
done

