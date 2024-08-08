#!/bin/bash
#SBATCH -A m2530_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 23:59:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"

mols=("c2h4")
bases=("6-31pgs")
for i in ${!mols[@]}; do
  baseoutfname="thetaR_lowtol_${mols[$i]}_${bases[$i]}.npz"
  inpart="${mols[$i]}_${bases[$i]}.npy"
  # PG fits
  outfname="./PGthetas100/${baseoutfname}"
  logfname="./logsPG/${mols[$i]}R_lowtol_${bases[$i]}_fit2.out"
  paggfname="./trajdata100R/one_p_${inpart}"
  pdotaggfname="./trajdata100R/one_pdot_${inpart}"
  python fitPG.py --gpu 0 --mol ${mols[$i]} --basis ${bases[$i]} --pagg ${paggfname} --pdotagg ${pdotaggfname} --dt 0.0008268 --tol 1e-16 --maxiter 100000 --restart ./PGthetas100/thetaR_${mols[$i]}_${bases[$i]}.npz --outfname ${outfname} > ${logfname}
done

