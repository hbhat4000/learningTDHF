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
bases=("sto-3g")
for i in ${!mols[@]}; do
  baseoutfname="thetaR_lowtol_${mols[$i]}_${bases[$i]}.npz"
  inpart="${mols[$i]}_${bases[$i]}.npy"
  paggfname="./trajdata100R/one_p_${inpart}"
  pdotaggfname="./trajdata100R/one_pdot_${inpart}"
  
  # HB fits
  outfname="./HBthetas100/${baseoutfname}"
  logfname="./logsHB/${mols[$i]}R_lowtol_${bases[$i]}_fit2.out"
  python fitHB.py --gpu 0 --mol ${mols[$i]} --basis ${bases[$i]} --pagg ${paggfname} --pdotagg ${pdotaggfname} --dt 0.0008268 --tol 1e-16 --maxiter 100000 --outfname ${outfname} > ${logfname}  
  
  # PG fits
  outfname="./PGthetas100/${baseoutfname}"
  logfname="./logsPG/${mols[$i]}R_lowtol_${bases[$i]}_fit2.out"
  python fitPG.py --gpu 0 --mol ${mols[$i]} --basis ${bases[$i]} --pagg ${paggfname} --pdotagg ${pdotaggfname} --dt 0.0008268 --tol 1e-16 --maxiter 100000 --outfname ${outfname} > ${logfname}  
done

