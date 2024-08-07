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
  baseoutfname="thetaR_${mols[$i]}_${bases[$i]}.npz"
  inpart="${mols[$i]}_${bases[$i]}.npy"
  paggfname="./trajdata100R/one_p_${inpart}"
  pdotaggfname="./trajdata100R/one_pdot_${inpart}"
  
  # MOiter fits
  outfname="./MOiterthetas100/${baseoutfname}"
  logfname="./logsMOiter/${mols[$i]}R_${bases[$i]}_fit2.out"
  python fitMOiter.py --gpu 0 --mol ${mols[$i]} --basis ${bases[$i]} --pagg ${paggfname} --pdotagg ${pdotaggfname} --dt 0.0008268 --tol 1e-16 --maxiter 200000 --outfname ${outfname} > ${logfname}  
done

python fitMOiter.py --gpu 0 --mol c2h4 --basis sto-3g --train ./trajdata100/train_c2h4_sto-3g.npy --ntrain 200000 --dt 0.0008268 --tol 1e-16 --maxiter 200000 --outfname ./MOiterthetas100/theta_c2h4_sto-3g.npz > ./logsMOiter/c2h4_sto-3g_fit2.out

