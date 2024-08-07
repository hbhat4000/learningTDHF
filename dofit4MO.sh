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
  
  # MO fits
  outfname="./MOthetas100/${baseoutfname}"
  logfname="./logsMO/${mols[$i]}R_${bases[$i]}_fit2.out"
  python fitMO.py --gpu 1 --mol ${mols[$i]} --basis ${bases[$i]} --pagg ${paggfname} --pdotagg ${pdotaggfname} --dt 0.0008268 --outfname ${outfname} > ${logfname}  
done

python fitMO.py --gpu 1 --mol c2h4 --basis sto-3g --train ./trajdata100/train_c2h4_sto-3g.npy --ntrain 200000 --dt 0.0008268 --outfname ./MOthetas100/theta_c2h4_sto-3g.npz > ./logsMO/c2h4_sto-3g_fit2.out


