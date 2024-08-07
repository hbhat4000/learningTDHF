#!/bin/bash
#SBATCH -A m4577_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 23:59:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

module load cudnn/8.9.3_cuda12

mols=("heh+")
bases=("6-31g")
for i in ${!mols[@]}; do
  baseoutfname="thetaR_${mols[$i]}_${bases[$i]}.npz"
  inpart="${mols[$i]}_${bases[$i]}.npy"
  paggfname="./trajdata100R/one_p_${inpart}"
  pdotaggfname="./trajdata100R/one_pdot_${inpart}"
  
  # MO fits
  outfname="./MOthetas100/${baseoutfname}"
  logfname="./logsMO/${mols[$i]}R_${bases[$i]}_fit2.out"
  python fitMO.py --gpu 0 --mol ${mols[$i]} --basis ${bases[$i]} --pagg ${paggfname} --pdotagg ${pdotaggfname} --dt 0.0008268 --outfname ${outfname} > ${logfname}  
done

python fitMO.py --gpu 0 --mol heh+ --basis 6-31g --train ./trajdata100/train_heh+_6-31g.npy --ntrain 200000 --dt 0.0008268 --outfname ./MOthetas100/theta_heh+_6-31g.npz > ./logsMO/heh+_6-31g_fit2.out


