#!/bin/bash
#SBATCH -A m4577_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 23:59:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -o ./PGlosses.out

module load cudnn

molsys=("heh+ 6-31g" "heh+ 6-311G" "lih 6-31g" "c2h4 sto-3g")

for i in ${!molsys[@]}; do
    set -- ${molsys[$i]}  # Convert the "tuple" into the param args $1 $2...
    echo -n $1 $2 ""
    python ./printlossPG.py --gpu 0 --mol $1 --basis $2 --train ./trajdata100/train_$1_$2.npy --ntrain 200000 --dt 0.0008268 
done

molsys=("lih 6-311ppgss" "c2h4 6-31pgs" "c6h6n2o2 sto-3g")

for i in ${!molsys[@]}; do
    set -- ${molsys[$i]}  # Convert the "tuple" into the param args $1 $2...
    echo -n $1 $2 ""
    python ./printlossPG.py --gpu 0 --mol $1 --basis $2 --train ./trajdata100/train_$1_$2.npy --ntrain 200000 --stride 10 --dt 0.0008268 
done


