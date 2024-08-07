#!/bin/bash
#SBATCH -A m4577_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 23:59:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --output="./HBhamerrs.out"

module load cudnn/8.9.3_cuda12

molsys=("heh+ 6-31g" "heh+ 6-311G" "lih 6-31g" "lih 6-311ppgss" "c2h4 sto-3g" "c2h4 6-31pgs" "c6h6n2o2 sto-3g")

echo "Single field-free trajectory + LSMR models"

for i in ${!molsys[@]}; do
    set -- ${molsys[$i]}  # Convert the "tuple" into the param args $1 $2...
    echo -n $1 $2 ""
    
    if [ $i -eq 0 ] || [ $i -eq 1 ] || [ $i -eq 2 ] || [ $i -eq 4 ]; then
        thetapath="./HBthetas100"
        python hamerrorHB.py --gpu 0 --mol $1 --basis $2 --theta custom --custom $thetapath/theta_lowtol_$1_$2.npz --npzkey x
    else
        thetapath="./HBthetas100"
        python hamerrorHB.py --gpu 0 --mol $1 --basis $2 --theta custom --custom $thetapath/theta_lowtol_$1_$2.npz --npzkey x
    fi
done


echo ""
echo "Ensemble of field-free trajectories + LSMR models"

for i in ${!molsys[@]}; do
    set -- ${molsys[$i]}  # Convert the "tuple" into the param args $1 $2...
    echo -n $1 $2 ""

    if [ $i -eq 0 ] || [ $i -eq 1 ] || [ $i -eq 2 ] || [ $i -eq 4 ]; then
        thetapath="./HBthetas100"
        python hamerrorHB.py --gpu 0 --mol $1 --basis $2 --theta custom --custom ./HBthetas100/thetaR_lowtol_$1_$2.npz --npzkey x
    else
        thetapath="./HBthetas100"
        python hamerrorHB.py --gpu 0 --mol $1 --basis $2 --theta custom --custom ./HBthetas100/thetaR_lowtol_$1_$2.npz --npzkey x
    fi
done

echo ""

