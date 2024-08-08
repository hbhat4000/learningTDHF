#!/bin/bash

mols=("heh+" "heh+" "lih" "c2h4" "lih" "c2h4" "c6h6n2o2")
bases=("6-31g" "6-311G" "6-31g" "sto-3g" "6-311ppgss" "6-31pgs" "sto-3g")

for i in ${!mols[@]}; do
  for j in {0..99}; do
    mystr="W$j"
    python randic2.py --mol ${mols[$i]} --basis ${bases[$i]} --fac 10.0 --postfix $mystr
  done
done
