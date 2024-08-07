#!/bin/bash
python extractaggregates.py --mol lih --basis 6-311ppgss --pagg ./trajdata100R/aggregate_p_lih_6-311ppgss.npy --pdotagg ./trajdata100R/aggregate_pdot_lih_6-311ppgss.npy --train ./trajdata100/train_lih_6-311ppgss.npy --ntrain 200000 --stride 10 --dt 0.0008268

python extractaggregates.py --mol lih --basis 6-311ppgss --pagg ./trajdata100R/aggregateR_p_lih_6-311ppgss.npy --pdotagg ./trajdata100R/aggregateR_pdot_lih_6-311ppgss.npy --train ./trajdata100/trainW*_lih_6-311ppgss.npy --ntrain 20000 --stride 100 --dt 0.0008268

python combiner.py --files ./trajdata100R/aggregate_p_lih_6-311ppgss.npy ./trajdata100R/aggregateR_p_lih_6-311ppgss.npy --out one_p_lih_6-311ppgss.npy

python combiner.py --files ./trajdata100R/aggregate_pdot_lih_6-311ppgss.npy ./trajdata100R/aggregateR_pdot_lih_6-311ppgss.npy --out one_pdot_lih_6-311ppgss.npy
