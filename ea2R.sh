#!/bin/bash
python extractaggregates.py --mol lih --basis 6-31g --pagg ./trajdata100R/aggregate_p_lih_6-31g.npy --pdotagg ./trajdata100R/aggregate_pdot_lih_6-31g.npy --train ./trajdata100/train_lih_6-31g.npy --ntrain 200000 --stride 5 --dt 0.0008268

python extractaggregates.py --mol lih --basis 6-31g --pagg ./trajdata100R/aggregateR_p_lih_6-31g.npy --pdotagg ./trajdata100R/aggregateR_pdot_lih_6-31g.npy --train ./trajdata100/trainW*_lih_6-31g.npy --ntrain 20000 --stride 50 --dt 0.0008268

python combiner.py --files ./trajdata100R/aggregate_p_lih_6-31g.npy ./trajdata100R/aggregateR_p_lih_6-31g.npy --out one_p_lih_6-31g.npy

python combiner.py --files ./trajdata100R/aggregate_pdot_lih_6-31g.npy ./trajdata100R/aggregateR_pdot_lih_6-31g.npy --out one_pdot_lih_6-31g.npy