#!/bin/bash
python extractaggregates.py --mol c6h6n2o2 --basis sto-3g --pagg ./trajdata100R/aggregate_p_c6h6n2o2_sto-3g.npy --pdotagg ./trajdata100R/aggregate_pdot_c6h6n2o2_sto-3g.npy --train ./trajdata100/train_c6h6n2o2_sto-3g.npy --ntrain 200000 --stride 10 --dt 0.0008268

python extractaggregates.py --mol c6h6n2o2 --basis sto-3g --pagg ./trajdata100R/aggregateR_p_c6h6n2o2_sto-3g.npy --pdotagg ./trajdata100R/aggregateR_pdot_c6h6n2o2_sto-3g.npy --train ./trajdata100/trainW*_c6h6n2o2_sto-3g.npy --ntrain 20000 --stride 100 --dt 0.0008268

python combiner.py --files ./trajdata100R/aggregate_p_c6h6n2o2_sto-3g.npy ./trajdata100R/aggregateR_p_c6h6n2o2_sto-3g.npy --out ./trajdata100R/one_p_c6h6n2o2_sto-3g.npy

python combiner.py --files ./trajdata100R/aggregate_pdot_c6h6n2o2_sto-3g.npy ./trajdata100R/aggregateR_pdot_c6h6n2o2_sto-3g.npy --out ./trajdata100R/one_pdot_c6h6n2o2_sto-3g.npy
