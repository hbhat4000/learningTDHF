#!/bin/bash
python extractaggregates.py --mol c2h4 --basis 6-31pgs --pagg ./trajdata100R/aggregate_p_c2h4_6-31pgs.npy --pdotagg ./trajdata100R/aggregate_pdot_c2h4_6-31pgs.npy --train ./trajdata100/train_c2h4_6-31pgs.npy --ntrain 200000 --stride 5 --dt 0.0008268

python extractaggregates.py --mol c2h4 --basis 6-31pgs --pagg ./trajdata100R/aggregateR_p_c2h4_6-31pgs.npy --pdotagg ./trajdata100R/aggregateR_pdot_c2h4_6-31pgs.npy --train /tmp/trajdata100/trainW*_c2h4_6-31pgs.npy --ntrain 20000 --stride 50 --dt 0.0008268

python combiner.py --files ./trajdata100R/aggregate_p_c2h4_6-31pgs.npy ./trajdata100R/aggregateR_p_c2h4_6-31pgs.npy --out ./trajdata100R/one_80k_p_c2h4_6-31pgs.npy

python combiner.py --files ./trajdata100R/aggregate_pdot_c2h4_6-31pgs.npy ./trajdata100R/aggregateR_pdot_c2h4_6-31pgs.npy --out ./trajdata100R/one_80k_pdot_c2h4_6-31pgs.npy
