python gentraj.py --gpu 1 --mol lih --basis 6-31g --field off --theta true --dt 0.0008268 --nsteps 200000 --outfname ./trajdata100/train_lih_6-31g.npy

python gentraj.py --gpu 1 --mol lih --basis 6-31g --field on --theta true --dt 0.0008268 --nsteps 200000 --outfname ./trajdata100WF/train_lih_6-31g.npy
