python gentraj.py --gpu 1 --mol c2h4 --basis sto-3g --field off --theta true --dt 0.0008268 --nsteps 200000 --outfname ./trajdata100/train_c2h4_sto-3g.npy

python gentraj.py --gpu 1 --mol c2h4 --basis sto-3g --field on --theta true --dt 0.0008268 --nsteps 200000 --outfname ./trajdata100WF/train_c2h4_sto-3g.npy

