python gentraj.py --gpu 0 --mol heh+ --basis 6-31g --field off --theta true --dt 0.0008268 --nsteps 200000 --outfname ./trajdata100/train_heh+_6-31g.npy

python gentraj.py --gpu 0 --mol heh+ --basis 6-31g --field on --theta true --dt 0.0008268 --nsteps 200000 --outfname ./trajdata100WF/train_heh+_6-31g.npy
