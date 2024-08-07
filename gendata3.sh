python gentraj.py --gpu 0 --mol lih --basis 6-311ppgss --field off --theta true --dt 0.0008268 --nsteps 200000 --outfname ./trajdata100/train_lih_6-311ppgss.npy

python gentraj.py --gpu 0 --mol lih --basis 6-311ppgss --field on --theta true --dt 0.0008268 --nsteps 200000 --outfname ./trajdata100WF/train_lih_6-311ppgss.npy
