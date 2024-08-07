python ../fitPG.py --gpu 0 --mol heh+ --basis 6-31g --train ../trajdata/train_heh+_6-31g.npy --ntrain 20000 --tol 1e-32 --maxiter 20000 --outfname ../PGthetas/theta_heh+_6-31g.npz > heh+_6-31g_fit0.out

python ../fitPG.py --gpu 0 --mol heh+ --basis 6-31g --train ../trajdata10/train_heh+_6-31g.npy --ntrain 20000 --dt 0.008268 --tol 1e-32 --maxiter 20000 --outfname ../PGthetas10/theta_heh+_6-31g.npz > heh+_6-31g_fit1.out

python ../fitPG.py --gpu 0 --mol heh+ --basis 6-31g --train ../trajdata100/train_heh+_6-31g.npy --ntrain 20000 --dt 0.0008268 --tol 1e-32 --maxiter 20000 --outfname ../PGthetas100/theta_heh+_6-31g.npz > heh+_6-31g_fit2.out
