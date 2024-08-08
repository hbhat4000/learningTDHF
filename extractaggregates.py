import argparse
import os
import os.path
import numpy as np

parser = argparse.ArgumentParser(prog='fit',
                                 description='Fit beta1 and gamma1 using new symmetry approach!')

parser.add_argument('--mol', help='name of molecule')
parser.add_argument('--basis', help='name of basis set')
parser.add_argument('--pagg', required=True, help='single aggregated p training file')
parser.add_argument('--pdotagg', required=True, help='single aggregated pdot training file')
parser.add_argument('--train', nargs='*', required=True, help='full paths of training trajectories')
parser.add_argument('--ntrain', type=int, required=False, help='how many steps of training trajectory to use')
parser.add_argument('--stride', type=int, required=False, help='trim both p and pdot using this stride')
parser.add_argument('--dt', type=float, required=False, help='time step of training trajectory')

args = parser.parse_args()

mol = args.mol
basis = args.basis
overlap = np.load('./moldata/ke+en+overlap+ee_twoe+dip_hf_delta_s0_'+mol+'_'+basis+'.npz',allow_pickle=True)

if args.ntrain:
    stepsperfile = args.ntrain
else:
    stepsperfile = 2000

if args.dt:
    dt = args.dt
else:
    dt = 0.08268

# put things into better variables
kinmat = overlap['ke_data']
enmat = overlap['en_data']
eeten = overlap['ee_twoe_data']
drc = eeten.shape[0]

# this bit of code sets up an array to store everything
# first we have to determine how big this array will be
ntrainfiles = len(args.train)
xx = np.arange(stepsperfile)
xx = xx[2:-2]
if args.stride:
    xx = xx[::args.stride]

rowsperfile = xx.shape[0]
print("rowsperfile = " + str(rowsperfile))
p_train_all = np.zeros((rowsperfile*ntrainfiles,drc,drc),dtype=np.complex128)
p_dot_all = np.zeros((rowsperfile*ntrainfiles,drc,drc),dtype=np.complex128)

# LOAD TRAINING DATA FROM DISK
# remember that args.train is now a list (multi-traj)

for ii in range(ntrainfiles):
    tr = args.train[ii]
    print(tr)
    thisP = np.load(tr).reshape((-1,drc,drc))
    thisP = thisP[:stepsperfile,:,:]
    thisPdot = (-thisP[4:,:,:] + 8*thisP[3:-1,:,:] - 8*thisP[1:-3,:,:] + thisP[:-4,:,:])/(12*dt)
    thisP = thisP[2:-2,:,:]
    if args.stride:
        thisPdot = thisPdot[::args.stride]
        thisP = thisP[::args.stride]
    print(thisPdot.shape)
    print(thisP.shape)
    p_dot_all[ ii*rowsperfile : (ii+1)*rowsperfile, :, : ] = thisPdot
    p_train_all[ ii*rowsperfile : (ii+1)*rowsperfile, :, : ] = thisP

# ptrainCP = cp.asarray(np.concatenate(p_train_list))
# pdotCP = cp.asarray(np.concatenate(p_dot_list))

# np.save('./trajdata100R/aggregate_p_'+mol+'_'+basis+'.npy',np.concatenate(p_train_list))
# np.save('./trajdata100R/aggregate_pdot_'+mol+'_'+basis+'.npy',np.concatenate(p_dot_list))

np.save(args.pagg,p_train_all)
np.save(args.pdotagg,p_dot_all)
