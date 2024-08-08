import argparse
import os
import os.path
import numpy as np

parser = argparse.ArgumentParser(prog='fit',
                                 description='Fit beta only using old rethinking approach!')

parser.add_argument('--gpu', type=int, help='which GPU to use')
parser.add_argument('--mol', help='name of molecule')
parser.add_argument('--basis', help='name of basis set')
parser.add_argument('--train', nargs='*', help='full paths of training trajectories')
parser.add_argument('--ntrain', type=int, required=False, help='how many steps of training trajectory to use')
parser.add_argument('--stride', type=int, required=False, help='trim both p and pdot using this stride')
parser.add_argument('--pagg', required=False, help='single aggregated p training file')
parser.add_argument('--pdotagg', required=False, help='single aggregated pdot training file')
parser.add_argument('--dt', type=float, required=False, help='time step of training trajectory')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import cupy as cp
import cupyx.scipy.sparse as css
import culsmr

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

# need these for orthogonalization below
sevalsfname = './rawtraj/sevals_'+mol+'_'+basis+'.npy'
sevals = np.load(sevalsfname)
sevecsfname = './rawtraj/sevecs_'+mol+'_'+basis+'.npy'
sevecs = np.load(sevecsfname)
sevalsCP = cp.asarray(sevals)
sevecsCP = cp.asarray(sevecs)
xmat = sevecs @ np.diag(sevals**(-0.5))

kinmatCP = cp.asarray(kinmat)
enmatCP = cp.asarray(enmat)
xmatCP = cp.asarray(xmat)

eetenflat = eeten.reshape((-1))

# put eeten in CO basis
eetenCO = -np.einsum('ja,jklm,lp,mq,kb->abpq',xmat,eeten,xmat,xmat,xmat,optimize=True)
# define critical tensor
mathcalE = 2*eetenCO-np.transpose(eetenCO,(0,2,1,3))
# flatten and put on GPU
mathcalECP = cp.asarray(mathcalE.reshape((-1)))

# LOAD TRAINING DATA FROM DISK
# remember that args.train is now a list (multi-traj)
if args.train:
    p_train_list = []
    p_dot_list = []
    for tr in args.train:
        thisP = np.load(tr).reshape((-1,drc,drc))
        thisP = thisP[:stepsperfile,:,:]
        thisPdot = (-thisP[4:,:,:] + 8*thisP[3:-1,:,:] - 8*thisP[1:-3,:,:] + thisP[:-4,:,:])/(12*dt)
        thisP = thisP[2:-2,:,:]
        if args.stride:
            thisPdot = thisPdot[::args.stride]
            thisP = thisP[::args.stride]
        p_dot_list.append(thisPdot)
        p_train_list.append(thisP)
    pinp = cp.asarray(np.concatenate(p_train_list))
    pdot = cp.asarray(np.concatenate(p_dot_list))

# np.save('./trajdata100R/aggregate_p_'+mol+'_'+basis+'.npy',np.concatenate(p_train_list))
# np.save('./trajdata100R/aggregate_pdot_'+mol+'_'+basis+'.npy',np.concatenate(p_dot_list))
# import sys
# sys.exit()

if args.pagg:
    pinp = cp.asarray(np.load(args.pagg))
    pdot = cp.asarray(np.load(args.pdotagg))

mynumsteps = pinp.shape[0]

pinpC = pinp.conj()
pinpH = cp.transpose(pinpC, axes=(0,2,1))
xinp = cp.real(pinp)
yinp = cp.imag(pinp)

# ground truth intercept
beta0true = xmatCP.conj().T @ (kinmatCP - enmatCP) @ xmatCP
beta0CP = -beta0true.reshape((-1))

def resid(theta):
    # use the same parameters for both the real and imag targets!
    beta1 = theta.reshape((drc**2, drc**2))
    mat = (beta0CP + cp.matmul(pinp.reshape((-1, drc**2)), beta1)).reshape((-1, drc, drc))
    # produce real target
    rmat = cp.real(mat)
    # produce imag target
    qmat = cp.imag(mat) 
    h = 0.5*(rmat + cp.transpose(rmat,(0,2,1))) + 0.5j*(qmat - cp.transpose(qmat,(0,2,1)))
    # [h, p]
    commutator = cp.matmul(h,pinp) - cp.matmul(pinp,h)
    # now compute residual
    s = 1j*pdot - commutator
    return s

def loss(thetain):
    theta = cp.asarray(thetain)
    s = resid(theta)
    return cp.sum(cp.real(s * cp.conj(s)))

print(loss(mathcalECP))

