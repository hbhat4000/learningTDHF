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
parser.add_argument('--tol', type=float, required=False, help='optimization tolerance')
parser.add_argument('--maxiter', type=int, required=False, help='maximum number of optimization iterations')
parser.add_argument('--restart', required=False, help='restart training from previous theta file')
parser.add_argument('--outfname', help='output file name')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import cupy as cp
import cupyx.scipy.sparse as css
import culsmr
print(cp.show_config())

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
    
if args.tol:
    mytol = args.tol
else:
    mytol = 1e-16
    
if args.maxiter:
    mi = args.maxiter
else:
    mi = 500

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

# LOAD TRAINING DATA FROM DISK
# remember that args.train is now a list (multi-traj)
if args.train:
    p_train_list = []
    p_dot_list = []
    for tr in args.train:
        print(tr)
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

print("Sizes of ptrain and pdot:")
print([pinp.shape,pdot.shape])
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

# assume that v has shape drc x drc x drc x drc
def jpbeta1R(vin):
    v = vin.reshape((drc, drc, drc, drc))
    # first contract X^{(t)}_{a,b} with v_{abcd}
    # think of xv as xv^{(t)}_{cd}
    xv = cp.einsum('ijk,jklm->ilm',xinp,v,optimize=True)
    xvT = cp.transpose(xv, axes=(0,2,1))
    # now we get four terms
    res1 = -0.5*(cp.matmul(xv,pinp) + cp.matmul(xvT,pinp) - cp.matmul(pinp,xv) - cp.matmul(pinp,xvT))
    resout1 = res1.reshape((-1))
    # first contract Y^{(t)}_{a,b} with v_{abcd}
    # think of yv as yv^{(t)}_{cd}
    yv = cp.einsum('ijk,jklm->ilm',yinp,v,optimize=True)
    yvT = cp.transpose(yv, axes=(0,2,1))
    # now we get four terms
    res2 = -0.5j*(cp.matmul(yv,pinp) - cp.matmul(yvT,pinp) - cp.matmul(pinp,yv) + cp.matmul(pinp,yvT))
    resout2 = res2.reshape((-1))
    resout = resout1 + resout2
    return resout

# assume w is of the form w^{(t)_{k,m}}
def jpbeta1L(win):
    w = win.reshape((mynumsteps, drc, drc))
    # first compute the parenthetical term
    wT = cp.transpose(w, axes=(0,2,1))
    parter = cp.matmul(w,pinpH) + cp.matmul(pinpC,wT) - cp.matmul(pinpH,w) - cp.matmul(wT,pinpC)
    # now we take a tensor or outer product and sum over time while we're at it
    ressum1 = -0.5*cp.einsum('iab,icd->abcd',xinp,parter,optimize=True)
    parter = cp.matmul(w,pinpH) - cp.matmul(pinpC,wT) - cp.matmul(pinpH,w) + cp.matmul(wT,pinpC)
    # now we take a tensor or outer product and sum over time while we're at it
    ressum2 = 0.5j*cp.einsum('iab,icd->abcd',yinp,parter,optimize=True)
    ressum = ressum1 + ressum2
    return ressum.reshape((-1))

def jdotv(beta1):
    res = jpbeta1R(beta1)
    resnum = cp.concatenate([cp.real(res), cp.imag(res)])
    return resnum

def jHdotw(win):
    w = cp.asarray(win[:lenjac] + 1j*win[lenjac:])
    res = jpbeta1L(w)
    resnum = cp.real(res)
    return resnum

lentheta = drc**4
print('len(theta) = ' + str(lentheta))
lenjac = mynumsteps*drc*drc
print('len(jac) = ' + str(lenjac))

bcplx = resid(cp.zeros(lentheta)).reshape((-1))
b = cp.concatenate([cp.real(bcplx), cp.imag(bcplx)])

# restart from file if flag is set
if args.restart:
    thetastar0 = cp.asarray(np.load(args.restart)['x'])
else:
    thetastar0 = cp.zeros(lentheta)

outfname = args.outfname

res = culsmr.culsmr(2*lenjac, lentheta, jdotv, jHdotw, -b, show=True, atol=mytol, btol=mytol, maxiter=mi, x0=thetastar0, outfname=outfname)

np.savez(outfname, x=cp.asnumpy(res[0]))

print('loss = ' + str(loss(res[0])))
