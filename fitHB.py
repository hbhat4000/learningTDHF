import argparse
import os
import os.path
import numpy as np

parser = argparse.ArgumentParser(prog='fit',
                                 description='Fit beta1 and gamma1 using new symmetry approach!')

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
X = xmat

kinmatCP = cp.asarray(kinmat)
enmatCP = cp.asarray(enmat)
xmatCP = cp.asarray(xmat)

# new representer matrices
nprime = (drc+1)*drc//2
npripri = (drc-1)*drc//2
print("nprime = " + str(nprime))
print("npripri = " + str(npripri))

# check whether we have the realmat stored on disk
# if not, create it and save it!
fname = './moldata/realmat_' + mol + '_' + basis + '.npy'
if os.path.isfile(fname):
    realmat = np.load(fname)
else:
    realmat = np.zeros((drc**2,nprime),dtype=np.int16)
    for i in range(drc):
        for j in range(drc):
            row = i*drc + j
            if i<=j:
                col = i*drc + j - i*(i+1)//2
            else:
                col = j*drc + i - j*(j+1)//2
            realmat[row,col]=1
    
    np.save(fname,realmat)

# check whether we have the imagmat stored on disk
# if not, create it and save it!
fname = './moldata/imagmat_' + mol + '_' + basis + '.npy'
if os.path.isfile(fname):
    imagmat = np.load(fname)
else:
    imagmat = np.zeros((drc**2,npripri),dtype=np.int16)
    for i in range(drc):
        for j in range(drc):
            row = i*drc + j
            if i<j:
                col = i*drc + j - (i+1)*(i+2)//2
                imagmat[row,col]=1
            if i>j:
                col = j*drc + i - (j+1)*(j+2)//2
                imagmat[row,col]=-1

    np.save(fname,imagmat)

realweights = np.diag(realmat.T @ realmat)
imagweights = np.diag(imagmat.T @ imagmat)

# check whether we have the truetheta stored on disk
# if not, create it and save it!
fname = './moldata/newtruetheta_' + mol + '_' + basis + '.npz'
if os.path.isfile(fname):
    newtruethetaNPZ = np.load(fname)
    truebeta0 = cp.asarray(newtruethetaNPZ['truebeta0'])
    truebeta1 = cp.asarray(newtruethetaNPZ['truebeta1'].reshape((-1)))
    truegamma0 = cp.asarray(newtruethetaNPZ['truegamma0'])
    truegamma1 = cp.asarray(newtruethetaNPZ['truegamma1'].reshape((-1)))
else:
    # first compute the "old" beta0,beta1,gamma0,gamma1
    beta0trueNP = X.conj().T @ (kinmat - enmat) @ X
    gamma0trueNP = np.zeros(beta0trueNP.shape)
    oldtruebeta0 = -beta0trueNP.reshape((drc,drc))
    oldtruegamma0 = -gamma0trueNP.reshape((drc,drc))
    # need two masks
    # upper mask is matrix whose (u,v)-th element is 0 unless u <= v
    # lower mask is matrix whose (u,v)-th element is 0 unless u > v
    upper = np.zeros((drc,drc),dtype=np.float64)
    lower = np.zeros((drc,drc),dtype=np.float64)
    for u in range(drc):
        for v in range(drc):
            if u <= v:
                upper[u,v] = 1.0
            if u > v:
                lower[u,v] = 1.0
    ru1 = 2*np.einsum('uv,uk,ma,sb,uvms,vl->klab',upper,X,X,X,eeten,X,optimize=True)
    ru2 = np.einsum('uv,uk,ma,sb,umvs,vl->klab',upper,X,X,X,eeten,X,optimize=True)
    rl1 = 2*np.einsum('uv,uk,ma,sb,vums,vl->klab',lower,X,X,X,eeten,X,optimize=True)
    rl2 = np.einsum('uv,uk,ma,sb,vmus,vl->klab',lower,X,X,X,eeten,X,optimize=True)
    beta1trueNP = ru1 - ru2 + rl1 - rl2
    gamma1trueNP = ru1 - ru2 - rl1 + rl2
    oldtruebeta1 = -beta1trueNP.reshape((drc,drc,drc,drc))
    oldtruegamma1 = -gamma1trueNP.reshape((drc,drc,drc,drc))
    
    symbeta0 = 0.5*(oldtruebeta0 + oldtruebeta0.T)
    truebeta0 = symbeta0[np.triu_indices(drc)]
    
    symbeta1 = 0.5*(oldtruebeta1 + np.transpose(oldtruebeta1,axes=(0,1,3,2)))
    truebeta1 = np.zeros((drc,drc,drc*(drc+1)//2))
    for i in range(drc):
        for j in range(drc):
            truebeta1[i,j,:] = symbeta1[i,j,:,:][np.triu_indices(drc)]
    
    antisymgamma0 = 0.5*(oldtruegamma0 - oldtruegamma0.T)
    truegamma0 = antisymgamma0[np.triu_indices(drc,1)]
    
    antisymgamma1 = 0.5*(oldtruegamma1 - np.transpose(oldtruegamma1,axes=(0,1,3,2)))
    truegamma1 = np.zeros((drc,drc,drc*(drc-1)//2))
    for i in range(drc):
        for j in range(drc):
            truegamma1[i,j,:] = antisymgamma1[i,j,:,:][np.triu_indices(drc,1)]
    
    np.savez(fname,truebeta0=truebeta0,truebeta1=truebeta1,
             truegamma0=truegamma0,truegamma1=truegamma1)


truethetaNP = np.concatenate([truebeta1.reshape((-1)),truegamma1.reshape((-1))])

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

# pdot2 = (p_train[2:,:,:] - p_train[:-2,:,:])/(2*dt)
# pdot4 = (-p_train[4:,:,:] + 8*p_train[3:-1,:,:] - 8*p_train[1:-3,:,:] + p_train[:-4,:,:])/(12*dt)
# ptrainCP = cp.asarray(p_train[2:-2,:,:])
# pinp = cp.asarray(p_train[2:-2,:,:])
# pdot = cp.asarray(pdot4)

pinpC = pinp.conj()
pinpH = cp.transpose(pinpC, axes=(0,2,1))
xinp = cp.real(pinp.reshape((-1,drc**2)))
yinp = cp.imag(pinp.reshape((-1,drc**2)))

# MOVE SOME FIXED MATRICES TO GPU
realmat = cp.asarray(realmat)
imagmat = cp.asarray(imagmat)
realweights = cp.asarray(realweights)
imagweights = cp.asarray(imagweights)
truebeta0 = cp.asarray(truebeta0)
truegamma0 = cp.asarray(truegamma0)
truebeta1 = cp.asarray(truebeta1)
truegamma1 = cp.asarray(truegamma1)
truetheta = cp.asarray(truethetaNP)

def resid(theta):
    # THE FOLLOWING IS IF YOU WANT TO SPECIFY ONLY v1,w1 as unknown parameters
    v0 = truebeta0
    v1 = theta[:(drc**2)*nprime].reshape((drc**2, nprime))
    w0 = truegamma0
    w1 = theta[(drc**2)*nprime:].reshape((drc**2, npripri))
    
    # produce Hamiltonian
    term0 = realmat @ v0
    term1 = (xinp @ v1) @ realmat.T
    term2 = imagmat @ w0
    term3 = (yinp @ w1) @ imagmat.T
    rboth = (term0 + term1).reshape((-1,drc,drc))
    iboth = (term2 + term3).reshape((-1,drc,drc))
    h = rboth + 1j*iboth
    # compute [H,P]
    commutator = h @ pinp - pinp @ h
    # residual = i*\dot{P} - [H,P]
    return 1j*pdot - commutator

def loss(thetain):
    theta = cp.asarray(thetain)
    s = resid(theta)
    loss = cp.sum(cp.real(s * cp.conj(s)))
    return cp.asnumpy(loss)

print("Loss at truetheta:")
print(loss(truetheta))

def jacobv1R(inphi):
    phi = inphi.reshape((drc**2,nprime))
    contract = cp.einsum('ai,km,im->ak',xinp,realmat,phi,optimize=True).reshape((mynumsteps,drc,drc))
    commutator = contract @ pinp - pinp @ contract
    return -commutator.reshape((-1))

def jacobw1R(inphi):
    phi = inphi.reshape((drc**2,npripri))
    contract = cp.einsum('ai,km,im->ak',yinp,imagmat,phi,optimize=True).reshape((mynumsteps,drc,drc))
    commutator = contract @ pinp - pinp @ contract
    return -1j*commutator.reshape((-1))

def jacobv1L(inphi):
    phi = inphi.reshape((mynumsteps,drc,drc))
    jrealten = realmat.reshape((drc,drc,nprime))
    term1 = cp.einsum('jkl,krm,jrl->jm',phi,jrealten,pinpC,optimize=True)
    term2 = cp.einsum('jkl,jkr,rlm->jm',phi,pinpC,jrealten,optimize=True)
    einterms = -term1+term2
    return cp.einsum('ai,am->im',xinp,einterms,optimize=True).reshape((-1))

def jacobw1L(inphi):
    phi = inphi.reshape((mynumsteps,drc,drc))
    jimagten = imagmat.reshape((drc,drc,npripri))
    term1 = cp.einsum('jkl,krm,jrl->jm',phi,jimagten,pinpC,optimize=True)
    term2 = cp.einsum('jkl,jkr,rlm->jm',phi,pinpC,jimagten,optimize=True)
    einterms = term1-term2
    return 1j*cp.einsum('ai,am->im',yinp,einterms,optimize=True).reshape((-1))

def jdotv(theta):
    # beta0 = v[:(drc**2)]
    v1 = theta[:(drc**2)*nprime]
    # gamma0 = v[(drc**2 + drc**4) : (2*drc**2 + drc**4)]
    w1 = theta[(drc**2)*nprime:]
    prod1 = jacobv1R(v1)
    prod3 = jacobw1R(w1)
    res = prod1 + prod3
    resnum = cp.concatenate([cp.real(res), cp.imag(res)])
    return resnum

def jHdotw(win):
    w = cp.asarray(win[:lenjac] + 1j*win[lenjac:])
    prod1 = jacobv1L(w)
    prod3 = jacobw1L(w)
    res = cp.concatenate([prod1, prod3])
    resnum = cp.real(res)
    return resnum

lentheta = drc**4
print('len(theta) = ' + str(lentheta))
lenjac = mynumsteps*drc*drc
print('len(jac) = ' + str(lenjac))

bcplx = resid(cp.zeros(lentheta)).reshape((-1))
b = cp.concatenate([cp.real(bcplx), cp.imag(bcplx)])

outfname = args.outfname

# restart from file if flag is set
if args.restart:
    thetastar0 = cp.asarray(np.load(args.restart)['x'])
else:
    thetastar0 = cp.zeros(lentheta)

res = culsmr.culsmr(2*lenjac, lentheta, jdotv, jHdotw, -b, show=True, atol=mytol, btol=mytol, maxiter=mi, x0=thetastar0, outfname=outfname)

np.savez(outfname, x=cp.asnumpy(res[0]))

print('loss = ' + str(loss(res[0])))
