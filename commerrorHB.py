import argparse
import os
import os.path
import numpy as np

parser = argparse.ArgumentParser(prog='commerrorHB',
                                 description='Commutator error for Hermitian representation model')

parser.add_argument('--gpu', type=int, help='which GPU to use')
parser.add_argument('--mol', help='name of molecule')
parser.add_argument('--basis', help='name of basis set')
parser.add_argument('--train', nargs='*', help='full paths of training trajectories')
parser.add_argument('--ntrain', type=int, required=False, help='how many steps of training trajectory to use')
parser.add_argument('--stride', type=int, required=False, help='trim both p and pdot using this stride')
parser.add_argument('--pagg', required=False, help='single aggregated p training file')
parser.add_argument('--pdotagg', required=False, help='single aggregated pdot training file')
parser.add_argument('--theta', required=True, choices=['true','custom'], help='which theta vector to use')
parser.add_argument('--custom', required=False, help='full path of custom theta, required for --theta custom')
parser.add_argument('--npzkey', required=False, help='key for .npz custom theta, required if --custom file is a .npz')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import cupy as cp
import cupyx.scipy.sparse as css
import culsmr

if args.theta=='custom':
    if args.custom is None:
        parser.error("--theta custom requires --custom.")
    if not os.path.isfile(args.custom):
        parser.error('custom theta file not found!')
    if args.custom[-3:]=='npz':
        if args.npzkey is None:
            parser.error('--npzkey required if custom theta file has .npz extension!')
    if args.custom[-3:]=='npz':
        thetastar = cp.asarray(np.load(args.custom,allow_pickle=True)[args.npzkey])
    else:
        thetastar = cp.asarray(np.load(args.custom))

mol = args.mol
basis = args.basis
overlap = np.load('./moldata/ke+en+overlap+ee_twoe+dip_hf_delta_s0_'+mol+'_'+basis+'.npz',allow_pickle=True)

if args.ntrain:
    stepsperfile = args.ntrain
else:
    stepsperfile = 2000

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

# new representer matrices
nprime = (drc+1)*drc//2
npripri = (drc-1)*drc//2

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

# new building blocks for sparse representer
fname = './moldata/representer_' + mol + '_' + basis + '.npz'
ncomp = drc * (drc+1) * (drc**2 + drc + 2) // 8
if os.path.isfile(fname):
    representerNPZ = np.load(fname)
    rowinds = representerNPZ['rowinds']
    rowinds2 = representerNPZ['rowinds2']
    colinds = representerNPZ['colinds']
    selector = representerNPZ['selector']    
else:
    blankten = np.zeros((drc,drc,drc,drc),dtype=np.int32)
    cnt = 0
    curind = 0
    rowinds = np.zeros(drc**4,dtype=np.int32)
    rowinds2 = np.zeros(drc**4,dtype=np.int32)
    colinds = np.zeros(drc**4,dtype=np.int32)
    selector = np.zeros(ncomp,dtype=np.int32)
    for i in range(drc):
        for j in range(drc):
            for k in range(drc):
                for l in range(drc):
                    if blankten[i,j,k,l] == 0:
                        symm = [(i,j,k,l),(j,i,l,k),(k,l,i,j),(l,k,j,i),(j,i,k,l),(l,k,i,j),(i,j,l,k),(k,l,j,i)]
                        symm = list(set(symm))
                        inds = np.array(symm).T
                        for s in symm:
                            blankten[s] = cnt
                        theserowinds = inds[0]*(drc**3)+inds[1]*(drc**2)+inds[2]*drc+inds[3]
                        theserowinds2 = inds[0]*(drc**3)+inds[2]*(drc**2)+inds[1]*drc+inds[3]
                        ltr = theserowinds.shape[0]
                        rowinds[curind:curind+ltr] = theserowinds
                        rowinds2[curind:curind+ltr] = theserowinds2
                        colinds[curind:curind+ltr] = cnt
                        selector[cnt] = theserowinds[0]
                        curind += ltr
                        cnt += 1
    np.savez(fname,rowinds=rowinds,rowinds2=rowinds2,colinds=colinds,selector=selector)

rowindsCP = cp.asarray(rowinds)
rowinds2CP = cp.asarray(rowinds2)
colindsCP = cp.asarray(colinds)
SmatCP = css.csr_matrix((cp.ones(drc**4),(rowindsCP,colindsCP)),shape=(drc**4,ncomp))
BmatCombCP = 2*css.csr_matrix((cp.ones(drc**4),(rowindsCP,colindsCP)),shape=(drc**4,ncomp))
BmatCombCP -= css.csr_matrix((cp.ones(drc**4),(rowinds2CP,colindsCP)),shape=(drc**4,ncomp))

kinmatCP = cp.asarray(kinmat)
enmatCP = cp.asarray(enmat)
xmatCP = cp.asarray(xmat)

# kinetic and electron-nuclear part of Hamiltonian in AO basis
hcoreAO = kinmatCP - enmatCP
# convert to canonically orthogonalized (CO) basis
hcore = -cp.einsum('ij,jk,kl->il',xmatCP.conj().T,hcoreAO,xmatCP,optimize=True)
# put eeten in CO basis
eetenCO = cp.einsum('ja,jklm,lp,mq,kb->abpq',xmatCP,cp.asarray(eeten),xmatCP,xmatCP,xmatCP,optimize=True)
# compute true theta
truetheta = eetenCO.reshape((-1))[selector]

# LOAD TRAINING DATA FROM DISK
# remember that args.train is now a list (multi-traj)
if args.train:
    p_train_list = []
    for tr in args.train:
        thisP = np.load(tr).reshape((-1,drc,drc))
        thisP = thisP[:stepsperfile,:,:]
        thisP = thisP[2:-2,:,:]
        if args.stride:
            thisP = thisP[::args.stride]
        p_train_list.append(thisP)
    ptrainCP = cp.asarray(np.concatenate(p_train_list))

if args.pagg:
    ptrainCP = cp.asarray(np.load(args.pagg))

mynumsteps = ptrainCP.shape[0]
xinp = cp.real(ptrainCP.reshape((-1,drc**2)))
yinp = cp.imag(ptrainCP.reshape((-1,drc**2)))

fname = './moldata/newtruetheta_' + mol + '_' + basis + '.npz'
if os.path.isfile(fname):
    newtruethetaNPZ = np.load(fname)
    truebeta0 = cp.asarray(newtruethetaNPZ['truebeta0'])
    truebeta1 = cp.asarray(newtruethetaNPZ['truebeta1'].reshape((-1)))
    truegamma0 = cp.asarray(newtruethetaNPZ['truegamma0'])
    truegamma1 = cp.asarray(newtruethetaNPZ['truegamma1'].reshape((-1)))
    
if args.theta == 'true':
    truethetaNP = np.concatenate([truebeta1.reshape((-1)),truegamma1.reshape((-1))])
    truetheta = cp.asarray(truethetaNP)
    whichtheta = truetheta
elif args.theta == 'custom':
    whichtheta = thetastar
    
# ground truth intercept
beta0true = xmatCP.conj().T @ (kinmatCP - enmatCP) @ xmatCP
beta0CP = -beta0true.reshape((-1))

# THE FOLLOWING IS IF YOU WANT TO SPECIFY ONLY v1,w1 as unknown parameters
v0 = truebeta0
v1 = whichtheta[:(drc**2)*nprime].reshape((drc**2, nprime))
w0 = truegamma0
w1 = whichtheta[(drc**2)*nprime:].reshape((drc**2, npripri))

realmat = cp.asarray(realmat)
imagmat = cp.asarray(imagmat)

# produce Hamiltonian
term0 = realmat @ v0
term1 = (xinp @ v1) @ realmat.T
term2 = imagmat @ w0
term3 = (yinp @ w1) @ imagmat.T
modrboth = (term0 + term1).reshape((-1,drc,drc))
modiboth = (term2 + term3).reshape((-1,drc,drc))
modh = modrboth + 1j*modiboth
# compute [H,P]
predcomm = modh @ ptrainCP - ptrainCP @ modh

trueten = (BmatCombCP @ truetheta).reshape((drc,drc,drc,drc))
# compute true two-electron term
truetwoe = -cp.einsum('ijkl,akl->aij',trueten,ptrainCP,optimize=True)
# compute full model Hamiltonian
trueh = hcore + truetwoe
# predicted commutator
truecomm = cp.einsum('aij,ajk->aik',trueh,ptrainCP,optimize=True) - cp.einsum('aij,ajk->aik',ptrainCP,trueh,optimize=True)

print( cp.max(cp.abs(predcomm - truecomm)), cp.sum(cp.abs(predcomm - truecomm)**2) )

