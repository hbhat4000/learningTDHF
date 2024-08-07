import argparse
import os
import os.path
import numpy as np

parser = argparse.ArgumentParser(prog='hamerrorHB',
                                 description='Hamiltonian error for Hermitian representation model')

parser.add_argument('--gpu', type=int, help='which GPU to use')
parser.add_argument('--mol', help='name of molecule')
parser.add_argument('--basis', help='name of basis set')
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
xmatCP = cp.asarray(xmat)

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

# put eeten in CO basis
eetenCO = cp.einsum('ja,jklm,lp,mq,kb->abpq',xmatCP,cp.asarray(eeten),xmatCP,xmatCP,xmatCP,optimize=True)

mathcalE = 2*eetenCO-cp.transpose(eetenCO,(0,2,1,3))
mathcalEsymm = 0.5*(mathcalE + cp.transpose(mathcalE,(1,0,2,3)))
mathcalEantisymm = 0.5*(mathcalE - cp.transpose(mathcalE,(1,0,2,3)))
truev = cp.zeros((drc,drc,nprime))
for i in range(drc):
    for j in range(drc):
        truev[i,j,:] = -mathcalEsymm[i,j,:,:][np.triu_indices(drc)]

truew = cp.zeros((drc,drc,npripri))
for i in range(drc):
    for j in range(drc):
        truew[i,j,:] = -mathcalEantisymm[i,j,:,:][np.triu_indices(drc,1)]

realmatCP = cp.asarray(realmat.reshape((drc,drc,nprime)))
imagmatCP = cp.asarray(imagmat.reshape((drc,drc,npripri)))
truebeta = cp.einsum('abk,cdk->cdab',realmatCP,truev)
truegamma = cp.einsum('abk,cdk->cdab',imagmatCP,truew)

modv = thetastar[:(drc**2)*nprime].reshape((drc,drc,nprime))
modw = thetastar[(drc**2)*nprime:].reshape((drc,drc,npripri))
modbeta = cp.einsum('abk,cdk->cdab',realmatCP,modv)
modgamma = cp.einsum('abk,cdk->cdab',imagmatCP,modw)

errbeta = cp.abs(modbeta - truebeta).reshape((-1))
errgamma = cp.abs(modgamma - truegamma).reshape((-1))
err = cp.concatenate([errbeta, errgamma])
print( cp.max(err), cp.mean(err) )

