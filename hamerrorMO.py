import argparse
import os
import os.path
import numpy as np

parser = argparse.ArgumentParser(prog='hamerrorMO',
                                 description='Hamiltonian error for 8-fold symmetry-preserving model')

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

eetenCO = cp.einsum('ja,jklm,lp,mq,kb->abpq',xmatCP,cp.asarray(eeten),xmatCP,xmatCP,xmatCP,optimize=True)
# compute true theta
truetheta = eetenCO.reshape((-1))[selector]

if args.theta == 'true':
    whichtheta = truetheta
elif args.theta == 'custom':
    whichtheta = thetastar

err = cp.abs(whichtheta.reshape((-1)) - truetheta.reshape((-1)))
print(cp.max(err), cp.mean(err))


