import argparse
import os
import os.path
import numpy as np

parser = argparse.ArgumentParser(prog='fitMOiter',
                                 description='Fit beta1 and gamma1 using new symmetry approach!')

parser.add_argument('--gpu', type=int, help='which GPU to use')
parser.add_argument('--mol', help='name of molecule')
parser.add_argument('--basis', help='name of basis set')
parser.add_argument('--train', nargs='*', required=False, help='full paths of training trajectories')
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
test = cp.einsum('ja,jklm,lp,mq,kb->abpq',xmatCP,cp.asarray(eeten),xmatCP,xmatCP,xmatCP,optimize=True)
# compute true theta
truetheta = test.reshape((-1))[selector]

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
    ptrainCP = cp.asarray(np.concatenate(p_train_list))
    pdotCP = cp.asarray(np.concatenate(p_dot_list))

# np.save('./trajdata100R/aggregate_p_'+mol+'_'+basis+'.npy',np.concatenate(p_train_list))
# np.save('./trajdata100R/aggregate_pdot_'+mol+'_'+basis+'.npy',np.concatenate(p_dot_list))
# import sys
# sys.exit()

if args.pagg:
    ptrainCP = cp.asarray(np.load(args.pagg))
    pdotCP = cp.asarray(np.load(args.pdotagg))

print("Sizes of ptrain and pdot:")
print([ptrainCP.shape,pdotCP.shape])
mynumsteps = ptrainCP.shape[0]

def resid(theta):
    # reconstruct transformed two-electron tensor
    modelten = (BmatCombCP @ theta).reshape((drc,drc,drc,drc))
    # compute two-electron term, result will be ntrain x drc x drc
    twoe = -cp.einsum('ijkl,akl->aij',modelten,ptrainCP,optimize=True)
    # compute full Hamiltonian
    h = hcore + twoe
    # commutator
    commutator = cp.einsum('aij,ajk->aik',h,ptrainCP,optimize=True) - cp.einsum('aij,ajk->aik',ptrainCP,h,optimize=True)
    out = 1j*pdotCP - commutator
    return out

def loss(theta):
    s = resid(theta)
    return cp.sum(cp.real(s * s.conj()))

print("Loss at truetheta:")
print(loss(truetheta))

def jdotvCP(vin):
    Bv = (BmatCombCP @ vin).reshape((drc,drc,drc,drc))
    term1 = cp.einsum('rlsm,ism,ilk->irk',Bv,ptrainCP,ptrainCP,optimize=True)
    term2 = -cp.einsum('ijl,lksp,isp->ijk',ptrainCP,Bv,ptrainCP,optimize=True)
    jdotvflat = (term1+term2).reshape((-1))
    return cp.concatenate([cp.real(jdotvflat), cp.imag(jdotvflat)])

def jHdotwCP(win):
    w = cp.asarray(win[:lenjac] + 1j*win[lenjac:]).reshape((-1,drc,drc))
    term1 = cp.einsum('irk,ism,ick->rcsm',w,ptrainCP.conj(),ptrainCP.conj(),optimize=True)
    term2 = cp.einsum('ijt,ijr,isp->rtsp',w,ptrainCP.conj(),ptrainCP.conj(),optimize=True)
    out = term1.reshape((-1)) @ BmatCombCP - term2.reshape((-1)) @ BmatCombCP
    return cp.real(out)

lentheta = ncomp
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

res = culsmr.culsmr(2*lenjac, lentheta, jdotvCP, jHdotwCP, -b, show=True, atol=mytol, btol=mytol, maxiter=mi, x0=thetastar0, outfname=outfname)
finaltheta = cp.asnumpy(res[0])
np.savez(outfname, x=finaltheta)

print('loss = ' + str(loss(cp.asarray(finaltheta))))
