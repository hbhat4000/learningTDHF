import argparse
import os
import os.path
import numpy as np

parser = argparse.ArgumentParser(prog='gentraj',
                                 description='Generate training or test trajectories!')
parser.add_argument('--gpu', required=True, type=int, help='which GPU to use')
parser.add_argument('--mol', required=True, help='name of molecule')
parser.add_argument('--basis', required=True, help='name of basis set')
parser.add_argument('--field', required=True, choices=['off', 'on'], help='whether to switch field off or on')
parser.add_argument('--postfix', required=False, help='identifying string to use when loading the initial condition')
parser.add_argument('--theta', required=True, choices=['true','custom'], help='which theta vector to use')
parser.add_argument('--custom', required=False, help='full path of custom theta, required for --theta custom')
parser.add_argument('--npzkey', required=False, help='key for .npz custom theta, required if --custom file is a .npz')
parser.add_argument('--ic', required=False, help='full path to load initial condition at time step 0')
parser.add_argument('--sic', required=False, help='full path to save density at last time step')
parser.add_argument('--nsteps', type=int, required=False, help='how many time steps to generate')
parser.add_argument('--dt', type=float, required=False, help='time step of training trajectory')
parser.add_argument('--outfname', required=True, help='output file name')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
import cupy as cp
import cupyx.scipy.sparse as css
# print(cp.show_config())

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
didat = [[]]*3
didat[0] = cp.asarray(overlap['dipx_data'])
didat[1] = cp.asarray(overlap['dipy_data'])
didat[2] = cp.asarray(overlap['dipz_data'])

# need these for orthogonalization below
# s = overlap['overlap_data']
# sevals, sevecs = np.linalg.eigh(s)
# sevalsCP = cp.asarray(sevals)
# sevecsCP = cp.asarray(sevecs)
sevalsfname = './rawtraj/sevals_'+mol+'_'+basis+'.npy'
sevals = np.load(sevalsfname)
sevecsfname = './rawtraj/sevecs_'+mol+'_'+basis+'.npy'
sevecs = np.load(sevecsfname)
sevalsCP = cp.asarray(sevals)
sevecsCP = cp.asarray(sevecs)
xmat = cp.matmul(sevecsCP, cp.diag(sevalsCP**(-0.5)))
X = cp.asnumpy(xmat)

# initial condition (IC)
if args.postfix is None:
    postfix = ''
else:
    postfix = args.postfix

if args.ic:
    denMOinit = np.load(args.ic)
else:
    denMOinit = np.load('./moldata/denMOtrueinit'+postfix+'_'+mol+'_'+basis+'.npy')

ic0 = cp.asarray(denMOinit)

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
BmatCombCP = 2*css.csr_matrix((cp.ones(drc**4),(rowindsCP,colindsCP)),shape=(drc**4,ncomp)) - css.csr_matrix((cp.ones(drc**4),(rowinds2CP,colindsCP)),shape=(drc**4,ncomp))

kinmatCP = cp.asarray(kinmat)
enmatCP = cp.asarray(enmat)
xmatCP = cp.asarray(xmat)

eetenflat = eeten.reshape((-1))
truethetaNP = eetenflat[selector]
truetheta = cp.asarray(truethetaNP)

# note: R must be anti-Hermitian for the code below to work
# this is a matrix exponential specialized for anti-Hermitian R
# note that it returns both exp(R) and exp(-R)
def cexpm(R):
    Q = R/1j
    # print(cp.mean(cp.abs(Q - Q.conj().T)))
    evals, evecs = cp.linalg.eigh(Q)
    U = evecs @ cp.diag(cp.exp(1j*evals)) @ evecs.conj().T
    Uinv = evecs @ cp.diag(cp.exp(-1j*evals)) @ evecs.conj().T
    return U, Uinv

# Hamiltonian in the CO (canonically orthogonalized) basis
def HamCO(eetenmodelCP,t,PCO,field):
    PAO = xmatCP @ PCO @ xmatCP.conj().T
    HAO = kinmatCP - enmatCP + (eetenmodelCP @ PAO.reshape((-1)))
    if field:
        freq = 0.0428
        if t > 2*cp.pi/freq:
            ez = 0
        elif t < 0:
            ez = 0
        else:
            ez = 0.05*cp.sin(0.0428*t)
        hfieldAO = ez*didat[2]
        HAO += hfieldAO
    H = -xmatCP.conj().T @ HAO @ xmatCP
    return H

def MMUTHO_CP(theta, initial_density, dt=0.08268, ntvec=2000, field=False):
    eetenmodelCP = (BmatCombCP @ theta).reshape((drc,drc,drc**2))
    P0 = initial_density.reshape((drc, drc))
    alldens = cp.zeros((ntvec, drc, drc), dtype=cp.complex128)
    alldens[0,:,:] = P0
    for i in range(ntvec-1):
        t = i*dt
        if i % 1000 == 0:
            print(i)
        P0 = alldens[i,:,:]
        k1 = -1j*dt*HamCO(eetenmodelCP,t,P0,field)
        Q1 = k1
        u2 = 0.5*Q1
        expu2, expmu2 = cexpm(u2)
        k2 = -1j*dt*HamCO(eetenmodelCP,t+0.5*dt,expu2 @ P0 @ expmu2,field)
        Q2 = k2 - k1
        u3 = 0.5*Q1 + 0.25*Q2
        expu3, expmu3 = cexpm(u3)
        k3 = -1j*dt*HamCO(eetenmodelCP,t+0.5*dt,expu3 @ P0 @ expmu3,field)
        Q3 = k3 - k2
        u4 = Q1 + Q2
        expu4, expmu4 = cexpm(u4)
        k4 = -1j*dt*HamCO(eetenmodelCP,t+dt,expu4 @ P0 @ expmu4,field)
        Q4 = k4 - 2*k2 + k1
        Q1Q2 = Q1 @ Q2 - Q2 @ Q1
        u5 = Q1/2.0 + Q2/4.0 + Q3/3.0 - Q4/24.0 - Q1Q2/48.0
        expu5, expmu5 = cexpm(u5)
        k5 = -1j*dt*HamCO(eetenmodelCP,t+0.5*dt,expu5 @ P0 @ expmu5,field)
        Q5 = k5 - k2
        u6 = Q1 + Q2 + (2.0/3.0)*Q3 + Q4/6.0 - Q1Q2/6.0
        expu6, expmu6 = cexpm(u6)
        k6 = -1j*dt*HamCO(eetenmodelCP,t+dt,expu6 @ P0 @ expmu6,field)
        Q6 = k6 - 2*k2 + k1
        R = Q2 - Q3 + Q5 + Q6/2.0
        v = Q1 + Q2 + (2.0/3.0)*Q5 + Q6/6.0 - (Q1 @ R - R @ Q1)/6.0
        expv, expmv = cexpm(v)
        alldens[i+1, :, :] = expv @ P0 @ expmv
    return alldens

if args.nsteps:
    mynumsteps = args.nsteps
else:
    mynumsteps = 2000

if args.dt:
    dt = args.dt
else:
    dt = 0.08268

if args.theta == 'true':
    whichtheta = truetheta
elif args.theta == 'custom':
    whichtheta = thetastar

if args.field == 'on':
    fieldFlag = True
else:
    fieldFlag = False

print('Propagating ' + args.theta + ' Hamiltonian for ' + args.mol + ' in ' + args.basis + ' with field ' + args.field + ' for ' + str(mynumsteps) + ' steps at dt = ' + str(dt))

propden = MMUTHO_CP(whichtheta, initial_density=ic0, dt=dt, ntvec=mynumsteps, field=fieldFlag).get()
np.save(args.outfname, propden)

if args.sic:
    np.save(args.sic, propden[-1])

hermcheck = np.linalg.norm(propden - np.transpose(propden,[0,2,1]).conj())
propden2 = np.einsum('ijk,ikl->ijl',propden,propden,optimize=True)
idemcheck = np.linalg.norm(propden2 - propden)
print('Hermitian check: ' + str(hermcheck))
print('Idempotency check: ' + str(idemcheck))
