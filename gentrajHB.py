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

denMOinit = np.load('./moldata/denMOtrueinit'+postfix+'_'+mol+'_'+basis+'.npy')
ic = cp.asarray(denMOinit)

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
truetheta = cp.asarray(truethetaNP)

# MOVE SOME FIXED MATRICES TO GPU
realmat = cp.asarray(realmat)
imagmat = cp.asarray(imagmat)
realweights = cp.asarray(realweights)
imagweights = cp.asarray(imagweights)
truebeta0 = cp.asarray(truebeta0)
truegamma0 = cp.asarray(truegamma0)
truebeta1 = cp.asarray(truebeta1)
truegamma1 = cp.asarray(truegamma1)

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

def HamCO(theta,t,p,field=False):
    # THE FOLLOWING IS IF YOU WANT TO SPECIFY ONLY v1,w1 as unknown parameters
    v0 = truebeta0
    v1 = theta[:(drc**2)*nprime].reshape((drc**2, nprime))
    w0 = truegamma0
    w1 = theta[(drc**2)*nprime:].reshape((drc**2, npripri))
    # produce Hamiltonian
    term0 = realmat @ v0
    x = cp.real(p).reshape((drc**2))
    term1 = (x @ v1) @ realmat.T
    term2 = imagmat @ w0
    y = cp.imag(p).reshape((drc**2))
    term3 = (y @ w1) @ imagmat.T
    rboth = (term0 + term1).reshape((drc,drc))
    iboth = (term2 + term3).reshape((drc,drc))
    h = rboth + 1j*iboth
    if field:
        freq = 0.0428
        if t > 2*np.pi/freq:
            ez = 0
        elif t < 0:
            ez = 0
        else:
            ez = 0.05*np.sin(0.0428*t)

        hfieldAO = cp.array(ez*didat[2], dtype=cp.complex128)
        h -= xmat.conj().T @ hfieldAO @ xmat
    return h

def MMUTHO_CP(theta, initial_density, dt=0.08268, ntvec=2000, field=False):
    P0 = initial_density.reshape((drc, drc))
    alldens = cp.zeros((ntvec, drc, drc), dtype=cp.complex128)
    alldens[0,:,:] = P0
    for i in range(ntvec-1):
        t = i*dt
        if i % 1000 == 0:
            print(i)
        P0 = alldens[i,:,:]
        k1 = -1j*dt*HamCO(theta,t,P0,field)
        Q1 = k1
        u2 = 0.5*Q1
        expu2, expmu2 = cexpm(u2)
        k2 = -1j*dt*HamCO(theta,t+0.5*dt,expu2 @ P0 @ expmu2,field)
        Q2 = k2 - k1
        u3 = 0.5*Q1 + 0.25*Q2
        expu3, expmu3 = cexpm(u3)
        k3 = -1j*dt*HamCO(theta,t+0.5*dt,expu3 @ P0 @ expmu3,field)
        Q3 = k3 - k2
        u4 = Q1 + Q2
        expu4, expmu4 = cexpm(u4)
        k4 = -1j*dt*HamCO(theta,t+dt,expu4 @ P0 @ expmu4,field)
        Q4 = k4 - 2*k2 + k1
        Q1Q2 = Q1 @ Q2 - Q2 @ Q1
        u5 = Q1/2.0 + Q2/4.0 + Q3/3.0 - Q4/24.0 - Q1Q2/48.0
        expu5, expmu5 = cexpm(u5)
        k5 = -1j*dt*HamCO(theta,t+0.5*dt,expu5 @ P0 @ expmu5,field)
        Q5 = k5 - k2
        u6 = Q1 + Q2 + (2.0/3.0)*Q3 + Q4/6.0 - Q1Q2/6.0
        expu6, expmu6 = cexpm(u6)
        k6 = -1j*dt*HamCO(theta,t+dt,expu6 @ P0 @ expmu6,field)
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

propden = cp.asnumpy(MMUTHO_CP(whichtheta, initial_density=ic, dt=dt, ntvec=mynumsteps, field=fieldFlag))
np.save(args.outfname, propden)

hermcheck = np.linalg.norm(propden - np.transpose(propden,[0,2,1]).conj())
propden2 = np.einsum('ijk,ikl->ijl',propden,propden,optimize=True)
idemcheck = np.linalg.norm(propden2 - propden)
print('Hermitian check: ' + str(hermcheck))
print('Idempotency check: ' + str(idemcheck))
