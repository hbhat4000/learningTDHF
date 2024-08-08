import argparse
import os
import os.path
import numpy as np

parser = argparse.ArgumentParser(prog='randic',
                                 description='Generate random initial condition!')
parser.add_argument('--mol', required=True, help='name of molecule')
parser.add_argument('--basis', required=True, help='name of basis set')
parser.add_argument('--fac', type=float, required=True, help='factor that governs noise amplitude')
parser.add_argument('--postfix', required=True, help='identifying string to use after the word train when saving the initial condition')

args = parser.parse_args()

mol = args.mol
basis = args.basis
overlap = np.load('./moldata/ke+en+overlap+ee_twoe+dip_hf_delta_s0_'+mol+'_'+basis+'.npz',allow_pickle=True)

eeten = overlap['ee_twoe_data']
drc = eeten.shape[0]

# load the true IC
denMOinit = np.load('./moldata/denMOtrueinit'+'_'+mol+'_'+basis+'.npy').reshape((drc,drc))

# first add a small Hermitian perturbation
noiseraw = np.random.normal(size=drc**2) + 1j*np.random.normal(size=drc**2)
noiseraw = noiseraw.reshape((drc,drc))
noise = 0.5*(noiseraw + noiseraw.conj().T)
mae = np.mean(np.abs(denMOinit))
amp = args.fac*mae
denMOinitR = denMOinit + amp*noise

# next compute eigendecomposition
icevals, icevecs = np.linalg.eigh(denMOinitR)

# make the initial condition idempotent
icevals[icevals>0.5] = 1.0
icevals[icevals<=0.5] = 0.0

# put it back together
ic = icevecs @ np.diag(icevals) @ icevecs.conj().T

# save randomized IC with postfix
postfix = args.postfix
np.save('./moldata/denMOtrueinit'+postfix+'_'+mol+'_'+basis+'.npy',ic)
