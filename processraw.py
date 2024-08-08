import numpy as np

molsys = {}
mols = ['c2h4','heh+','lih']
bases = [['6-31pgs','sto-3g'],['6-311G','6-31g'],['6-311ppgss','6-31g']]
# mols = ['c6h6n2o2']
# bases = [['sto-3g']]
for j in range(3):
    mol = mols[j]
    for basis in bases[j]:
        overlap = np.load('./moldata/ke+en+overlap+ee_twoe+dip_hf_delta_s0_'+mol+'_'+basis+'.npz',
                          allow_pickle=True)
        sevalsfname = './rawtraj/sevals_'+mol+'_'+basis+'.npy'
        sevals = np.load(sevalsfname)
        sevecsfname = './rawtraj/sevecs_'+mol+'_'+basis+'.npy'
        sevecs = np.load(sevecsfname)
        xmat = sevecs @ np.diag(sevals**(-0.5))
        fname = './rawtraj/td_dens_re+im_rt-tdexx_delta_s0_'+mol+'_'+basis+'.npz'
        denAOnpz = np.load(fname)
        denAO = denAOnpz['td_dens_re_data'] + 1j*denAOnpz['td_dens_im_data']
        denAOinit = denAO[2,:,:]
        denMOinit = np.diag(sevals**(0.5)) @ sevecs.T @ denAOinit @ sevecs @ np.diag(sevals**(0.5))
        print('Computed denMOinit!')
        hermcheck = np.linalg.norm(denMOinit - denMOinit.conj().T)
        print('Hermitian check: ' + str(hermcheck))
        idemcheck = np.linalg.norm(denMOinit @ denMOinit - denMOinit)
        print('Idempotency check: ' + str(idemcheck))
        outfname = './moldata/denMOtrueinit_'+mol+'_'+basis+'.npy'
        np.save(outfname, denMOinit)
