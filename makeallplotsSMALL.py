import numpy as np
import matplotlib.pyplot as plt
import matplotlib

scalefac = 100
dt = 0.08268/scalefac
if scalefac==1:
    dtstr = ''
else:
    dtstr = str(scalefac)

labelmap = {'OldSymm':'',
            'OldSymmE':'R',
            'OldSymmEH':'EH',
            'OldSymmEHR':'EHR',
            'Tied':'PG_lowtol',
            'TiedE':'PGR_lowtol',
            'Herm':'HB_lowtol',
            'HermE':'HBR_lowtol',
            'SymmH':'MO',
            'SymmHE':'MOR',
            'Symm':'MOiter',
            'SymmE':'MOiterR'}

colormap = {'OldSymm':'brown',
            'OldSymmE':'grey',
            'OldSymmEH':'red',
            'OldSymmEHR':'pink',
            'Tied':'cyan',
            'TiedE':'blue',
            'Herm':'lightgreen',
            'HermE':'darkgreen',
            'SymmH':'orange',
            'SymmHE':'brown',
            'Symm':'grey',
            'SymmE':'black'}    

mb = [['heh+','6-31g'],
      ['heh+','6-311G'],
      ['lih','6-31g'],
      ['c2h4','sto-3g']]

fp = [True, False]
zm = [True, False]

for zoom in zm:
    for molbasis in mb:
        mol = molbasis[0]
        basis = molbasis[1]
        for field in fp:
            if field==True:
                fieldstr = 'WF'
                onoff = 'on'
            else:
                fieldstr = ''
                onoff = 'off'

            traindata = np.load('./trajdata'+dtstr+fieldstr+'/train_'+mol+'_'+basis+'.npy')
            npts = traindata.shape[0]
            drc = traindata.shape[1]
            if zoom:
                labels = ['SymmE','HermE','TiedE']
                offset = 5000
                zoomstr = '_zoom'
            else:
                labels = ['SymmH','Symm','Herm','Tied','SymmHE','SymmE','HermE','TiedE']
                offset = 0
                zoomstr = ''

            allmae = []
            for label in labels:
                thistest = np.load('./testdata'+dtstr+fieldstr+'/test'+labelmap[label]+'_'+mol+'_'+basis+'.npy')
                abserr = (np.abs(traindata[:npts]-thistest[:npts])).reshape((-1,drc**2))
                allmae.append( np.mean(abserr,axis=1) )
                if not zoom:
                    print(mol,basis,field,label,np.max(np.abs(traindata[:npts]-thistest[:npts])))

            tvec = dt*np.arange(npts)
            colors = []
            for label in labels:
                colors.append(colormap[label])

            matplotlib.rcParams.update({'font.size': 16})

            plt.figure(figsize=(9,6))
            for j in range(0,len(allmae)):
                plt.semilogy(tvec[offset:],allmae[j][offset:],color=colors[j],label=labels[j])

            plt.legend(loc="center right", ncol=1, bbox_to_anchor=(1.3,0.5))
            plt.xlabel('time (a.u.)')

            plt.title(mol + ' in ' + basis + ' with field ' + onoff)
            plt.ylabel('mean abs error')
            plt.savefig('./figs/'+mol+'_'+basis+fieldstr+zoomstr+'_LT.pdf',bbox_inches='tight')
            plt.savefig('./figs/'+mol+'_'+basis+fieldstr+zoomstr+'_LT.png',bbox_inches='tight')
