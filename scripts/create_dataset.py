from tqdm import tqdm 
from rdkit import Chem
import pickle 
from EcTs.graphs import Grep_T1X_Path,RP_pair
from EcTs.utils import xyz2mol,xyz2AC
import os 
import numpy as np 
sourcepath='/mnt_191/myxu/EC_Ts/scripts/Transition1x_RP_Pairs'
rpts_pairs=[]

for i in tqdm(range(10073)):
    try:
        if i< 10000:
            rxnid=f'rxn{i:04d}'
        else:
            rxnid=f'rxn{i:05d}'
        with open(f'{sourcepath}/{rxnid}_final.pkl','rb') as f:
            molecules=pickle.load(f)
            ratoms=[int(i) for i in molecules[0]['atomic_numbers']]
            rcoords=molecules[0]['positions']
            #radjs,rmol=xyz2AC(ratoms,rcoords,charge=0)
            rmol=xyz2mol(ratoms,rcoords,charge=0)[0]
            renergy=molecules[0]['wB97x_6-31G(d).atomization_energy']
            rforces=molecules[0]['wB97x_6-31G(d).forces']
 
            patoms=[int(i) for i in molecules[1]['atomic_numbers']]
            pcoords=molecules[1]['positions']
            pmol=xyz2mol(patoms,pcoords,charge=0)[0]
            #padjs,pmol=xyz2AC(patoms,pcoords,charge=0)
            penergy=molecules[1]['wB97x_6-31G(d).atomization_energy']
            pforces=molecules[1]['wB97x_6-31G(d).forces']
 
            tsatoms=[int(i) for i in molecules[2]['atomic_numbers']]
            tscoords=molecules[2]['positions']
            tsenergy=molecules[1]['wB97x_6-31G(d).atomization_energy']
            tsforces=molecules[1]['wB97x_6-31G(d).forces']
 
            tsadjs,tsmol=xyz2AC(tsatoms,tscoords,charge=0)
            rp_pair=RP_pair(rmol=rmol,pmol=pmol,
                            tsatoms=tsatoms,
                            tsadjs=tsadjs,
                            tscoords=tscoords,
                            renergy=renergy,penergy=penergy,tsenergy=tsenergy,
                            rforces=rforces,pforces=pforces,tsforces=tsforces,
                            idx=rxnid)
 
            rpts_pairs.append(rp_pair)
    except:
        pass

import random
import math 

random.shuffle(rpts_pairs)
totalnum=len(rpts_pairs)
print (totalnum)
cutnum=math.ceil(totalnum*0.9)
with open('rp_pairs_train.pkl','wb') as f:
    pickle.dump(rpts_pairs[:cutnum],f)
with open('rp_pairs_test.pkl','wb') as f:
    pickle.dump(rpts_pairs[cutnum:],f)

        
        
