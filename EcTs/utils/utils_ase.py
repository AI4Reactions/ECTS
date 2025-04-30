from ase import io
from ase.mep.neb import NEB,NEBOptimizer
from ase.optimize import MDMin

from ase import Atoms
from ase.calculators.emt import EMT
from copy import deepcopy 
import math 
import torch 
import numpy as np 

def NEB_step(atom_numbers, coords, calc, r_coords,p_coords,weight=1):
    # Set up the NEB calculation
    n_images = len(coords)
    try: 
        natoms=len(atom_numbers)
        pos_ori=coords.clone().detach().cpu().numpy()[:,:natoms] 
        centers=pos_ori.mean(axis=1)
        pos_ori=pos_ori-centers[:,None]
        r_pos=r_coords.clone().detach().cpu().numpy()[:natoms]
        p_pos=p_coords.clone().detach().cpu().numpy()[:natoms]
        
        #print (r_pos.shape,p_pos.shape)
        images = [Atoms(numbers=atom_numbers,positions=r_pos)]+\
                [Atoms(numbers=atom_numbers,positions=pos_ori[i]) for i in range(n_images)]+\
                [Atoms(numbers=atom_numbers,positions=p_pos)]
                
        for image in images[1:-1]:
            image.calc=deepcopy(calc)
        
        neb = NEB(images)
        opt = NEBOptimizer(neb, trajectory='neb.traj',method='static',alpha=0.001)
        
        pos=opt.neb.get_positions().reshape(-1)
        
        F=opt.force_function(pos).reshape(-1,natoms,3)
        forces=torch.zeros_like(coords).to(coords)
        forces[:,:natoms]=torch.from_numpy(F).to(coords)
        energies=np.array(opt.neb.energies)
        
        images[0].calc=deepcopy(calc)
        images[-1].calc=deepcopy(calc)
        energies[0]=images[0].get_potential_energy()
        energies[-1]=images[-1].get_potential_energy()
        fmax=opt.get_residual()
    except Exception as e:
        forces=torch.zeros_like(coords).to(coords)
        energies=np.ones(n_images+2)*10000
        fmax=10000
        print (f'neb step failed due to {e}')
    return forces,fmax,energies

