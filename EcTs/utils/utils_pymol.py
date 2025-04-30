import numpy as np
import os
from tqdm import tqdm
import rdkit
from rdkit.Chem import rdmolfiles
from rdkit import Chem

def write_pymol_script(fname,path,scriptname,outputname):
    with open (f'{path}/{scriptname}','w') as f:
        f.write(f'load {fname}\n')
        f.write('hide spheres\n')
        f.write('show sticks\n')
        f.write('set stick_ball, on\n')
        f.write('set stick_radius, 0.2\n')
        f.write('set stick_ball_ratio, 1.5\n')
        f.write('ray 512, 512\n')
        f.write(f'save {outputname}.pse\n')
        f.write(f'png {outputname}.png,dpi=300\n')

def write_combine_pymol_script(flist,path,scriptname,outputname):
    with open (f'{path}/{scriptname}','w') as f:
        for fname in flist:
            f.write(f'load {fname}\n')
        f.write('hide spheres\n')
        f.write('show sticks\n')
        f.write('set stick_ball, on\n')
        f.write('set stick_radius, 0.2\n')
        f.write('set stick_ball_ratio, 1.5\n')
        f.write('ray 512, 512\n')
        f.write(f'save {outputname}.pse\n')
        #f.write(f'png {outputname}.png,dpi=300\n') 

