from ase.io import read, gaussian
from ase.calculators.gaussian import Gaussian

REFERENCE_ENERGIES = {
    1: -13.62222753701504,
    6: -1029.4130839658328,
    7: -1484.8710358098756,
    8: -2041.8396277138045,
    9: -2712.8213146878606,
}

def get_molecular_reference_energy(atomic_numbers):
    molecular_reference_energy = 0
    for atomic_number in atomic_numbers:
        molecular_reference_energy += REFERENCE_ENERGIES[atomic_number]

    return molecular_reference_energy


def xyz2gjf(xyzfile, gjffile):
    mol = read(xyzfile)
    with open(gjffile, 'w') as f:
        gaussian.write_gaussian_in(f, mol, properties=['energy','force'],basis='6-31G*',method='wb97x',charge=0, mult=1)
    return 

def DFT_energy_ase(xyzfile):
    print (xyzfile)
    xyz2gjf(xyzfile, f"{xyzfile.strip('.xyz')}.gjf")
    mol = read(xyzfile)
    mol.calc = Gaussian(label='calc/gaussian',
                xc='wb97x',
                basis='6-31G*',
                scf='max',
                charge=0,
                mult=1
                )
    
    energy=mol.get_potential_energy()
    #forces=mol.get_forces()
    with open (f"{xyzfile.strip('.xyz')}.dft",'w') as f:
        f.write(f'Energy: {energy}\n')
        #f.write(f'Forces: \n')
        #for i in range(len(forces)):
        #    f.write(f'{forces[i][0]} {forces[0][1]} {forces[0][2]}\n')        
    return

def DFT_energy(xyzfile):
    print (xyzfile)
    xyz2gjf(xyzfile, f"{xyzfile.strip('.xyz')}.gjf")
    mol = read(xyzfile)
    import os 
    os.system('g16 '+f"{xyzfile.strip('.xyz')}.gjf")
    mol=read(f"{xyzfile.strip('.xyz')}.log",format='gaussian-out')
    energy=mol.get_potential_energy()
    #forces=mol.get_forces()
    print (energy)
    with open (f"{xyzfile.strip('.xyz')}.dft",'w') as f:
        f.write(f'Energy: {energy}\n')
    return

def DFT_energy_parallel(xyzfiles,nprocs=14):
    from multiprocessing import Pool,Queue,Manager,Process
    manager=Manager()
    DQueue=manager.Queue()
    njobs=len(xyzfiles)
    steps=2
    p=Pool(nprocs)
    resultlist=[]
    for i in range(njobs):
        result=p.apply_async(DFT_energy,(xyzfiles[i],))
        resultlist.append(result)
    for i in range(len(resultlist)):
        tmp=resultlist[i].get()
        print (tmp)
    p.terminate()
    p.join()
    print (f'DFT calculation finished!')
    return
"""
xyzpath='/home/myxu/MNT_191/myxu/EC_Ts/scripts/ASE/0'
#xyz2gjf(xyzfile=f'{xyzpath}/0.xyz', gjffile=f'{xyzpath}/0.gjf')
energy=DFT_energy(xyzfile=f'{xyzpath}/0.xyz')   
print (energy)
"""

