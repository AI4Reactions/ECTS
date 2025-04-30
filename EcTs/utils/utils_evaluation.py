from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np 
import numpy as np
from scipy.spatial.distance import cdist
import torch 
from .utils_rdkit import read_xyz_file 
from .utils_rmsd import kabsch_rmsd,pymatgen_rmsd,xyz2pmg

def calc_performance_stats(true_mols, model_mols):

    threshold = np.arange(0, 2.0, .1)
    rmsd_list = []
    for tc in true_mols:
        for mc in model_mols:
            try:
                rmsd_val = AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
            except RuntimeError:
                return None
            rmsd_list.append(rmsd_val)

    rmsd_array = np.array(rmsd_list).reshape(len(true_mols), len(model_mols))

    coverage_recall = np.sum(rmsd_array.min(axis=1, keepdims=True) < threshold, axis=0) / len(true_mols)
    amr_recall = rmsd_array.min(axis=1).mean()

    coverage_precision = np.sum(rmsd_array.min(axis=0, keepdims=True) < np.expand_dims(threshold, 1), axis=1) / len(model_mols)
    amr_precision = rmsd_array.min(axis=0).mean()

    return coverage_recall, amr_recall, coverage_precision, amr_precision

def D_MAE(ref_xyz, gen_xyz):
    distance_matrix_ref = cdist(ref_xyz, ref_xyz)
    distance_matrix_ref_triu=np.triu(distance_matrix_ref,k=1)
    distance_matrix_gen = cdist(gen_xyz, gen_xyz)
    distance_matrix_gen_triu=np.triu(distance_matrix_gen,k=1)

    diff=np.sum(np.abs(distance_matrix_ref-distance_matrix_gen))*2/(distance_matrix_gen.shape[0]*(distance_matrix_gen.shape[0]-1))
    return np.mean(diff)

def path_rmsds(predicted_path_files,ref_path_files):
    refxyzs=[]
    for fname in ref_path_files:
        atoms,charge,xyzs=read_xyz_file(fname)
        refxyzs.append(xyzs)
    predxyzs=[]
    for fname in predicted_path_files:
        atoms,charge,xyzs=read_xyz_file(fname)
        predxyzs.append(xyzs)
    path_rmsds=[]
    for refxyz in refxyzs:
        rmsds=[]
        for predxyz in predxyzs:
            rmsd=kabsch_rmsd(refxyz,predxyz)
            rmsds.append(rmsd)
        point_rmsd=np.min(rmsds)
        path_rmsds.append(point_rmsd)
    mean=np.mean(path_rmsds,axis=0)
    return mean

def path_rmsds_pymatgen(predicted_path_files,ref_path_files):
    refxyzs=[]
    for fname in ref_path_files:
        mol=xyz2pmg(fname)
        refxyzs.append(mol)
    predxyzs=[]
    for fname in predicted_path_files:
        mol=xyz2pmg(fname)
        predxyzs.append(mol)
    path_rmsds=[]
    for refxyz in refxyzs:
        rmsds=[]
        for predxyz in predxyzs:
            try:
                rmsd=pymatgen_rmsd(refxyz,predxyz,ignore_chirality=True,threshold=0.5,same_order=True)
            except:
                rmsd=2.0
            rmsds.append(np.min([rmsd,2.0]))
        point_rmsd=np.min(rmsds)
        path_rmsds.append(point_rmsd)
    mean=np.mean(path_rmsds,axis=0)
    return mean


        
