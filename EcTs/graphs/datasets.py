import pickle 
import os 
import numpy as np 
from rdkit import Chem
from .mol import * 
import random 
import math 
from tqdm import tqdm 
from torch.utils.data import Dataset,DataLoader
from ..comparm import *
import copy

def Deal_GEOM_Dataset_with_energy(flist,max_conf_per_conf,save_path='molgraphs.pickle',max_atoms=50):
    molgraphs=[]
    with open('error.list','a') as errf:
        for fname in tqdm(flist):
            with open(f'{fname}','rb') as f:
                #try:
                if True:
                    a=pickle.load(f)
                    conformers=a['conformers']
                    energies=[]
                    for conf in conformers:
                        energies.append(conf['totalenergy'])
                    energies=np.array(energies)
                    lowest_ids=np.argsort(energies)[:min(max_conf_per_conf,len(conformers))]
                    selected_mols=[conformers[i]['rd_mol'] for i in lowest_ids]
                    molsupp=Chem.SDWriter(fname.strip('\.pickle')+'.sdf')
                    for mol in selected_mols:
                        molsupp.write(mol)
                        try:
                            mol_noH=Chem.rdmolops.RemoveHs(mol,sanitize=True)
                            mol_noH=Neutralize_atoms(mol)
                            Chem.Kekulize(mol_noH)
                            atoms=[atom.GetAtomicNum() for atom in mol_noH.GetAtoms()]
                            natoms=sum([1 for i in atoms if i!=1])
                            smi=Chem.MolToSmiles(mol_noH)
                            if natoms>3 and natoms < max_atoms and '.' not in smi:
                                molgraph=Molgraph(mol_noH,smiles=a['smiles'])
                                #print (max(sum(np.where(molgraph.adjs,1,0))))
                                if max(sum(np.where(molgraph.adjs,1,0)))<=4:
                                    molgraphs.append(molgraph)
                                else:
                                    print (fname)
                                    errf.write(f'{fname}\n')
                        except Exception as e:
                            print (fname)
                            errf.write(f'{fname}\n')
                    molsupp.close()
                #except:
                #    print (fname)
                #    errf.write(f'{fname}\n')
    with open(save_path,'wb') as f:
        pickle.dump(molgraphs,f)
    return 

def Deal_GEOM_Dataset(flist,max_conf_per_conf,save_path='molgraphs.pickle',max_atoms=50):
    molgraphs=[]
    bad_case=0
    with open('error.list','a') as errf:
        for fname in tqdm(flist):
            try:
                with open(f'{fname}','rb') as f:
                    a=pickle.load(f)
                    if a.get('uniqueconfs') > len(a.get('conformers')):
                        bad_case += 1
                        continue
                    if a.get('uniqueconfs') <= 0:
                        bad_case += 1
                        continue
                    if a.get('uniqueconfs') == max_conf_per_conf:
                        # use all confs
                        conf_ids = np.arange(a.get('uniqueconfs'))
                    else:
                        # filter the most probable 'conf_per_mol' confs
                        all_weights = np.array([_.get('boltzmannweight', -1.) for _ in a.get('conformers')])
                        descend_conf_id = (-all_weights).argsort()
                        conf_ids = descend_conf_id[:max_conf_per_conf]
                    conformers=a['conformers']
                    selected_mols=[conformers[i]['rd_mol'] for i in conf_ids]
                    molsupp=Chem.SDWriter(save_path+'/'+fname[49:].strip('\.pickle')+'.sdf')
                    for mol in selected_mols:
                        molsupp.write(mol)
                        try:
                            mol_noH=Chem.rdmolops.RemoveHs(mol,sanitize=True)
                            mol_noH=Neutralize_atoms(mol)
                            Chem.Kekulize(mol_noH)
                            atoms=[atom.GetAtomicNum() for atom in mol_noH.GetAtoms()]
                            natoms=sum([1 for i in atoms if i!=1])
                            smi=Chem.MolToSmiles(mol_noH)
                            if natoms>3 and natoms < max_atoms and '.' not in smi:
                                molgraph=Molgraph(mol_noH,smiles=a['smiles'])
                                #print (max(sum(np.where(molgraph.adjs,1,0))))
                                if max(sum(np.where(molgraph.adjs,1,0)))<=4:
                                    molgraphs.append(molgraph)
                                else:
                                    print (fname)
                                    errf.write(f'{fname}\n')
                        except Exception as e:
                            print (fname)
                            errf.write(f'{fname}\n')
                    molsupp.close()
            except:
                print (fname)
                errf.write(f'{fname}\n')
                pass

    #with open(save_path,'wb') as f:
    #    pickle.dump(molgraphs,f)
    return 

def Multi_Process_Creat_GEOM_Molgraphs(flist,nmols_per_process,max_conf_per_mol,savepath='./molgraphs',nprocs=14):
    from multiprocessing import Pool,Queue,Manager,Process
    manager=Manager()
    DQueue=manager.Queue()
    nmols=len(flist)
    njobs=math.ceil(nmols/nmols_per_process)

    if not os.path.exists(savepath):
        os.system(f'mkdir -p {savepath}')
    with open(f'{savepath}/flist.csv','w') as f:
        for fname in flist:
            f.write(fname+'\n')

    p=Pool(nprocs)
    resultlist=[]
    for i in range(njobs):
        result=p.apply_async(Deal_GEOM_Dataset,(flist[i*nmols_per_process:(i+1)*nmols_per_process],max_conf_per_mol,f'{savepath}'))
        resultlist.append(result)
    for i in range(len(resultlist)):
        tmp=resultlist[i].get()
        print (tmp)
    p.terminate()
    p.join()
    print (f'Mols have all trans to Molgraphs in {savepath}')
    return 

def Grep_T1X_Path(rxnid,savepath):
    with open(f'{savepath}/{rxnid}_final.pkl','rb') as f:
        molecules=pickle.load(f)
        fmids=[]
        fenergies=[]
        for mid,molecule in enumerate(molecules):
            fmids.append(mid)
            fenergies.append(molecule['wB97x_6-31G(d).energy'])
    
    with open(f'{savepath}/{rxnid}.pkl','rb') as f:
        molecules=pickle.load(f)
        pathes=[]
        reagent=molecules[0]
        product=molecules[1]
        pts=[2]
        for mid,molecule in enumerate(molecules):
            if mid >2 and mid < len(molecules)-1:
                if (molecule['wB97x_6-31G(d).energy'] < molecules[mid+1]['wB97x_6-31G(d).energy']) and (molecule['wB97x_6-31G(d).energy'] < molecules[mid-1]['wB97x_6-31G(d).energy']):
                    pts.append(mid)
        
        pts.append(len(molecules))
        pathes=[]
        for i in range(len(pts)-1):
            path=[]
            path.append(reagent)
            for j in range(pts[i],pts[i+1]):
                path.append(molecules[j])
            path.append(product)
            pathes.append(path)
        with open (f'{savepath}/{rxnid}_final_path.pkl','wb') as f1:
            pickle.dump(pathes[-1],f1)

def Statistic_GPARAMS(MGFiles):
    params={'atom_types':[],'max_atoms':0}
    Hmols=0
    for fname in MGFiles:
        with open(fname,'rb') as f:
            mgs=pickle.load(f)
        for mg in tqdm(mgs):
            if mg.natoms > params["max_atoms"]:
                params["max_atoms"]=mg.natoms
            if 1 in mg.atoms:
                Hmols+=1
            for a in mg.atoms:
                if a not in params["atom_types"]:
                    params["atom_types"].append(a)
    params["atom_types"]=np.sort(params["atom_types"])
    print (Hmols)
    return params

def Statistic_GPARAMS_for_RPs(RPFiles):
    params={'atom_types':[],'max_atoms':0}

    for fname in RPFiles:
        with open(fname,'rb') as f:
            mgs=pickle.load(f)
        for mg in tqdm(mgs):
            if mg.natoms > params["max_atoms"]:
                params["max_atoms"]=mg.natoms
            for a in mg.atoms:
                if a not in params["atom_types"]:
                    params["atom_types"].append(a)
    params["atom_types"]=np.sort(params["atom_types"])

    return params

class MG_Dataset(Dataset):
    def __init__(self,MGlist,name):
        super(Dataset,self).__init__()
        self.mglist=MGlist
        self.name=name
        self.nmols=len(self.mglist)
        self.max_atoms=GP.max_atoms
        return 
    def __len__(self):
        return len(self.mglist)
    def __getitem__(self,idx):
        return self.getitem__(idx)
    def getitem__(self,idx):
        mg=copy.deepcopy(self.mglist[idx])
            
        atoms,chiraltags,adjs,coords,zmats,masks=mg.Get_3D_Graph_Tensor_Ts(max_atoms=self.max_atoms)
        if not GP.if_chiral:
            feats=atoms
        else:
            feats=torch.concat((atoms,chiraltags),axis=-1)
        return {
                'Feats':feats,
                "Adjs":adjs,
                "Coords":coords,
                "Zmats":zmats,
                "Masks":masks
                }

class RP_Dataset(Dataset):
    def __init__(self,RPlist,name='RP_datasets'):
        super(Dataset,self).__init__()
        self.rplist=RPlist
        self.name=name
        self.npairs=len(self.rplist)
        self.max_atoms=GP.max_atoms
        self.max_bonds=GP.max_bonds
        self.max_angles=GP.max_angles
        self.max_torsions=GP.max_torsions
        return 
    
    def __len__(self):
        return len(self.rplist)
    
    def __getitem__(self,idx):
        return self.getitem_Ts__(idx)

    
    def getitem_Ts__(self,idx):
        rp_pair=copy.deepcopy(self.rplist[idx])
        ratoms,patoms,radjs,padjs,redges,pedges,rcoords,pcoords,tscoords,masks,\
                zb,zb_masks,za,za_masks,zd,zd_masks,\
                renergy,penergy,tsenergy,rforces,pforces,tsforces=rp_pair.Get_3D_Graph_Tensor_Ts(max_atoms=self.max_atoms,
                                                                                                 max_bonds=self.max_bonds,
                                                                                                 max_angles=self.max_angles,
                                                                                                 max_torsions=self.max_torsions)
        return {
                'RFeats':ratoms,
                'PFeats':patoms,
                "RAdjs":radjs,
                "PAdjs":padjs,
                "REdges":redges,
                "PEdges":pedges,
                "RCoords":rcoords,
                "PCoords":pcoords,
                "TsCoords":tscoords,
                "Zb":zb,
                "Zb_Masks":zb_masks,
                "Za":za,
                "Za_Masks":za_masks,
                "Zd":zd,
                "Zd_Masks":zd_masks,
                "REnergies":renergy,
                "PEnergies":penergy,
                "TsEnergies":tsenergy,
                "RForces":rforces,
                "PForces":pforces,
                "TsForces":tsforces,
                "Masks":masks
                }


class RP_Confidence_Dataset(Dataset):
    def __init__(self,RP_Confidence_list,name='RP_confidence_datasets'):
        super(Dataset,self).__init__()
        self.rp_confidence_list=RP_Confidence_list
        self.name=name
        self.npairs=len(self.rp_confidence_list)

        self.max_atoms=GP.max_atoms
        self.max_bonds=GP.max_bonds
        self.max_angles=GP.max_angles
        self.max_torsions=GP.max_torsions

        return 
    
    def __len__(self):
        return len(self.rp_confidence_list)
    
    def __getitem__(self,idx):
        return self.getitem_Ts__(idx)
    
    def getitem_Ts__(self,idx):
        try:
            #print (idx,len(self.rp_confidence_list),self.name,self.npairs)
            rp_pair=self.rp_confidence_list[idx][0]
            #print (self.rp_confidence_list[idx])
            label=torch.Tensor([self.rp_confidence_list[idx][1]]).long()

            ratoms,patoms,radjs,padjs,redges,pedges,rcoords,pcoords,tscoords,masks,\
                zb,zb_masks,za,za_masks,zd,zd_masks,\
                renergy,penergy,tsenergy,rforces,pforces,tsforces=rp_pair.Get_3D_Graph_Tensor_Ts(max_atoms=self.max_atoms,
                                                                                                 max_bonds=self.max_bonds,
                                                                                                 max_angles=self.max_angles,
                                                                                                 max_torsions=self.max_torsions)
        
            return {
                'RFeats':ratoms,
                'PFeats':patoms,
                "RAdjs":radjs,
                "PAdjs":padjs,
                "REdges":redges,
                "PEdges":pedges,
                "RCoords":rcoords,
                "PCoords":pcoords,
                "TsCoords":tscoords,
                "Masks":masks,
                "Labels":label
                }
        except:
            print (idx,len(self.rp_confidence_list),self.name,self.npairs)
            return None

        



    



        