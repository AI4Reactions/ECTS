from rdkit import Chem 
from rdkit.Chem import AllChem
from ..utils.utils_np import *
from ..comparm import *
import copy
import torch 
import networkx as nx
from ..utils.utils_graphroute import *
from ..utils.utils_rdkit import *
import random 
from .mol import Atoms_to_Idx,Atoms_to_Onek, Adjs_to_Onek
try:
    import iodata

    from iodata.utils import LineIterator as lit
    from rdkit.Chem import rdDetermineBonds
except:
    pass
from tqdm import tqdm 

def qchem2rdmol(qchemlogfile,):
    try:
        mol=iodata.load_one(qchemlogfile,fmt='qchemlog')
        #print (mol.keys())
        #print (mol)
        xyzfilename=f'{qchemlogfile[:-4]}.xyz'
        with open (xyzfilename,'w') as f:
            iodata.formats.xyz.dump_one(f,mol)
        atcoords=mol.atcoords
        mol=Chem.MolFromXYZFile(xyzfilename)
        rdDetermineBonds.DetermineBonds(mol,charge=0)
        return mol,atcoords
    except:
        return None,None

class RP_pair:
    def __init__(self,rmol=None,pmol=None,tsmol=None,
                 ratoms=None,patoms=None,tsatoms=None,
                 radjs=None,padjs=None,tsadjs=None,
                 rcoords=None,pcoords=None,tscoords=None,
                 renergy=None,penergy=None,tsenergy=None,
                 rforces=None,pforces=None,tsforces=None,
                 idx=None):
        if rmol is not None and pmol is not None:
            self.ratoms=[atom.GetAtomicNum() for atom in rmol.GetAtoms()]
            self.patoms=[atom.GetAtomicNum() for atom in pmol.GetAtoms()]
            self.ratoms=np.array(self.ratoms)
            self.patoms=np.array(self.patoms)
        else:
            assert ratoms is not None and patoms is not None, "Either reactants and products or atom numbers should be provided"
            self.ratoms=ratoms
            self.patoms=patoms
        
        self.idx=idx

        assert len(self.ratoms)==len(self.patoms) , "Atom numbers should be same in reactants and products"

        #print (self.chiraltags)
        self.natoms=len(self.ratoms)

        if rmol is not None and pmol is not None:
            for i in range(self.natoms):
                assert self.patoms[i] == self.ratoms[i], "Atoms should in the same order in reactants and products"
            self.radjs=np.zeros((self.natoms,self.natoms))
            self.padjs=np.zeros((self.natoms,self.natoms))

            for bond in rmol.GetBonds():
                a1=bond.GetBeginAtom().GetIdx()
                a2=bond.GetEndAtom().GetIdx()
                bt=bond.GetBondType() 
                ch=GP.bond_types.index(bt)
                self.radjs[a1,a2]=ch+1
                self.radjs[a2,a1]=ch+1

            for bond in pmol.GetBonds():
                a1=bond.GetBeginAtom().GetIdx()
                a2=bond.GetEndAtom().GetIdx()
                bt=bond.GetBondType() 
                ch=GP.bond_types.index(bt)
                self.padjs[a1,a2]=ch+1
                self.padjs[a2,a1]=ch+1 

            self.rcoords=np.array(rmol.GetConformer(0).GetPositions())
            self.pcoords=np.array(pmol.GetConformer(0).GetPositions())
            
        else:
            self.radjs=radjs
            self.padjs=padjs
            self.rcoords=rcoords
            self.pcoords=pcoords
            if tscoords is not None:
                self.tscoords=tscoords
            else:
                self.tscoords=np.zeros((self.natoms,3))

        padjs_onek=Adjs_to_Onek(self.padjs)
        self.pzmats=np_adjs_to_zmat(padjs_onek)[:,:4]

        if tsmol is not None:
            self.tsatoms=[atom.GetAtomicNum() for atom in tsmol.GetAtoms()]
            self.tsadjs=np.zeros((self.natoms,self.natoms))
            for bond in tsmol.GetBonds():
                a1=bond.GetBeginAtom().GetIdx()
                a2=bond.GetEndAtom().GetIdx()
                bt=bond.GetBondType() 
                ch=GP.bond_types.index(bt)
                self.tsadjs[a1,a2]=ch+1
                self.tsadjs[a2,a1]=ch+1 
        else:
            self.tsatoms=tsatoms
            if tsadjs is not None:
                self.tsadjs=tsadjs
            else:
                self.tsadjs=np.zeros((self.natoms,self.natoms))

        if tscoords is not None:
            self.tscoords=np.array(tscoords)
        else:
            self.tscoords=np.zeros((self.natoms,3))

        if tsadjs is not None:
            tsadjs_onek=Adjs_to_Onek(self.tsadjs)
            self.tszmats=np_adjs_to_zmat(tsadjs_onek)[:,:4]
        else:
            self.tszmats=np.zeros((self.natoms,4))

        self.renergy=renergy
        self.rforces=rforces

        self.penergy=penergy
        self.pforces=pforces

        self.tsenergy=tsenergy
        self.tsforces=tsforces

        return 

    def Get_3D_Graph_Tensor_Ts(self,max_atoms= None,max_bonds=None,max_angles=None,max_torsions=None):
        if max_atoms is None:
            max_atoms=self.natoms

        zb_,za_,zd_=Adjs_to_IC_targets(self.tsadjs)
        #print (zb_)

        if max_bonds is None:
            max_bonds=len(zb_)
        if max_angles is None:
            max_angles=len(za_)
        if max_torsions is None:
            max_torsions=len(zd_)
            
        #print (max_bonds,max_angles,max_torsions)
        
        redges=np.where(self.radjs>0,1,0)
        pedges=np.where(self.padjs>0,1,0)

        radjs_mat=torch.zeros((max_atoms,max_atoms,len(GP.bond_types)+1)).long()
        padjs_mat=torch.zeros((max_atoms,max_atoms,len(GP.bond_types)+1)).long()
        redges_mat=torch.zeros((max_atoms,max_atoms,len(GP.bond_types)+1)).long()
        pedges_mat=torch.zeros((max_atoms,max_atoms,len(GP.bond_types)+1)).long()

        rcoords=torch.zeros((max_atoms,3))
        pcoords=torch.zeros((max_atoms,3))
        tscoords=torch.zeros((max_atoms,3))
        masks=torch.zeros(max_atoms).bool()
        masks[:self.natoms]=True
        
        ratom_feat=torch.zeros(max_atoms,len(GP.atom_types)).long()
        patom_feat=torch.zeros(max_atoms,len(GP.atom_types)).long()
        ratom_feat_=Atoms_to_Onek(self.ratoms,GP.atom_types)
        patom_feat_=Atoms_to_Onek(self.patoms,GP.atom_types)
        ratom_feat[:self.natoms]=torch.Tensor(ratom_feat_).long()
        patom_feat[:self.natoms]=torch.Tensor(patom_feat_).long()

        radjs_mat[:self.natoms,:self.natoms]=torch.Tensor(Adjs_to_Onek(self.radjs,len(GP.bond_types)+1)).long()
        padjs_mat[:self.natoms,:self.natoms]=torch.Tensor(Adjs_to_Onek(self.padjs,len(GP.bond_types)+1)).long()
        redges_mat[:self.natoms,:self.natoms]=torch.Tensor(Adjs_to_Onek(redges,len(GP.bond_types)+1)).long()
        pedges_mat[:self.natoms,:self.natoms]=torch.Tensor(Adjs_to_Onek(pedges,len(GP.bond_types)+1)).long()

        #print (self.rcoords,rcoords[:self.natoms].shape,self.natoms)
        #print (torch.Tensor(self.rcoords)-torch.mean(torch.Tensor(self.rcoords),dim=0,keepdims=True))
        rcoords[:self.natoms]=torch.Tensor(self.rcoords)-torch.mean(torch.Tensor(self.rcoords),dim=0,keepdims=True)

        pcoords[:self.natoms]=torch.Tensor(self.pcoords)-torch.mean(torch.Tensor(self.pcoords),dim=0,keepdims=True)
        
        zb=torch.zeros((max_bonds,2)).long()
        za=torch.zeros((max_angles,3)).long()
        zd=torch.zeros((max_torsions,4)).long()

        zb_mask=torch.zeros(max_bonds).bool()
        za_mask=torch.zeros(max_angles).bool()
        zd_mask=torch.zeros(max_torsions).bool()

        zb_mask[:zb_.shape[0]]=True
        za_mask[:za_.shape[0]]=True
        zd_mask[:zd_.shape[0]]=True
        zb[:zb_.shape[0]]=torch.Tensor(zb_).long()
        za[:za_.shape[0]]=torch.Tensor(za_).long()
        zd[:zd_.shape[0]]=torch.Tensor(zd_).long()

        if np.sum(self.tscoords)!=0:
            tscoords[:self.natoms]=torch.Tensor(self.tscoords)-torch.mean(torch.Tensor(self.tscoords),dim=0,keepdims=True)
        if self.renergy is not None:
            renergy=torch.Tensor([self.renergy]).float()
        else:
            renergy=torch.zeros(1).float()
        if self.penergy is not None:
            penergy=torch.Tensor([self.penergy]).float()
        else:
            penergy=torch.zeros(1).float()
        if self.tsenergy is not None:
            tsenergy=torch.Tensor([self.tsenergy]).float()
        else:
            tsenergy=torch.zeros(1).float()

        rforces=torch.zeros((max_atoms,3)).float()
        pforces=torch.zeros((max_atoms,3)).float()
        tsforces=torch.zeros((max_atoms,3)).float()
        if self.rforces is not None:
            rforces[:self.natoms]=torch.Tensor(self.rforces).float()
        if self.pforces is not None:
            pforces[:self.natoms]=torch.Tensor(self.pforces).float()
        if self.tsforces is not None:
            tsforces[:self.natoms]=torch.Tensor(self.tsforces).float()
        
        return  ratom_feat,patom_feat,radjs_mat,padjs_mat,redges_mat,pedges_mat,rcoords,pcoords,tscoords,masks,\
                zb,zb_mask,za,za_mask,zd,zd_mask,\
                renergy,penergy,tsenergy,rforces,pforces,tsforces
        


         
