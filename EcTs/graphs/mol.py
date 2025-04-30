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

class Molgraph:
    def __init__(self,rdkitmol,smiles='',RemoveHs=True,permindex='fix'):
        self.smiles=smiles
        self.atoms=[atom.GetAtomicNum() for atom in rdkitmol.GetAtoms()]
        self.atoms=np.array(self.atoms)
        #print (self.chiraltags)
        self.natoms=len(self.atoms)
        self.adjs=np.zeros((self.natoms,self.natoms))
        for bond in rdkitmol.GetBonds():
            a1=bond.GetBeginAtom().GetIdx()
            a2=bond.GetEndAtom().GetIdx()
            bt=bond.GetBondType() 
            ch=GP.bond_types.index(bt)
            self.adjs[a1,a2]=ch+1
            self.adjs[a2,a1]=ch+1
        try:
            self.coords=np.array(rdkitmol.GetConformer(0).GetPositions())
        except:
            self.coords=np.zeros((self.natoms,3))
        #self.Standardrize(permindex,RemoveHs)
        return 
    
    def RemoveHs(self):
        noH_idx=np.array([i for i in range(len(self.atoms)) if self.atoms[i]!=1])
        n_heavy_atoms=len(noH_idx)
        coords=self.coords[noH_idx]
        adjs=self.adjs[np.ix_(noH_idx,noH_idx)]
        self.atoms=self.atoms[noH_idx]

        self.adjs=adjs
        self.chiraltags=self.chiraltags[noH_idx]        
        self.coords=coords
        self.natoms=n_heavy_atoms
        return
    
    def PermIndex(self,mode='random'):
        if mode=='random':
            start_id=random.choice(np.arange(self.natoms).astype(int))
        else:
            start_id=0
        graph=nx.Graph()
        bonds=[]
        for i in range(self.natoms):
            for j in range(i+1,self.natoms):
                if self.adjs[i,j]!=0:
                    bonds.append((i,j))
        graph.add_edges_from(bonds)
        atom_order=bfs_seq(graph,start_id)
        self.roll_back_order=np.argsort(atom_order)
        self.atoms=self.atoms[np.ix_(atom_order)]
        self.chiraltags=self.chiraltags[np.ix_(atom_order)]
        self.coords=self.coords[np.ix_(atom_order)]
        self.adjs=self.adjs[np.ix_(atom_order,atom_order)]
        return

    def Generate_Zmats(self):
        adjs_onek=Adjs_to_Onek(self.adjs)
        zmats=np_adjs_to_zmat(adjs_onek)[:,:4]
        return zmats
    
    def Standardrize(self,permindex='fix',RemoveHs=True):
        if RemoveHs:
            self.RemoveHs()
        self.PermIndex(mode=permindex)
        self.zmats=self.Generate_Zmats()
        return 

    def Get_3D_Graph_Tensor(self,max_atoms= None,with_zmat=False):
        if max_atoms:

            adjs=torch.zeros((max_atoms,max_atoms)).long()
            zmats=torch.zeros((max_atoms,4)).long()
            coords=torch.zeros((max_atoms,3))
            masks=torch.zeros(max_atoms).bool()
            masks[:self.natoms]=True

            if not GP.if_chiral:
                atom_idx=torch.zeros(max_atoms).long()
                atom_chiraltags=torch.zeros(max_atoms).long()
                atom_idx_=Atoms_to_Idx(self.atoms,GP.atom_types)
                atom_idx[:self.natoms]=torch.Tensor(atom_idx_).long()
            else:
                atom_idx=torch.zeros((max_atoms,len(GP.atom_types)))
                atom_chiraltags=torch.zeros((max_atoms,len(GP.chiral_types)))
                atom_idx_=Atoms_to_Onek(self.atoms,GP.atom_types)
                #print (self.__dict__.keys())
                atom_idx[:self.natoms]=torch.Tensor(atom_idx_)

            zmats[:self.natoms]=torch.Tensor(self.zmats).long()
            adjs[:self.natoms,:self.natoms]=torch.Tensor(self.adjs).long()
            coords[:self.natoms]=torch.Tensor(self.coords)
            return atom_idx,atom_chiraltags,adjs,coords,zmats,masks
        else:
            #print (self.natoms)
            atom_idx_=Atoms_to_Idx(self.atoms,GP.atom_types)
            zmats=torch.zeros((self.natoms,4)).long()
            if with_zmat:
                zmats=torch.Tensor(self.zmats).long()
            return torch.Tensor(atom_idx_).long(),torch.Tensor(self.chiraltags),torch.Tensor(self.adjs).long(),torch.Tensor(self.coords),zmats,torch.ones(self.natoms).bool()    

    
    def Trans_to_Rdkitmol(self,rollback=True):
        molecule=Chem.RWMol()
        for j in range(self.natoms):
            new_atom=Chem.Atom(int(self.atoms[j]))
            molecule_idx=molecule.AddAtom(new_atom)
        adjs=copy.deepcopy(self.adjs)
        row,col=np.diag_indices_from(adjs)
        adjs[row,col]=0
        idx1,idx2=np.where(adjs!=0)
        for id1,id2 in zip(idx1,idx2):
            if id1<id2:
                molecule.AddBond(int(id1),int(id2),GP.bond_types[int(adjs[id1,id2])-1])
        mol=molecule.GetMol()
        Chem.SanitizeMol(mol)
        AllChem.Compute2DCoords(mol)
        mol=Change_mol_xyz(mol,self.coords)
        if rollback:
            mol_rollback=Chem.RenumberAtoms(mol,[int(i) for i in self.roll_back_order])
            return mol_rollback
        else:
            return mol
    
    def Update_Coords(self,coords):
        self.coords=coords



def Adjs_to_Onek(adjs,nchannels=5):
    #nchannels=np.max(adjs)-1
    adjs_onek=np.zeros((adjs.shape[0],adjs.shape[0],nchannels))
    idx1,idx2=np.where(adjs)
    channel_idx=adjs[idx1,idx2].astype(int)-1
    for id1,id2,cid in zip(idx1,idx2,channel_idx):
        adjs_onek[id1,id2,cid]=1
    return adjs_onek.astype(int)

def Atoms_to_Idx(atoms,possible_atom_types=[1,6,7,8,9,15,16,17,35,53]):
    atom_idx_=[possible_atom_types.index(int(a))+1 for a in atoms] 
    return atom_idx_

def Atoms_to_Onek(atoms,possible_atom_types=[1,6,7,8,9,15,16,17,35,53]):
    atoms_onek=np.zeros((len(atoms),len(possible_atom_types)))
    for i in range(len(atoms)):
        atoms_onek[i][possible_atom_types.index(atoms[i])]=1
    return atoms_onek.astype(int)

