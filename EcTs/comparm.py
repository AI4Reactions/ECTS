from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType
import json 

class GPARMAS:
    def __init__(self):
        self.atom_types=[1,6,7,8,9,15,16,17,35,53]
        self.bond_types=[Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC]
        self.max_atoms=50
        self.max_bonds=150
        self.max_angles=150
        self.max_torsions=150

        self.batchsize=50
        self.device='cuda'
        self.x_sca_hidden=512
        self.x_vec_hidden=256
        self.edge_sca_hidden=256
        self.edge_vec_hidden=128
        self.n_head=4

        self.dropout=0.1
        self.update_coor_clamp=None
        self.c_block=4
        self.consistency_training_steps=25
        self.sigma_min=0.002
        self.sigma_max=80.0
        self.coords_scale=10
        self.rho=7.0
        self.sigma_data=0.5
        self.initial_timesteps=2
        self.final_timesteps=25
        self.init_lr=0.000025
        self.lr_patience=100
        self.lr_cooldown=100
        self.n_workers=20
        self.refw_min=0.1
        self.refw_max=0.5
        self.lossw=[1,0.5,0.5,0.2,0.5,1,0.2]
        self.calc_type="Nequip" #"PaiNN"
        self.load_energy_calc_path=None
        self.load_online_model_path=None
        self.load_ema_model_path=None  
        self.load_energy_online_model_path=None
        self.load_energy_ema_model_path=None
        self.load_path_online_model_path=None
        self.load_path_ema_model_path=None
        self.load_confidence_model_path=None
        self.with_ema_model=True
        self.with_path_model=True
        self.with_energy_guide=True
        self.with_confidence_model=True
        self.n_mid_states=4
        self.predict_energy_barrier=False
        

def Loaddict2obj(dict,obj):
    objdict=obj.__dict__
    for i in dict.keys():
        if i not in objdict.keys():
            print ("%s not is not a standard setting option!"%i)
        objdict[i]=dict[i]
    obj.__dict__==objdict

def Update_PARAMS(obj,jsonfile):
    with open(jsonfile,'r') as f:
        jsondict=json.load(f)
        Loaddict2obj(jsondict,obj)
    return obj 

GP=GPARMAS()
