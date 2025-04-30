from EcTs.model import EcTs_Model
import argparse as arg
import random ,math,pickle,os
from tqdm import tqdm 
from EcTs.comparm import Update_PARAMS,GP
from EcTs.utils import xyz2mol,xyz2AC,read_xyz_file
from EcTs.graphs import RP_pair
import numpy as np 

parser=arg.ArgumentParser(description='EcTs: equivariant consistency model for transization state predictions')
parser.add_argument('-i','--input')
parser.add_argument('-r','--reactant',type=str)
parser.add_argument('-p','--product',type=str)
parser.add_argument('-n','--name',type=str)
parser.add_argument('-s','--steps',type=int,default=25)
args=parser.parse_args()
jsonfile=args.input
rxyzfile=args.reactant
pxyzfile=args.product
savename=args.name
diffsteps=args.steps

GP=Update_PARAMS(GP,jsonfile)
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
local_rank=int(os.environ["LOCAL_RANK"])
savepath=f'./samples/{savename}'
def create_rp_pair(rfile,pfile,savename):
    ratoms,rcharge,rxyz=read_xyz_file(rfile)
    patoms,pcharge,pxyz=read_xyz_file(pfile)
    rmol=xyz2mol(ratoms,rxyz,charge=0)[0]
    pmol=xyz2mol(patoms,pxyz,charge=0)[0]
    rp_pair=RP_pair(rmol=rmol,pmol=pmol,
                idx=savename)
    return rp_pair
rp=create_rp_pair(rxyzfile,pxyzfile,savename)
print (rp)

Model=EcTs_Model(modelname="EcTs_Model",local_rank=local_rank)
GP.final_timesteps=diffsteps+1
Model.Sample_TS(rp,ts_num_per_mol=40,
                savepath=f'{savepath}')
#
#Result_Dict=Model.Sample_Ts(test_rps[i],ts_num_per_mol=40,
#                                 savepath=f'{savepath}/{diffsteps}/{test_rps[i].idx}')
