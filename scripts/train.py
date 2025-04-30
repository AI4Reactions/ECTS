from EcTs.model import EcTs_Model
import argparse as arg
import random ,math,pickle,os
from tqdm import tqdm 
from EcTs.comparm import Update_PARAMS,GP

parser=arg.ArgumentParser(description='EcTs: equivariant consistency model for transization state predictions')
parser.add_argument('-i','--input')
args=parser.parse_args()
jsonfile=args.input

GP=Update_PARAMS(GP,jsonfile)
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
local_rank=int(os.environ["LOCAL_RANK"])

Model=EcTs_Model(local_rank=local_rank)
Model.Fit(Train_RPFiles=['./Train.pkl'],Test_RPFiles=['./Test.pkl'],Epochs=100)
