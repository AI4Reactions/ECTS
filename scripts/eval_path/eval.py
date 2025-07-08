from EcTs.model import EcTs_Model
import argparse as arg
import random ,math,pickle,os
from tqdm import tqdm 
from EcTs.comparm import Update_PARAMS,GP

parser=arg.ArgumentParser(description='EcTs: equivariant consistency model for transization state predictions')
parser.add_argument('-i','--input')
parser.add_argument('-s','--steps',type=int,default=25)
args=parser.parse_args()

jsonfile=args.input
diffsteps=args.steps
GP=Update_PARAMS(GP,jsonfile)
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

local_rank=int(os.environ["LOCAL_RANK"])

with open('../../datasets/test.pkl','rb') as f:
    test_rps=pickle.load(f)

savepath='./sampled_ts'

if not os.path.exists(savepath):
    os.makedirs(savepath)

Model=EcTs_Model(modelname="EcTs_Model",local_rank=local_rank)
GP.final_timesteps=diffsteps+1

with open(f'{savepath}/Sample_results_{diffsteps}.txt','w') as f:
    f.write("RXNID,RMSD_of_AVERAGE_TS,MEAN_RMSD_of_TS,MIN_RMSD_of_TS,ERROR_of_AVERAGE_ENERGY,MEAN_ENERGY_ERROR,MIN_ENERGY_ERROR, AVERAGE_TS_ENERGY,BEST_TS_ENERGY,ERROR_of_AVEARGE_ENERGY_NN,MEAN_ENERGY_ERROR_NN,MIN_ENERGY_ERROR_NN,AVERAGE_TS_ENERGY_NN, BEST_TS_ENERGY_NN,REF_TS_ENERGY,ENERGY_of_REACTANT,ENERGY_of_PRODUCT,ENERGY_ERROR_RATE\n")
    for i in tqdm(range(len(test_rps))):
        Result_Dict=Model.Eval_Path(test_rps[i],path_num=1,
                                 savepath=f'{savepath}/{diffsteps}/{test_rps[i].idx}')
        f.write(f"{test_rps[i].idx},{Result_Dict['RMSD_of_AVERAGE_TS']:.3f},{Result_Dict['MEAN_RMSD_of_TS']:.3f},{Result_Dict['MIN_RMSD_of_TS']:.3f},{Result_Dict['ERROR_of_AVERAGE_ENERGY']:.3f},{Result_Dict['MEAN_ENERGY_ERROR']:.3f}, {Result_Dict['MIN_ENERGY_ERROR']:.3f},{Result_Dict['AVERAGE_TS_ENERGY']:.3f}, {Result_Dict['BEST_TS_ENERGY']},{Result_Dict['ERROR_of_AVERAGE_ENERGY_NN']:.3f},{Result_Dict['MEAN_ENERGY_ERROR_NN']:.3f},{Result_Dict['MIN_ENERGY_ERROR_NN']:.3f},{Result_Dict['AVERAGE_TS_ENERGY_NN']:.3f},{Result_Dict['BEST_TS_ENERGY_NN']:.3f},{Result_Dict['REF_TS_ENERGY']:.3f},{Result_Dict['ENERGY_of_REACTANT']:.3f},{Result_Dict['ENERGY_of_PRODUCT']:.3f},{Result_Dict['ENERGY_ERROR_RATE']:.3f}\n")
        f.flush()

