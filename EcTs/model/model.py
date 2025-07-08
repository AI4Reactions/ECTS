import torch
from .consistency import * 
import pickle,os,tempfile, shutil, zipfile, time, math, tqdm 
from datetime import datetime 
from ..comparm import * 
from ..utils.utils_torch import *
from ..utils.utils_envir import *
from .Tsgen import *
from nequip.ase import NequIPCalculator
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from ..graphs.datasets import *
from tqdm import tqdm 
from torch import distributed as dist
from ..utils.utils_evaluation import D_MAE
from rdkit.Chem import AllChem 
from ..utils.utils_pymol import write_pymol_script,write_combine_pymol_script
from ..utils.utils_rmsd import cal_rmsd
from ..graphs.rppairs import RP_pair
from ..utils.utils_rmsd import pymatgen_rmsd
from ..utils.utils_evaluation import path_rmsds_pymatgen
from neuralneb import painn
from neuralneb.utils import MLCalculator
from ase.io import read 

def Create_logger_file(filepath):
    logger_file=open(filepath,'a')
    now=datetime.now()
    logger_file.write('='*40+now.strftime("%d/%m/%Y %H:%M:%S")+'='*40+'\n')
    logger_file.flush() 
    return logger_file

def Set_Dataloader(trainset,validset=None,epoch=0,device="cuda"):
    if device!="cpu":
        Train_Sampler=torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader=DataLoader(trainset,batch_size=GP.batchsize,shuffle=False,num_workers=GP.n_workers,sampler=Train_Sampler)
        Train_Sampler.set_epoch(epoch)
        print ('train_dataset sampler is done')
        if validset is not None:
            Valid_Sampler=torch.utils.data.distributed.DistributedSampler(validset)
            validloader=DataLoader(validset,batch_size=GP.batchsize,shuffle=False,num_workers=GP.n_workers,sampler=Valid_Sampler)
            Valid_Sampler.set_epoch(epoch)
            print ('valid_dataset sampler is done')
            return trainloader,validloader
        else:
            return trainloader,None
    else:
        trainloader=DataLoader(trainset,batch_size=GP.batchsize,shuffle=True,num_workers=GP.n_workers)
        if validset is not None:
            validloader=DataLoader(validset,batch_size=GP.batchsize,shuffle=False,num_workers=GP.n_workers)
            return trainloader,validloader
        else:
            return trainloader,None

def masked_mean(P,pos_mask):
    denom = torch.sum(pos_mask, dim=1, keepdim=True)
    denom[denom==0] = 1.
    P_mean = torch.sum(P*pos_mask[:,:,None], dim=1, keepdim=True)/denom[:,:,None]
    return P_mean 

class EcTs_Model:
    def __init__(self,local_rank=None,**kwargs):
        epochs=kwargs.get('start')
        self.local_rank=local_rank
        self.device=GP.device

        self.mode="train"
        self.modelname='EcTs_Model'
        self.online_model=None
        self.ema_model=None
        self.path_online_model=None
        self.path_ema_model=None
        self.energy_online_model=None
        self.energy_ema_model=None
        self.confidence_model=None
            
        self.optim=None
        self.lr_scheduler=None

        if not os.path.exists(f'./{self.modelname}/model'):
            os.system(f'mkdir -p ./{self.modelname}/model')
        print (self.device)
        self.Load()
        if self.device!='cpu':
            self.Set_GPU_environ()
        

        device=next(self.online_model.parameters()).device
        
        if os.path.exists(GP.load_energy_calc_path):
            if GP.calc_type=="Nequip":
                self.calc = NequIPCalculator.from_deployed_model(model_path=GP.load_energy_calc_path,device=device)
                #pass
            else:
                statedict = torch.load(GP.load_energy_calc_path,map_location="cpu")
                painn_model = painn.PaiNN(3, 256, 5)
                painn_model.load_state_dict(statedict)
                painn_model.eval()
                self.calc = MLCalculator(painn_model)
                print (self.calc)
        else:
            self.calc = None  

        if GP.with_path_model:
            self.pathgen=Path_Sampler(self.online_model,self.path_online_model,self.energy_online_model,calc=self.calc, n_mid_states=GP.n_mid_states,sigma_data=GP.sigma_data,sigma_min=GP.sigma_min)
            print ("+"*80)
            print (self.calc)
        self.logger=Create_logger_file(f'./{self.modelname}/Training.log')
        self.batchsize=GP.batchsize
        return
    
    def __build_model(self):
        if self.device!='cpu':
            Set_gpu_envir(self.local_rank)
        self.online_model = TsGen()
        
        if GP.with_energy_model:
            self.energy_online_model=TseGen()

        if GP.with_path_model:
            self.path_online_model=TsGen()

        if GP.with_ema_model:
            self.ema_model = TsGen()
            if GP.with_energy_model:
                self.energy_ema_model=TseGen()
            if GP.with_path_model:
                self.path_ema_model=TsGen()

        self.consistency_training=ConsistencyTraining(
            sigma_min=GP.sigma_min,
            sigma_max=GP.sigma_max,
            sigma_data=GP.sigma_data,
            rho=GP.rho,
            initial_timesteps=GP.initial_timesteps,
            final_timesteps=GP.final_timesteps,
            with_energy=GP.with_energy_model,
            )

        self.consistency_sampling_and_editing = ConsistencySamplingAndEditing(
                        sigma_min = GP.sigma_min, # minimum std of noise
                        sigma_data = GP.sigma_data, # std of the data
                        with_energy=GP.with_energy_model, # whether to use energy model
                        )
        
    def Set_GPU_environ(self):
        Set_model_rank(self.online_model,self.local_rank)
        if GP.with_energy_model:
            Set_model_rank(self.energy_online_model,self.local_rank)
        if GP.with_path_model:
            Set_model_rank(self.path_online_model,self.local_rank)

        if GP.with_ema_model:
            Set_model_rank(self.ema_model,self.local_rank)
            if GP.with_energy_model:
                Set_model_rank(self.energy_ema_model,self.local_rank)
            if GP.with_path_model:
                Set_model_rank(self.path_ema_model,self.local_rank)

        return 
    
    def Save(self):
        self.optim=None,
        self.lr_scheduler=None,
        return
    
    def Load(self):
        self.__build_model()

        if GP.load_online_model_path is not None:
            modelcpkt=torch.load(GP.load_online_model_path,map_location="cpu")
            self.online_model.load_state_dict(modelcpkt["state_dict"],strict=False)
            print ('load online model successful')

        if GP.load_energy_online_model_path is not None and GP.with_energy_model:
            modelcpkt=torch.load(GP.load_energy_online_model_path,map_location="cpu")
            self.energy_online_model.load_state_dict(modelcpkt["state_dict"],strict=False)
            print ('load energy online model successful')           

        if GP.load_path_online_model_path is not None and GP.with_path_model:
            modelcpkt=torch.load(GP.load_path_online_model_path,map_location="cpu")
            self.path_online_model.load_state_dict(modelcpkt["state_dict"],strict=False)
            print ("load path online model successful")      

        if GP.with_ema_model:    
            
            if GP.load_ema_model_path is not None:
                modelcpkt=torch.load(GP.load_ema_model_path,map_location="cpu")
                self.ema_model.load_state_dict(modelcpkt["state_dict"],strict=False)
                print ('load ema model successful')

            if GP.load_energy_ema_model_path is not None and GP.with_energy_model:
                modelcpkt=torch.load(GP.load_energy_ema_model_path,map_location="cpu")
                self.energy_ema_model.load_state_dict(modelcpkt["state_dict"],strict=False)
                print ('load energy ema model successful')

            if GP.load_path_ema_model_path is not None and GP.with_path_model:
                modelcpkt=torch.load(GP.load_path_ema_model_path,map_location="cpu")
                self.path_ema_model.load_state_dict(modelcpkt["state_dict"],strict=False)
                print ("load path ema model successful") 
            
        return 

    def __to_device(self,tensor):
        if self.device!='cpu':
            return tensor.cuda(self.local_rank)
        else:
            return tensor

    def Dist_Loss(self,pred_coords,target_coords,gmasks):

        pred_dismat=torch.cdist(pred_coords,pred_coords,compute_mode='donot_use_mm_for_euclid_dist')
        target_dismat=torch.cdist(target_coords,target_coords,compute_mode='donot_use_mm_for_euclid_dist')
        gmasks_2D=gmasks.unsqueeze(-1)*gmasks.unsqueeze(-1).permute(0,2,1)

        loss_dismat=F.mse_loss(pred_dismat[gmasks_2D],target_dismat[gmasks_2D])
        return loss_dismat

    def Get_RP_Data_to_Device(self,Datas):
        rfeats,pfeats,radjs,padjs,rcoords,pcoords,tscoords,masks=Datas["RFeats"],Datas["PFeats"],Datas["RAdjs"],Datas["PAdjs"],Datas["RCoords"],Datas["PCoords"],Datas["TsCoords"],Datas["Masks"]
        redges,pedges=Datas["REdges"],Datas["PEdges"]
        renergy,penergy,tsenergy=Datas["REnergies"],Datas["PEnergies"],Datas["TsEnergies"]
        rforces,pforces,tsforces=Datas["RForces"],Datas["PForces"],Datas["TsForces"]

        rfeats=self.__to_device(rfeats)
        pfeats=self.__to_device(pfeats)
        radjs=self.__to_device(radjs)
        padjs=self.__to_device(padjs)
        redges=self.__to_device(redges)
        pedges=self.__to_device(pedges)

        rcoords=self.__to_device(rcoords)
        pcoords=self.__to_device(pcoords)
        tscoords=self.__to_device(tscoords)
        masks=self.__to_device(masks)

        renergy=self.__to_device(renergy)
        penergy=self.__to_device(penergy)
        tsenergy=self.__to_device(tsenergy)
        rforces=self.__to_device(rforces)
        pforces=self.__to_device(pforces)
        tsforces=self.__to_device(tsforces)

        return rfeats,pfeats,radjs,padjs,redges,pedges,rcoords,pcoords,tscoords,masks,\
            renergy,penergy,tsenergy,rforces,pforces,tsforces

    def Train_Step(self,Datas,step_id,mode='train',target='all'):
        if mode=='train':
            self.online_model.train()
            self.ema_model.train()
            if GP.with_energy_model:
                self.energy_online_model.train()
                self.energy_ema_model.train()
        else:
            self.online_model.eval()
            self.ema_model.eval()
            if GP.with_energy_model:
                self.energy_online_model.eval()
                self.energy_ema_model.eval()

        rfeats,pfeats,radjs,padjs,redges,pedges,rcoords,pcoords,tscoords,masks,\
            renergy,penergy,tsenergy,rforces,pforces,tsforces=self.Get_RP_Data_to_Device(Datas)

        self.optim.zero_grad()

        predicted_coords,predicted_energy,predicted_forces,target_coords,target_energy,target_forces=self.consistency_training(self.online_model,
                                                    self.ema_model,self.energy_online_model,self.energy_ema_model,rfeats,pfeats,radjs,padjs,redges,pedges,rcoords,pcoords,tscoords,masks,step_id,GP.final_timesteps)
        predicted_coords=predicted_coords-masked_mean(predicted_coords,masks)
        target_coords=target_coords-masked_mean(target_coords,masks)
        ref_coords=tscoords-masked_mean(tscoords,masks)

        loss_xyz_to_next=F.mse_loss(predicted_coords[masks],target_coords[masks])
        loss_xyz_to_ref=F.mse_loss(predicted_coords[masks],ref_coords[masks])
        loss_dismat_to_next=self.Dist_Loss(predicted_coords,target_coords,masks)
        loss_dismat_to_ref=self.Dist_Loss(predicted_coords,ref_coords,masks)

        if GP.with_energy_model:
            loss_energy_to_next=F.mse_loss(predicted_energy,target_energy)
            loss_forces_to_next=F.mse_loss(predicted_forces[masks],target_forces[masks])

            if GP.predict_energy_barrier:
                loss_current_energy_to_ref=F.mse_loss(predicted_energy,tsenergy.squeeze(-1)-renergy.squeeze(-1))
                loss_next_energy_to_ref=F.mse_loss(target_energy,tsenergy.squeeze(-1)-renergy.squeeze(-1))
            else:
                loss_current_energy_to_ref=F.mse_loss(predicted_energy-renergy.squeeze(-1),tsenergy.squeeze(-1)-renergy.squeeze(-1))
                loss_next_energy_to_ref=F.mse_loss(target_energy-renergy.squeeze(-1),tsenergy.squeeze(-1)-renergy.squeeze(-1))
            loss_current_forces_to_ref=F.mse_loss(predicted_forces[masks],tsforces[masks])

        self.lr=self.optim.state_dict()['param_groups'][0]['lr']
        
        lstr_next=f'Next Xyz: {loss_xyz_to_next.item():.3F} , D: {loss_dismat_to_next.item():.3F}, '
        lstr_ref=f'Current Xyz: {loss_xyz_to_ref.item():.3F} , D: {loss_dismat_to_ref.item():.3F},'
        if GP.with_energy_model:
            lstr_next+=f'E: {loss_energy_to_next.item():.3F}, '
            lstr_ref+=f' Curr E: {loss_current_energy_to_ref.item():.3F}, Next E: {loss_next_energy_to_ref.item():.3F}'
        
        self.lr=self.optim.state_dict()['param_groups'][0]['lr']

        loss_to_next=loss_xyz_to_next+loss_dismat_to_next

        loss_to_ref=loss_dismat_to_ref

        loss=loss_to_next+loss_to_ref*0.5
        if GP.with_energy_model:
            loss+=(loss_energy_to_next+loss_forces_to_next)*0.5+2*(loss_current_energy_to_ref+loss_current_forces_to_ref)
        
        lstr=lstr_next+' ; '+lstr_ref 
        
        if mode=='train':
            loss.backward()
            self.optim.step()
            num_timesteps=timesteps_schedule(step_id,GP.final_timesteps,initial_timesteps=GP.initial_timesteps,final_timesteps=GP.final_timesteps)
            
            ema_decay_rate = ema_decay_rate_schedule(
                                num_timesteps,
                                initial_ema_decay_rate=0.95,
                                initial_timesteps=2,
                            )
            
            if target=='all':
                update_ema_model(self.ema_model,self.online_model,ema_decay_rate)
                if GP.with_energy_model:
                    update_ema_model(self.energy_ema_model,self.energy_online_model,ema_decay_rate)
            elif target=='energy':
                update_ema_model(self.energy_ema_model,self.energy_online_model,ema_decay_rate)
            else:
                update_ema_model(self.ema_model,self.online_model,ema_decay_rate)

        return loss.item(),lstr
    
    def Fit(self,Train_RPFiles,Test_RPFiles,Epochs=100,miniepochs=10,target='all'):
        #print ('Here')
        assert self.device!='cpu', "CPU is not supported for training"
        if target=='all':
            params=[{"params":self.online_model.parameters()},]
            if GP.with_energy_model:
                params.append({"params":self.energy_online_model.parameters()})
            self.optim=Adam(params, lr = GP.init_lr, betas=(0.5,0.999))
        elif target=='energy':
            self.optim=Adam(self.energy_online_model.parameters(), lr = GP.init_lr, betas=(0.5,0.999))
        else:
            self.optim=Adam(self.online_model.parameters(), lr = GP.init_lr, betas=(0.5,0.999))

        self.lr_scheduler= ReduceLROnPlateau(
                self.optim, mode='min',
                factor=0.9, patience=GP.lr_patience,
                verbose=True, threshold=0.0001, threshold_mode='rel',
                cooldown=GP.lr_patience,
                min_lr=1e-08, eps=1e-08)
        
        train_rpdatas=[]

        for Fname in Train_RPFiles:
            with open(Fname,'rb') as f:
                RPs=pickle.load(f)
                train_rpdatas+=RPs
        random.shuffle(train_rpdatas)

        test_rpdatas=[]
        for Fname in Test_RPFiles:
            with open(Fname,'rb') as f:
                RPs=pickle.load(f)
                test_rpdatas+=RPs

        npairs_per_miniepoch=math.ceil(len(train_rpdatas)/miniepochs)
        self.epochs=0
        for epoch in range(Epochs):
            for mini in range(miniepochs):
                random.shuffle(test_rpdatas)
                rpdatas=train_rpdatas[mini*npairs_per_miniepoch:(mini+1)*npairs_per_miniepoch]
                Train_Dataset=RP_Dataset(rpdatas,name='trainset')
                Test_Dataset=RP_Dataset(test_rpdatas[:1000],name='validset')
                trainloader,testloader=Set_Dataloader(Train_Dataset,Test_Dataset)
                trainbar=tqdm(enumerate(trainloader))
                testbar=tqdm(enumerate(testloader))

                for bid,Datas in trainbar:
                    train_batch_loss=0
                    for step in range(GP.final_timesteps):
                        step_loss,step_lstr=self.Train_Step(Datas,step_id=step,target=target)
                        if self.local_rank==0 or self.local_rank is None:
                            lstr=f'Training -- Epochs: {self.epochs} bid: {bid} step: {step} lr: {self.lr:.3E} '+step_lstr
                            print (lstr)
                            self.logger.write(lstr+'\n')
                            self.logger.flush()
                            train_batch_loss+=step_loss
                self.lr_scheduler.step(metrics=train_batch_loss)
         
                for tid,tDatas in testbar:
                    for step in range(GP.final_timesteps):    
                        step_loss,step_lstr=self.Train_Step(tDatas,step_id=step,mode='eval',target=target)
                        lstr=f'Test -- Epochs: {self.epochs} bid: {tid} step: {step} lr: {self.lr:.3E} '+step_lstr
                        if self.local_rank==0 or self.local_rank is None:
                            print (lstr)
                            self.logger.write(lstr+'\n')
                            self.logger.flush()
         
                if self.local_rank==0 or self.local_rank is None:
                    self.Save_params(label='perepoch')
         
                if self.local_rank is not None:
                    dist.barrier() 

            self.epochs+=1
            if self.epochs%2==0:
                if self.local_rank==0 or self.local_rank is None:
                    self.Save_params(label=f'{self.epochs}')
            if self.local_rank is not None:
                dist.barrier()        
        return 

    def Save_params(self,label='0'):
        savepath=f'{self.modelname}/model/online_model_{label}.cpk'
        savedict={'epochs':self.epochs,'lr':self.lr,'state_dict':self.online_model.state_dict()}
        torch.save(savedict,savepath)    

        if GP.with_energy_model:
            savepath=f'{self.modelname}/model/energy_online_model_{label}.cpk'
            savedict={'epochs':self.epochs,'lr':self.lr,'state_dict':self.energy_online_model.state_dict()}
            torch.save(savedict,savepath)    
        
        if GP.with_path_model:
            savepath=f'{self.modelname}/model/path_online_model_{label}.cpk'
            savedict={'epochs':self.epochs,'lr':self.lr,'state_dict':self.path_online_model.state_dict()}
            torch.save(savedict,savepath)    
        
        if GP.with_ema_model:
            savepath=f'{self.modelname}/model/ema_model_{label}.cpk'
            savedict={'epochs':self.epochs,'lr':self.lr,'state_dict':self.ema_model.state_dict()}
            torch.save(savedict,savepath)

            if GP.with_energy_model:
                savepath=f'{self.modelname}/model/energy_ema_model_{label}.cpk'
                savedict={'epochs':self.epochs,'lr':self.lr,'state_dict':self.energy_ema_model.state_dict()}
                torch.save(savedict,savepath)        
            
            if GP.with_path_model:
                savepath=f'{self.modelname}/model/path_ema_model_{label}.cpk'
                savedict={'epochs':self.epochs,'lr':self.lr,'state_dict':self.path_ema_model.state_dict()}
                torch.save(savedict,savepath)     
        return 
    
    def sample_ts_batch(self,Datas):
        rfeats=self.__to_device(Datas["RFeats"])
        pfeats=self.__to_device(Datas["PFeats"])
        radjs=self.__to_device(Datas["RAdjs"])
        padjs=self.__to_device(Datas["PAdjs"])
        redges=self.__to_device(Datas["REdges"])
        pedges=self.__to_device(Datas["PEdges"])

        rcoords=self.__to_device(Datas["RCoords"])
        pcoords=self.__to_device(Datas["PCoords"])
        masks=self.__to_device(Datas["Masks"])

        if self.device!='cpu':
            rfeats,pfeats,radjs,padjs,rcoords,pcoords,masks=rfeats.cuda(self.local_rank),pfeats.cuda(self.local_rank),\
            radjs.cuda(self.local_rank),padjs.cuda(self.local_rank),\
            rcoords.cuda(self.local_rank),pcoords.cuda(self.local_rank),\
            masks.cuda(self.local_rank)
            redges,pedges=redges.cuda(self.local_rank),pedges.cuda(self.local_rank)

        with torch.no_grad():
            print (pcoords.device)
            sigmas = karras_schedule(
                GP.final_timesteps, GP.sigma_min, GP.sigma_max, GP.rho, pcoords.device
            )
            
            sigmas= reversed(sigmas)[:-1]
            init_y= torch.randn(rcoords.shape).to(pcoords)
            samples, energies, forces, samples_list, energies_list, forces_list = self.consistency_sampling_and_editing(
                                    self.online_model,
                                    self.energy_online_model,
                                    rfeats=rfeats,pfeats=pfeats,
                                    radjs=radjs,padjs=padjs,
                                    redges=redges,pedges=pedges,
                                    rcoords=rcoords,pcoords=pcoords,
                                    y=init_y, # used to infer the shapes
                                    sigmas=sigmas, # sampling starts at the maximum std (T)
                                    masks=masks,
                                    clip_denoised=False, # whether to clamp values to [-1, 1] range
                                    verbose=True,
                                    with_energy=GP.with_energy_model,
                                )
            
        return samples,energies,forces,samples_list,energies_list,forces_list

    def write_xyz(self,atoms,coords,fpath,title):
        symboldict={6:'C',7:'N',8:'O',1:'H',9:'F',11:'Na',15:'P',16:'S',17:'Cl'}
        with open (fpath,'w')  as f:
            f.write(f'{len(atoms)}\n')
            f.write(f'{title}\n')
            for i in range(len(atoms)):
                 f.write(f'{symboldict[atoms[i]]}  {float(coords[i][0])} {float(coords[i][1])} {float(coords[i][2])}\n')
        return 
 
    def Eval_Ts(self,RP,ts_num_per_mol=10,savepath='./samples',with_process=False):
        os.system(f'mkdir -p {savepath}')

        Final_RPs=[RP]*ts_num_per_mol
        self.write_xyz(RP.ratoms,RP.rcoords,f'{savepath}/r.xyz',title=f'{RP.idx}_r')
        self.write_xyz(RP.ratoms,RP.pcoords,f'{savepath}/p.xyz',title=f'{RP.idx}_p')
        self.write_xyz(RP.ratoms,RP.tscoords-np.mean(RP.tscoords,axis=0,keepdims=True),f'{savepath}/ref.xyz',title=f'{RP.idx}_ref')

        write_pymol_script(fname=f'r.xyz',path=f'{savepath}',scriptname=f'r.pml',outputname=f'r')
        write_pymol_script(fname=f'p.xyz',path=f'{savepath}',scriptname=f'p.pml',outputname=f'p')
        write_pymol_script(fname=f'ref.xyz',path=f'{savepath}',scriptname=f'ref.pml',outputname=f'ref')

        dataset=RP_Dataset(Final_RPs,name='sample')
        sample_loader,_=Set_Dataloader(dataset)

        bar=tqdm(enumerate(sample_loader))

        total_samples=[]
        total_energies=[]
        total_forces=[]

        for bid,Datas in bar:
            samples,energies,forces,samples_list,energies_list,forces_list=self.sample_ts_batch(Datas)
            
            samples_list=torch.cat([s.unsqueeze(0) for s in samples_list],axis=0).clone().detach().cpu().permute(1,0,2,3)
            energies_list=torch.cat([s.unsqueeze(0) for s in energies_list],axis=0).clone().detach().cpu().permute(1,0)
            forces_list=torch.cat([s.unsqueeze(0) for s in forces_list],axis=0).clone().detach().cpu().permute(1,0,2,3)

            total_samples.append(samples)
            total_energies.append(energies) 
            total_forces.append(forces)
            
        total_samples=torch.concat(total_samples,axis=0).clone().detach().cpu().numpy()
        total_energies=torch.concat(total_energies,axis=0).clone().detach().cpu().numpy()
        total_forces=torch.concat(total_forces,axis=0).clone().detach().cpu().numpy()
        average_energy=np.mean(total_energies)
        
        if GP.predict_energy_barrier:
            energy_errors=np.abs(total_energies-(RP.tsenergy-RP.renergy))*23.06
            print (energy_errors)
            delta_energy=np.abs(average_energy-(RP.tsenergy-RP.renergy))*23.06
        else:
            energy_errors=np.abs(total_energies-RP.tsenergy)*23.06
            delta_energy=np.abs(average_energy-RP.tsenergy)*23.06
        best_energy=total_energies[np.argmin(energy_errors)]
        energy_errors_mean=np.mean(energy_errors)
        energy_errors_min=np.min(energy_errors)

        total_rmsd_list=[] 
        coords_list=[]
        total_nn_energies=[]
        for i in range(total_samples.shape[0]):
            if with_process:
                os.system(f'mkdir -p {savepath}/{i}')
            else:
                os.system(f'mkdir -p {savepath}')

            sampled_coords=total_samples[i][:RP.natoms]
            sampled_coords=sampled_coords-np.mean(sampled_coords,axis=0,keepdims=True)
            coords_list.append(sampled_coords)
            #print (rmsd1,rmsd2)
            self.write_xyz(RP.ratoms,sampled_coords,f'{savepath}/{self.local_rank}-{i}.xyz',title=f'{RP.idx}_Ts')
            asemol=read(f'{savepath}/{self.local_rank}-{i}.xyz')
            asemol.calc=self.calc
            energy=asemol.get_potential_energy()
            total_nn_energies.append(energy)
            write_pymol_script(fname=f'{self.local_rank}-{i}.xyz',path=f'{savepath}/',scriptname=f'{self.local_rank}-{i}.pml',outputname=f'{self.local_rank}-{i}.pse')

            rmsd=pymatgen_rmsd(f'{savepath}/ref.xyz',f'{savepath}/{self.local_rank}-{i}.xyz',same_order=True,ignore_chirality=True)

            total_rmsd_list.append(rmsd)

        total_nn_energies=np.array(total_nn_energies)
        energy_errors_nn=np.abs(total_nn_energies-RP.tsenergy)*23.06
        best_energy_nn=total_nn_energies[np.argmin(energy_errors_nn)]
        energy_errors_mean_nn=np.mean(energy_errors_nn)
        energy_errors_min_nn=np.min(energy_errors_nn)
        delta_energy_nn=np.abs(np.mean(total_nn_energies)-RP.tsenergy)*23.06

        average_coords=cal_aligned_average_coords(coords_list)     
        self.write_xyz(RP.ratoms,average_coords,f'{savepath}/{self.local_rank}-average-ts.xyz',title=f'{RP.idx}_Average_Ts') 
        average_coords_rmsd= pymatgen_rmsd(f'{savepath}/ref.xyz',f'{savepath}/{self.local_rank}-average-ts.xyz',same_order=True,ignore_chirality=True)

        rmsd_mean=np.mean(total_rmsd_list)
        rmsd_min=np.min(total_rmsd_list)
        
        write_combine_pymol_script([f'{i}.xyz' for i in ['r','p','ref']+[f"{self.local_rank}-"+str(i) for i in range(total_samples.shape[0])]],
                                   path=f'{savepath}',scriptname=f'combine.pml',outputname=f'combine')
        if self.device!='cpu':
            dist.barrier() 
        
        #print (total_rmsd_list)

        with open(f'{savepath}/{self.local_rank}-result.txt','w') as f:
            for i in range(total_samples.shape[0]):
                f.write(f'{RP.idx}, {self.local_rank}, {i}, {(total_energies[i]-RP.renergy)*23.06:.3f}, {(RP.tsenergy-RP.renergy)*23.06:.3f},  {energy_errors[i]*23.06}, {total_rmsd_list[i]} \n')
            f.write(f"{RP.idx} Min RMSD: {np.min(total_rmsd_list)}\n")
            f.write(f"{RP.idx} Mean RMSD: {np.mean(total_rmsd_list)}\n")
            f.write(f"{RP.idx} Min Energy Error: {energy_errors_min}\n")
            f.write(f"{RP.idx} Mean Energy Error: {energy_errors_mean}\n")

        Result_Dict={
                "RMSD_of_AVERAGE_TS":average_coords_rmsd,
                "MEAN_RMSD_of_TS":rmsd_mean,
                "MIN_RMSD_of_TS":rmsd_min,
            
                "MEAN_ENERGY_ERROR":energy_errors_mean,
                "MIN_ENERGY_ERROR":energy_errors_min,
                "ERROR_of_AVERAGE_ENERGY":delta_energy,
                "AVERAGE_TS_ENERGY":average_energy*23.06,
                "BEST_TS_ENERGY":best_energy*23.06,

                "MEAN_ENERGY_ERROR_NN":energy_errors_mean_nn,
                "MIN_ENERGY_ERROR_NN":energy_errors_min_nn,
                "ERROR_of_AVERAGE_ENERGY_NN":delta_energy_nn,
                "AVERAGE_TS_ENERGY_NN":np.mean(total_nn_energies)*23.06,
                "BEST_TS_ENERGY_NN":best_energy_nn*23.06,

                "REF_TS_ENERGY":RP.tsenergy*23.06,
                "ENERGY_of_REACTANT":RP.renergy*23.06,
                "ENERGY_of_PRODUCT":RP.penergy*23.06,
                "ENERGY_ERROR_RATE":(average_energy-RP.tsenergy)/(RP.tsenergy-RP.renergy),
            }
        return Result_Dict 
    
    def Sample_TS(self,RP,ts_num_per_mol=10,savepath='./samples',with_process=False):
        os.system(f'mkdir -p {savepath}')

        Final_RPs=[RP]*ts_num_per_mol
        self.write_xyz(RP.ratoms,RP.rcoords,f'{savepath}/r.xyz',title=f'{RP.idx}_r')
        self.write_xyz(RP.ratoms,RP.pcoords,f'{savepath}/p.xyz',title=f'{RP.idx}_p')

        dataset=RP_Dataset(Final_RPs,name='sample')
        
        sample_loader,_=Set_Dataloader(dataset,device=self.device)

        bar=tqdm(enumerate(sample_loader))

        total_samples=[]
        total_energies=[]
        total_forces=[]

        for bid,Datas in bar:
            samples,energies,forces,samples_list,energies_list,forces_list=self.sample_ts_batch(Datas)
            
            samples_list=torch.cat([s.unsqueeze(0) for s in samples_list],axis=0).clone().detach().cpu().permute(1,0,2,3)
            total_samples.append(samples)
            if GP.with_energy_model:
                energies_list=torch.cat([s.unsqueeze(0) for s in energies_list],axis=0).clone().detach().cpu().permute(1,0)
                forces_list=torch.cat([s.unsqueeze(0) for s in forces_list],axis=0).clone().detach().cpu().permute(1,0,2,3)

                total_energies.append(energies) 
                total_forces.append(forces)
            
        total_samples=torch.concat(total_samples,axis=0).clone().detach().cpu().numpy()
        if GP.with_energy_model:
            total_energies=torch.concat(total_energies,axis=0).clone().detach().cpu().numpy()
            total_forces=torch.concat(total_forces,axis=0).clone().detach().cpu().numpy()
            average_energy=np.mean(total_energies)

        coords_list=[]
        if GP.with_energy_model:
            total_nn_energies=[]

        for i in range(total_samples.shape[0]):
            os.system(f'mkdir -p {savepath}')
            sampled_coords=total_samples[i][:RP.natoms]
            sampled_coords=sampled_coords-np.mean(sampled_coords,axis=0,keepdims=True)
            coords_list.append(sampled_coords)
            self.write_xyz(RP.ratoms,sampled_coords,f'{savepath}/{self.local_rank}-{i}.xyz',title=f'{RP.idx}_Ts')
            if GP.with_energy_model:
                asemol=read(f'{savepath}/{self.local_rank}-{i}.xyz')
                asemol.calc=self.calc
                energy=asemol.get_potential_energy()
                total_nn_energies.append(energy)
            write_pymol_script(fname=f'{self.local_rank}-{i}.xyz',path=f'{savepath}/',scriptname=f'{self.local_rank}-{i}.pml',outputname=f'{self.local_rank}-{i}.pse')

        if GP.with_energy_model:
            total_nn_energies=np.array(total_nn_energies)

        average_coords=cal_aligned_average_coords(coords_list)     
        self.write_xyz(RP.ratoms,average_coords,f'{savepath}/{self.local_rank}-average-ts.xyz',title=f'{RP.idx}_Average_Ts') 
        if self.device!='cpu':
            dist.barrier() 

        with open(f'{savepath}/{self.local_rank}-result.txt','w') as f:
            
            f.write("Index, Rank, ID, TS Energy Predictions with f_e (kcal/mol)\n")

            for i in range(total_samples.shape[0]):
                if GP.with_energy_model:
                    f.write(f'{RP.idx}, {self.local_rank}, {i}, {(total_energies[i])*23.06:.3f}\n')
                else:
                    f.write(f'{RP.idx}, {self.local_rank}, {i}, 0 \n')

        return 

    def sample_path_batch(self,Datas,atoms,sample_path_only=True):
        rfeats=self.__to_device(Datas["RFeats"])
        pfeats=self.__to_device(Datas["PFeats"])
        radjs=self.__to_device(Datas["RAdjs"])
        padjs=self.__to_device(Datas["PAdjs"])
        redges=self.__to_device(Datas["REdges"])
        pedges=self.__to_device(Datas["PEdges"])
        
        rcoords=self.__to_device(Datas["RCoords"])
        pcoords=self.__to_device(Datas["PCoords"])
        masks=self.__to_device(Datas["Masks"])
        print (rfeats.shape)
        if self.device!='cpu':
            rfeats,pfeats,radjs,padjs,rcoords,pcoords,masks=rfeats.cuda(self.local_rank),pfeats.cuda(self.local_rank),\
            radjs.cuda(self.local_rank),padjs.cuda(self.local_rank),\
            rcoords.cuda(self.local_rank),pcoords.cuda(self.local_rank),\
            masks.cuda(self.local_rank)
            redges,pedges=redges.cuda(self.local_rank),pedges.cuda(self.local_rank)

        sigmas=[0.15,0.075]

        sigmas_=karras_schedule(
                GP.final_timesteps-2, 0.002, 0.05, GP.rho, pcoords.device
            )

        sigmas=sigmas+list(reversed(sigmas_)[1:])
        pathes,ts_states,ts_energies,nn_energies,nn_max_energies,ts_masks=self.pathgen.path_ode_gen(atoms,rfeats,pfeats,radjs,padjs,redges,pedges,rcoords,pcoords,sigmas,masks)
            
        return pathes, ts_states, ts_energies, nn_energies, nn_max_energies, ts_masks

    def Eval_Path(self,RP,path_num=2,savepath='./sampled_pathes',sample_path_only=True,ref_reaction_path_dir='./RefPathes'):
        os.system(f'mkdir -p {savepath}')
        Final_RPs=[RP]*path_num
        self.write_xyz(RP.ratoms,RP.rcoords,f'{savepath}/r.xyz',title=f'{RP.idx}_r')
        self.write_xyz(RP.ratoms,RP.pcoords,f'{savepath}/p.xyz',title=f'{RP.idx}_p')
        self.write_xyz(RP.ratoms,RP.tscoords-np.mean(RP.tscoords,axis=0,keepdims=True),f'{savepath}/ref.xyz',title=f'{RP.idx}_ref')

        write_pymol_script(fname=f'r.xyz',path=f'{savepath}',scriptname=f'r.pml',outputname=f'r')
        write_pymol_script(fname=f'p.xyz',path=f'{savepath}',scriptname=f'p.pml',outputname=f'p')
        write_pymol_script(fname=f'ref.xyz',path=f'{savepath}',scriptname=f'ref.pml',outputname=f'ref')

        dataset=RP_Dataset(Final_RPs,name='sample_path')
        sample_loader,_=Set_Dataloader(dataset)
        ratoms=RP.ratoms

        bar=tqdm(enumerate(sample_loader))
        total_pathes=[]
        total_ts_states=[]
        total_ts_energies=[]
        total_nn_energies=[]
        total_nn_max_energies=[]
        for bid,Datas in bar:
            pathes,ts_states,ts_energies,nn_energies,nn_max_energies,ts_masks=self.sample_path_batch(Datas,ratoms,sample_path_only=sample_path_only)
            total_pathes.append(pathes)
            total_ts_states.append(ts_states)
            total_ts_energies.append(ts_energies)
            total_nn_max_energies.append(nn_max_energies)
            total_nn_energies.append(nn_energies)

        total_pathes=torch.concat(total_pathes,axis=0).clone().detach().cpu().numpy()

        if self.device!='cpu':
            dist.barrier() 

        os.system(f'mkdir -p {savepath}')
        if not sample_path_only:
            total_ts_states=torch.concat(total_ts_states,axis=0).clone().detach().cpu().numpy()
            total_ts_energies=torch.concat(total_ts_energies,axis=0).clone().detach().cpu().numpy()
            total_nn_energies=torch.concat(total_nn_energies,axis=0).clone().detach().cpu().numpy()
            total_nn_max_energies=torch.concat(total_nn_max_energies,axis=0).clone().detach().cpu().numpy()

            average_energy=np.mean(total_ts_energies)
            print (average_energy)
            if GP.predict_energy_barrier:
                energy_errors=np.abs(total_ts_energies-(RP.tsenergy-RP.renergy))*23.06
                delta_energy=np.abs(average_energy-(RP.tsenergy-RP.renergy))*23.06
            else:
                energy_errors=np.abs(total_ts_energies-RP.tsenergy)*23.06
                delta_energy=np.abs(average_energy-RP.tsenergy)*23.06
            best_energy=total_ts_energies[np.argmin(energy_errors)]
            energy_errors_mean=np.mean(energy_errors)
            energy_errors_min=np.min(energy_errors)
            
            average_nn_energy=np.mean(total_nn_energies)
            print (average_nn_energy)
            energy_errors_nn=np.abs(total_nn_energies-RP.tsenergy)*23.06
            best_nn_energy=total_nn_energies[np.argmin(energy_errors_nn)]
            energy_errors_mean_nn=np.mean(energy_errors_nn)
            energy_errors_min_nn=np.min(energy_errors_nn)
            delta_energy_nn=np.abs(average_nn_energy-RP.tsenergy)*23.06

            average_nn_max_energy=np.mean(total_nn_max_energies)
            print (average_nn_max_energy)
            energy_errors_nn_max=np.abs(total_nn_max_energies-RP.tsenergy)*23.06
            best_nn_max_energy=total_nn_max_energies[np.argmin(energy_errors_nn_max)]
            energy_errors_mean_nn_max=np.mean(energy_errors_nn_max)
            energy_errors_min_nn_max=np.min(energy_errors_nn_max)
            delta_energy_nn_max=np.abs(average_nn_max_energy-RP.tsenergy)*23.06

            total_rmsd_list=[] 
            n_samples=len(total_ts_states)
            coords_list=[]

            for i in range(total_ts_states.shape[0]):
                sampled_coords=total_ts_states[i][:RP.natoms]
                sampled_coords=sampled_coords-np.mean(sampled_coords,axis=0,keepdims=True)
                coords_list.append(sampled_coords)

                self.write_xyz(RP.ratoms,sampled_coords,f'{savepath}/{self.local_rank}-{i}-ts.xyz',title=f'{RP.idx}_Ts')

                write_pymol_script(fname=f'{self.local_rank}-{i}-ts.xyz',path=f'{savepath}/',scriptname=f'{self.local_rank}-{i}-ts.pml',outputname=f'{self.local_rank}-{i}.pse')

                rmsd=pymatgen_rmsd(f'{savepath}/ref.xyz',f'{savepath}/{self.local_rank}-{i}-ts.xyz',same_order=True,ignore_chirality=True)

                total_rmsd_list.append(rmsd)

            average_coords=cal_aligned_average_coords(coords_list)     
            self.write_xyz(RP.ratoms,average_coords,f'{savepath}/{self.local_rank}-average-ts.xyz',title=f'{RP.idx}_Average_Ts') 
            average_coords_rmsd= pymatgen_rmsd(f'{savepath}/ref.xyz',f'{savepath}/{self.local_rank}-average-ts.xyz',same_order=True,ignore_chirality=True)

            rmsd_mean=np.mean(total_rmsd_list)
            rmsd_min=np.min(total_rmsd_list)

            write_combine_pymol_script([f'{i}.xyz' for i in ['r','p','ref']+[f"{self.local_rank}-"+str(i)+"-ts" for i in range(total_ts_states.shape[0])]],
                                   path=f'{savepath}',scriptname=f'combine.pml',outputname=f'combine')
            
            print ("RMSDs to ref:", total_rmsd_list)

        total_pathes=rearrange(total_pathes,'(b n) a c -> b n  a c',n=2*GP.n_mid_states+1)
        
        print (total_pathes.shape)
        total_path_rmsds=[]
        for i in range(total_pathes.shape[0]):
            os.system(f'mkdir -p {savepath}/{self.local_rank}-{i}')
            for j in range(total_pathes.shape[1]):
                sampled_coords=total_pathes[i][j][:RP.natoms]
                sampled_coords=sampled_coords-np.mean(sampled_coords,axis=0,keepdims=True)
                self.write_xyz(RP.ratoms,sampled_coords,f'{savepath}/{self.local_rank}-{i}/{j}.xyz',title=f'{RP.idx}_path_{j}')

            path_rmsds= path_rmsds_pymatgen(predicted_path_files=[f'{savepath}/{self.local_rank}-{i}/{j}.xyz' for j in range(total_pathes.shape[1])],
                                            ref_path_files=[f'{ref_reaction_path_dir}/{RP.idx}/{j}.xyz' for j in range(1,9) if os.path.exists(f'{ref_reaction_path_dir}/{RP.idx}/{j}.xyz')],)
            total_path_rmsds.append(path_rmsds)
        
        average_path_rmsds=np.mean(total_path_rmsds)
        min_path_rmsds=np.min(total_path_rmsds)

        if not sample_path_only:

            Result_Dict={
                "RMSD_of_AVERAGE_TS":average_coords_rmsd,
                "MEAN_RMSD_of_TS":rmsd_mean,
                "MIN_RMSD_of_TS":rmsd_min,
            
                "MEAN_ENERGY_ERROR":energy_errors_mean,
                "MIN_ENERGY_ERROR":energy_errors_min,
                "ERROR_of_AVERAGE_ENERGY":delta_energy,
                
                "AVERAGE_TS_ENERGY":average_energy*23.06,
                "BEST_TS_ENERGY":best_energy*23.06,

                "MEAN_ENERGY_ERROR_NN":energy_errors_mean_nn,
                "MIN_ENERGY_ERROR_NN":energy_errors_min_nn,
                "ERROR_of_AVERAGE_ENERGY_NN":delta_energy_nn,

                "AVERAGE_TS_ENERGY_NN":average_nn_energy*23.06,
                "BEST_TS_ENERGY_NN":best_nn_energy*23.06,

                "MEAN_ENERGY_ERROR_NN_MAX":energy_errors_mean_nn_max,
                "MIN_ENERGY_ERROR_NN_MAX":energy_errors_min_nn_max,
                "ERROR_of_AVERAGE_ENERGY_NN_MAX":delta_energy_nn_max,
                "AVERAGE_TS_ENERGY_NN_MAX":average_nn_max_energy*23.06,
                "BEST_TS_ENERGY_NN_MAX":best_nn_max_energy*23.06,

                "REF_TS_ENERGY":RP.tsenergy*23.06,
                "ENERGY_of_REACTANT":RP.renergy*23.06,
                "ENERGY_of_PRODUCT":RP.penergy*23.06,
                "ENERGY_ERROR_RATE":(average_energy-RP.tsenergy)/(RP.tsenergy-RP.renergy),
            }

        else:
            Result_Dict={}
            
        Result_Dict["MEAN_PATH_RMSD"]=average_path_rmsds
        Result_Dict["MIN_PATH_RMSD"]=min_path_rmsds

        return Result_Dict

    def Sample_Path(self,RP,path_num=2,savepath='./sampled_pathes',sample_path_only=False,ref_reaction_path_dir='./RefPathes'):
        os.system(f'mkdir -p {savepath}')
        Final_RPs=[RP]*path_num
        self.write_xyz(RP.ratoms,RP.rcoords,f'{savepath}/r.xyz',title=f'{RP.idx}_r')
        self.write_xyz(RP.ratoms,RP.pcoords,f'{savepath}/p.xyz',title=f'{RP.idx}_p')

        dataset=RP_Dataset(Final_RPs,name='sample_path')
        sample_loader,_=Set_Dataloader(dataset,device=self.device)
        ratoms=RP.ratoms

        bar=tqdm(enumerate(sample_loader))
        total_pathes=[]
        total_ts_states=[]
        total_ts_energies=[]
        total_nn_energies=[]
        total_nn_max_energies=[]
        
        for bid,Datas in bar:
            pathes,ts_states,ts_energies,nn_energies,nn_max_energies,ts_masks=self.sample_path_batch(Datas,ratoms,sample_path_only=sample_path_only)
            total_pathes.append(pathes)
            total_ts_states.append(ts_states)
            total_ts_energies.append(ts_energies)
            total_nn_max_energies.append(nn_max_energies)
            total_nn_energies.append(nn_energies)

        total_pathes=torch.concat(total_pathes,axis=0).clone().detach().cpu().numpy()
        if self.device!='cpu':
            dist.barrier() 

        os.system(f'mkdir -p {savepath}')
        if not sample_path_only:
            total_ts_states=torch.concat(total_ts_states,axis=0).clone().detach().cpu().numpy()
            total_ts_energies=torch.concat(total_ts_energies,axis=0).clone().detach().cpu().numpy()
            total_nn_energies=torch.concat(total_nn_energies,axis=0).clone().detach().cpu().numpy()
            total_nn_max_energies=torch.concat(total_nn_max_energies,axis=0).clone().detach().cpu().numpy()

            average_energy=np.mean(total_ts_energies)
            average_nn_energy=np.mean(total_nn_energies)

            average_nn_max_energy=np.mean(total_nn_max_energies)

            n_samples=len(total_ts_states)
            coords_list=[]

            for i in range(total_ts_states.shape[0]):
                sampled_coords=total_ts_states[i][:RP.natoms]
                sampled_coords=sampled_coords-np.mean(sampled_coords,axis=0,keepdims=True)
                coords_list.append(sampled_coords)

                self.write_xyz(RP.ratoms,sampled_coords,f'{savepath}/{self.local_rank}-{i}-ts.xyz',title=f'{RP.idx}_Ts')

                write_pymol_script(fname=f'{self.local_rank}-{i}-ts.xyz',path=f'{savepath}/',scriptname=f'{self.local_rank}-{i}-ts.pml',outputname=f'{self.local_rank}-{i}.pse')


            average_coords=cal_aligned_average_coords(coords_list)     
            self.write_xyz(RP.ratoms,average_coords,f'{savepath}/{self.local_rank}-average-ts.xyz',title=f'{RP.idx}_Average_Ts') 


        total_pathes=rearrange(total_pathes,'(b n) a c -> b n  a c',n=2*GP.n_mid_states+1)
        
        print (total_pathes.shape)
        total_path_rmsds=[]
        for i in range(total_pathes.shape[0]):
            os.system(f'mkdir -p {savepath}/{self.local_rank}-{i}')
            for j in range(total_pathes.shape[1]):
                sampled_coords=total_pathes[i][j][:RP.natoms]
                sampled_coords=sampled_coords-np.mean(sampled_coords,axis=0,keepdims=True)
                self.write_xyz(RP.ratoms,sampled_coords,f'{savepath}/{self.local_rank}-{i}/{j}.xyz',title=f'{RP.idx}_path_{j}')
        if not sample_path_only:
            with open(f'{savepath}/{self.local_rank}-result.txt','w') as f:
                f.write("Index, Rank, ID, TS Energy Predictions with f_e\n")
                for i in range(total_pathes.shape[0]):
                    f.write(f'{RP.idx}, {self.local_rank}, {i}, {total_ts_energies[i]*23.06:.3f}\n')
        return