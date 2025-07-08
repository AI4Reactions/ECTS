import math
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from typing import Iterator
from einops import rearrange
from torch import Tensor, nn
from ...comparm import GP
from ...utils import dist_deriv
from ...utils import NEB_step
from .consistency_models import  * 

class Path_Sampler():
    def __init__(self,ts_model,path_model,energy_model,calc,n_mid_states,sigma_min: float = 0.002, sigma_data: float = 0.5, alpha=0.002) -> None:
        self.ts_model=ts_model
        self.path_model=path_model
        self.energy_model=energy_model
        self.calc=calc
        self.alpha=alpha
        self.n_mid_states=n_mid_states
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        return 

    def pad_reaction_path(self,rfeats,pfeats,radjs,padjs,redges,pedges,rcoords,pcoords,masks):
        rcoords = rcoords.unsqueeze(1).repeat(1, 2*self.n_mid_states+1,1,1)
        pcoords = pcoords.unsqueeze(1).repeat(1,2*self.n_mid_states+1,1,1)
        rfeats = rfeats.unsqueeze(1).repeat(1,2*self.n_mid_states+1,1,1)
        pfeats = pfeats.unsqueeze(1).repeat(1,2*self.n_mid_states+1,1,1)
        radjs = radjs.unsqueeze(1).repeat(1,2*self.n_mid_states+1,1,1,1)
        padjs = padjs.unsqueeze(1).repeat(1,2*self.n_mid_states+1,1,1,1)
        redges = redges.unsqueeze(1).repeat(1,2*self.n_mid_states+1,1,1,1)
        pedges = pedges.unsqueeze(1).repeat(1,2*self.n_mid_states+1,1,1,1)
        masks = masks.unsqueeze(1).repeat(1,2*self.n_mid_states+1,1)
        return rfeats,pfeats,radjs,padjs,redges,pedges,rcoords,pcoords,masks

    def select_ts_input(self,rfeats,pfeats,radjs,padjs,redges,pedges,rcoords,pcoords,sigma,masks,ts_masks):
        rfeats_ts=rfeats[ts_masks]
        pfeats_ts=pfeats[ts_masks]
        radjs_ts=radjs[ts_masks]
        padjs_ts=padjs[ts_masks]
        redges_ts=redges[ts_masks]
        pedges_ts=pedges[ts_masks]
        rcoords_ts=rcoords[ts_masks]
        pcoords_ts=pcoords[ts_masks]
        masks_ts=masks[ts_masks]
        sigma_ts=sigma[ts_masks]
        
        return rfeats_ts,pfeats_ts,radjs_ts,padjs_ts,redges_ts,pedges_ts,rcoords_ts,pcoords_ts,sigma_ts,masks_ts
    
    def sequeeze_path(self,rfeats,pfeats,radjs,padjs,redges,pedges,rcoords,pcoords,masks):
        rcoords = rearrange(rcoords,"b n c d-> (b n) c d")
        pcoords = rearrange(pcoords, "b n c d-> (b n) c d")
        rfeats = rearrange(rfeats, "b n c d-> (b n) c d")
        pfeats = rearrange(pfeats, "b n c d-> (b n) c d")
        radjs = rearrange(radjs, "b n c d e-> (b n) c d e")
        padjs = rearrange(padjs, "b n c d e-> (b n) c d e")
        redges = rearrange(redges, "b n c d e-> (b n) c d e")
        pedges = rearrange(pedges, "b n c d e-> (b n) c d e")
        masks = rearrange(masks, "b n c-> (b n) c")
        return rfeats,pfeats,radjs,padjs,redges,pedges,rcoords,pcoords,masks
    
    def analyze_reaction_variables(self,radjs,padjs):
        radjs_=radjs.sum(dim=-1)
        padjs_=padjs.sum(dim=-1)
        adjs_diff=(radjs_-padjs_)
        up_triu=torch.triu(adjs_diff,diagonal=1)
        break_bonds=torch.stack(torch.where(up_triu>0),dim=0).permute(1,0)
        bind_bonds=torch.stack(torch.where(up_triu<0),dim=0).permute(1,0)
        return break_bonds,bind_bonds
    
    def reaction_coords(self,x,break_bonds,bind_bonds,mask=None,eps=1e-7):
        batchsize=x.shape[0]
        nbreaks=break_bonds.shape[0]/batchsize
        nbinds=bind_bonds.shape[0]/batchsize
        break_dist=cal_bond_(x,break_bonds,mask,eps)
        bind_dist=cal_bond_(x,bind_bonds,mask,eps)
        break_dist=break_dist.reshape(batchsize,-1)    
        bind_dist=bind_dist.reshape(batchsize,-1)
        reaction_coords=torch.cat((break_dist,bind_dist),dim=-1)
        return reaction_coords

    def argsort_reaction_points(self,r_reac_coords,p_reac_coords,reac_coords):
        dist_r=torch.norm(reac_coords-r_reac_coords,dim=-1,p=2)
        dist_p=torch.norm(reac_coords-p_reac_coords,dim=-1,p=2)
        rank=torch.argsort(dist_r-dist_p,dim=1)
        return rank 

    def rerank_reaction_points(self,x,break_bonds,bind_bonds,r_reac_vec,p_reac_vec):
        x_reac_vec=self.reaction_coords(x,break_bonds,bind_bonds)
        x_reac_vec=rearrange(x_reac_vec,"(b n) c -> b n c", n=self.n_mid_states*2+1)
        rank=self.argsort_reaction_points(r_reac_vec,p_reac_vec,x_reac_vec)
        bids=torch.arange(rank.shape[0]).view(-1,1).tile((1,rank.shape[1])).to(rank)
        rank=rank+bids*self.n_mid_states
        rank=rank.view(-1)
        x_reac_vec=rearrange(x_reac_vec,"b n c -> (b n) c")
        x=torch.index_select(x,dim=0,index=rank)
        x_reac_vec=torch.index_select(x_reac_vec,dim=0,index=rank)
        return x , x_reac_vec
    
    def inject_ts_into_path(self, x, x_reac_vec, ts_x, break_bonds, bind_bonds):
        ts_reac_vec = self.reaction_coords(ts_x, break_bonds, bind_bonds)
        ts_reac_vec = ts_reac_vec.unsqueeze(1)
        dist = torch.norm(x_reac_vec - ts_reac_vec, dim=-1, p=2)
        ts_masks=torch.zeros_like(dist)
        ts_masks.scatter_(1,dist.argmin(dim=1, keepdim=True),1)
        ts_masks=ts_masks.view(-1).bool()
        x[ts_masks] = ts_x
        return x, ts_masks
    
    

    def interpolate(
        self,
        atoms,
        rfeats: Tensor,pfeats: Tensor,
        radjs: Tensor,padjs: Tensor,
        redges:Tensor,pedges:Tensor,
        rcoords: Tensor,pcoords: Tensor,
        sigmas: Iterable[Union[Tensor, float]],
        masks,
        clip_denoised: bool = False,
        verbose: bool = False,
        sample_path_only: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        
        # Obtain latent samples from the initial samples
        rfeats,pfeats,radjs,padjs,redges,pedges,rcoords,pcoords,masks=self.pad_reaction_path(rfeats,pfeats,radjs,padjs,redges,pedges,rcoords,pcoords,masks)
        #rp_ratios=(torch.arctan(torch.linspace(-5,5,2*self.n_mid_states+1))/3.1415+0.5).reshape(-1,1)
        rp_ratios=torch.linspace(0,1,2*self.n_mid_states+1).reshape(-1,1)
        #print (rp_ratios)
        rp_ratios=rp_ratios.unsqueeze(0).repeat(rcoords.shape[0],1,1)
        rp_ratios = rearrange(rp_ratios, "b n c -> (b n) c 1").to(rcoords)

        rfeats,pfeats,radjs,padjs,redges,pedges,rcoords,pcoords,masks=self.sequeeze_path(rfeats,pfeats,radjs,padjs,redges,pedges,rcoords,pcoords,masks)
    
        rp_middles_ori = rcoords + rp_ratios * (pcoords - rcoords)
        rp_middles = rp_middles_ori+sigmas[0] * torch.randn_like(rp_middles_ori)
        self.init_neb_environ(atoms,rp_middles,rcoords,pcoords)
        # Denoise the interpolated latents
        return  self.interpolate_run(
            atoms,
            rfeats,pfeats,
            radjs,padjs,
            redges,pedges,
            rcoords,pcoords,
            rp_middles,
            sigmas,
            masks,
            start_from_y=True,
            add_initial_noise=False,
            clip_denoised=clip_denoised,
            verbose=verbose,
            sample_path_only=sample_path_only,
            **kwargs,
        )
    
    def init_neb_environ(self,atoms,rp_middles,r_coords,p_coords):
        from ase import Atoms
        from copy import deepcopy
        from ase.mep.neb import NEB,NEBOptimizer

        self.natoms=len(atoms)
        r_pos=r_coords.clone().detach().cpu().numpy()[0,:self.natoms]
        p_pos=p_coords.clone().detach().cpu().numpy()[0,:self.natoms]
        mid_pos=rp_middles.clone().detach().cpu().numpy()[:,:self.natoms]

        images=[Atoms(numbers=atoms,positions=r_pos)]+[Atoms(numbers=atoms,positions=mid_pos[i]) for i in range(self.n_mid_states*2+1)]+[Atoms(numbers=atoms,positions=p_pos)]
        for image in images[1:-1]:
            image.calc=deepcopy(self.calc)
        self.neb=NEB(images)
        self.opt=NEBOptimizer(self.neb,method='static',alpha=self.alpha)
        
        return 
    
    def NEB_step(self,coords):
        #try:
        if True:
            coords_=coords.clone().detach().cpu().numpy()[:,:self.natoms].reshape(-1,3)
            self.opt.neb.set_positions(coords_)
            forces_=self.opt.force_function(coords_.reshape(-1)).reshape(-1,self.natoms,3)
            energies_=self.opt.neb.energies
            forces=torch.zeros_like(coords)
            forces[:,:self.natoms]=torch.tensor(forces_).to(coords)
            energies=torch.tensor(energies_).to(coords)
            fmax=self.opt.get_residual()

        #except:
        #    forces=torch.zeros_like(coords).to(coords)
        #    energies=torch.ones(coords.shape[0]).to(coords)*10000
        #    fmax=10000000

        return forces,fmax,energies
    
    def interpolate_run(
        self,
        atoms,
        rfeats: Tensor,pfeats: Tensor,
        radjs: Tensor,padjs: Tensor,
        redges:Tensor,pedges:Tensor,
        rcoords: Tensor,pcoords: Tensor,
        y: Tensor,
        sigmas: Iterable[Union[Tensor, float]],
        masks: Tensor,
        transform_mask: Optional[Tensor] = None,
        transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        inverse_transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        start_from_y: bool = False,
        add_initial_noise: bool = True,
        clip_denoised: bool = False,
        verbose: bool = False,
        sample_path_only: bool = False,
        **kwargs: Any,):

        npaths=int(pcoords.shape[0]/(self.n_mid_states*2+1))

        break_bonds,bind_bonds=self.analyze_reaction_variables(radjs,padjs)
        r_reac_vec=self.reaction_coords(rcoords,break_bonds,bind_bonds,masks)
        p_reac_vec=self.reaction_coords(pcoords,break_bonds,bind_bonds,masks)

        r_reac_vec=rearrange(r_reac_vec,"(b n) c -> b n c", n=2*self.n_mid_states+1)
        p_reac_vec=rearrange(p_reac_vec,"(b n) c -> b n c", n=2*self.n_mid_states+1)

        with torch.no_grad():
            if transform_mask is None:
                transform_mask = torch.ones_like(y)
            x = y if start_from_y else torch.zeros_like(y)
            x_list=[]
            # Sample at the end of the schedule
            y = self.__mask_transform(x, y, transform_mask, transform_fn, inverse_transform_fn)

            #x = y + sigmas[0] * torch.randn_like(y) if add_initial_noise else y
            x = y #+ sigmas[0] * torch.randn_like(y) if add_initial_noise else y

            sigma = torch.full((x.shape[0],), sigmas[0], dtype=x.dtype, device=x.device)
            if not sample_path_only:
                ts_masks=torch.zeros((npaths,2*self.n_mid_states+1)).to(rcoords)
                ts_masks[:,self.n_mid_states]=1
                ts_masks=ts_masks.view(-1).bool()
                ts_y=y[ts_masks]
                ts_x=x[ts_masks]

                ts_transform_mask=transform_mask[ts_masks]
                rfeats_ts,pfeats_ts,radjs_ts,padjs_ts,redges_ts,pedges_ts,rcoords_ts,pcoords_ts,sigma_ts,masks_ts=self.select_ts_input(rfeats,pfeats,radjs,padjs,redges,pedges,rcoords,pcoords,sigma,masks,ts_masks)
            
                ts_x = self.ts_model(
                    rfeats_ts,pfeats_ts,radjs_ts,padjs_ts,rcoords_ts,pcoords_ts,ts_x,sigma_ts,masks_ts,self.sigma_data,self.sigma_min,
                )
                ts_e,_=self.energy_model(
                    rfeats_ts,pfeats_ts,redges_ts,pedges_ts,rcoords_ts,pcoords_ts,ts_x,sigma_ts,masks_ts
                )
            x=self.path_model(
                rfeats, pfeats, radjs, padjs, rcoords, pcoords, x, sigma, masks, self.sigma_data, self.sigma_min,
            )

            x = self.__mask_transform(x, y, transform_mask, transform_fn, inverse_transform_fn) 
            x,x_reac_vec = self.rerank_reaction_points(x,break_bonds,bind_bonds,r_reac_vec,p_reac_vec)
            x_reac_vec=rearrange(x_reac_vec,"(b n) c -> b n c", n=2*self.n_mid_states+1)

            if not sample_path_only:
                ts_x = self.__mask_transform(ts_x, ts_y, ts_transform_mask, transform_fn, inverse_transform_fn)
                break_bonds_ts,bind_bonds_ts=self.analyze_reaction_variables(radjs_ts,padjs_ts)
                x,ts_masks = self.inject_ts_into_path(x,x_reac_vec,ts_x,break_bonds_ts,bind_bonds_ts)


        if GP.with_energy_guide and self.calc is not None:
            x,e=self.NEB_guide(x,guide_steps_interval=1)
        
        x_list.append(x)
        
        pbar=tqdm(sigmas[1:-1],disable=(not verbose))
        for sid,sigma in tqdm(enumerate(pbar)):
            pbar.set_description(f"interpolating (σ={sigma:.4f})")
            with torch.no_grad():
                sigma=torch.full((x.shape[0],),sigma,dtype=x.dtype,device=x.device)

                x = x + pad_dims_like((sigma**2-self.sigma_min**2)**0.5,x)*torch.randn_like(x)
                if not sample_path_only:                
                    ts_y=y[ts_masks]
                    ts_x=x[ts_masks]
                    ts_transform_mask=transform_mask[ts_masks]
        
                    rfeats_ts,pfeats_ts,radjs_ts,padjs_ts,redges_ts,pedges_ts,rcoords_ts,pcoords_ts,sigma_ts,masks_ts=self.select_ts_input(rfeats,pfeats,radjs,padjs,redges,pedges,rcoords,pcoords,sigma,masks,ts_masks)
        
                    ts_x = self.ts_model(
                        rfeats_ts,pfeats_ts,radjs_ts,padjs_ts,rcoords_ts,pcoords_ts,ts_x,sigma_ts,masks_ts,self.sigma_data,self.sigma_min,
                    )

                    ts_e,_=self.energy_model(
                        rfeats_ts,pfeats_ts,redges_ts,pedges_ts,rcoords_ts,pcoords_ts,ts_x,sigma_ts,masks_ts
                    )
        
                x=self.path_model(
                    rfeats, pfeats, radjs, padjs, rcoords, pcoords, x, sigma, masks, self.sigma_data, self.sigma_min,
                )
        
                x = self.__mask_transform(x, y, transform_mask, transform_fn, inverse_transform_fn) 
                x,x_reac_vec = self.rerank_reaction_points(x,break_bonds,bind_bonds,r_reac_vec,p_reac_vec)
                x_reac_vec=rearrange(x_reac_vec,"(b n) c -> b n c", n=2*self.n_mid_states+1)
        
                if not sample_path_only: 
                    ts_x = self.__mask_transform(ts_x, ts_y, ts_transform_mask, transform_fn, inverse_transform_fn)
                    break_bonds_ts,bind_bonds_ts=self.analyze_reaction_variables(radjs_ts,padjs_ts)
 
                    x,ts_masks = self.inject_ts_into_path(x,x_reac_vec,ts_x,break_bonds_ts,bind_bonds_ts)

        
            if GP.with_energy_guide and self.calc is not None:
                x,e=self.NEB_guide(x,guide_steps_interval=1)
        
            x_list.append(x)

        if not sample_path_only:
            ts_states=x[ts_masks]
            nn_e=e[ts_masks]
            return x,ts_states,ts_e,nn_e,x_list
        else:
            return x,None,None,None,x_list

    def NEB_guide(self,x,guide_steps_interval=1):
        for guide_step in range(guide_steps_interval):
            neb_forces=[]
            neb_energies=[]

            x_=x.clone().detach()
            
            x_=rearrange(x_,"(b n) c d -> b n c d", n=2*self.n_mid_states+1)
            
            neb_forces=[]
            neb_energies=[] 

            for i in range(x_.shape[0]):
                neb_force,force_max,neb_energy=self.NEB_step(x_[i])
                neb_forces.append(neb_force)
                neb_energies.append(neb_energy[1:-1])
                #print (neb_force.shape,neb_energy.shape)
                rpstr=f'Neb {guide_step} th force calculation in 1th diffusion step of {i}th reaction path -- Force max: {force_max:.4f}'
                print (rpstr)

            with torch.no_grad():
                neb_forces=torch.stack(neb_forces,dim=0)
                neb_forces=rearrange(neb_forces,"b n c d -> (b n) c d")
                neb_energies=torch.stack(neb_energies,dim=0)
                neb_energies=rearrange(neb_energies,"b n -> (b n)")
                x = x+neb_forces*self.alpha
        return x,neb_energies
    
    def __mask_transform(
        self,
        x: Tensor,
        y: Tensor,
        transform_mask: Tensor,
        transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        inverse_transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
    ) -> Tensor:
        return inverse_transform_fn(transform_fn(y) * (1.0 - transform_mask) + x * transform_mask)
    
