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
import numpy as np 


def cal_bond_(x,bond_indexes,mask=None,eps=1e-7):
    batch_num=x.shape[0]
    max_atom_num=x.shape[1]
    #print (batch_num,max_atom_num)
    it=bond_indexes[:,0]*max_atom_num+bond_indexes[:,1]
    #print ('it:',it)
    jt=bond_indexes[:,0]*max_atom_num+bond_indexes[:,2]
    x=x.view(-1,3)
    x0=torch.index_select(x,dim=0,index=it).view(-1,3)
    x1=torch.index_select(x,dim=0,index=jt).view(-1,3)
    dist,J_dist=dist_deriv(x0,x1)
    return dist



def pad_dims_like(x: Tensor, other: Tensor) -> Tensor:
    """Pad dimensions of tensor `x` to match the shape of tensor `other`.

    Parameters
    ----------
    x : Tensor
        Tensor to be padded.
    other : Tensor
        Tensor whose shape will be used as reference for padding.

    Returns
    -------
    Tensor
        Padded tensor with the same shape as other.
    """
    ndim = other.ndim - x.ndim
    return x.view(*x.shape, *((1,) * ndim))


def _update_ema_weights(
    ema_weight_iter: Iterator[Tensor],
    online_weight_iter: Iterator[Tensor],
    ema_decay_rate: float,
) -> None:
    for ema_weight, online_weight in zip(ema_weight_iter, online_weight_iter):
        if ema_weight.data is None:
            ema_weight.data.copy_(online_weight.data)
        else:
            ema_weight.data.lerp_(online_weight.data, 1.0 - ema_decay_rate)


def update_ema_model(
    ema_model: nn.Module, online_model: nn.Module, ema_decay_rate: float
) -> nn.Module:
    """Updates weights of a moving average model with an online/source model.

    Parameters
    ----------
    ema_model : nn.Module
        Moving average model.
    online_model : nn.Module
        Online or source model.
    ema_decay_rate : float
        Parameter that controls by how much the moving average weights are changed.

    Returns
    -------
    nn.Module
        Updated moving average model.
    """
    # Update parameters
    _update_ema_weights(
        ema_model.parameters(), online_model.parameters(), ema_decay_rate
    )
    # Update buffers
    _update_ema_weights(ema_model.buffers(), online_model.buffers(), ema_decay_rate)

    return ema_model
    
def timesteps_schedule(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 2,
    final_timesteps: int = 150,
) -> int:
    """Implements the proposed timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.
    """
    num_timesteps = final_timesteps**2 - initial_timesteps**2
    num_timesteps = current_training_step * num_timesteps / total_training_steps
    num_timesteps = math.ceil(math.sqrt(num_timesteps + initial_timesteps**2) - 1)

    return num_timesteps + 1


def ema_decay_rate_schedule(
    num_timesteps: int, initial_ema_decay_rate: float = 0.95, initial_timesteps: int = 2
) -> float:
    """Implements the proposed EMA decay rate schedule.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps at the current point in training.
    initial_ema_decay_rate : float, default=0.95
        EMA rate at the start of training.
    initial_timesteps : int, default=2
        Timesteps at the start of training.

    Returns
    -------
    float
        EMA decay rate at the current point in training.
    """
    return math.exp(
        (initial_timesteps * math.log(initial_ema_decay_rate)) / num_timesteps
    )

def karras_schedule(
    num_timesteps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device = None,
) -> Tensor:
    """Implements the karras schedule that controls the standard deviation of
    noise added.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps at the current point in training.
    sigma_min : float, default=0.002
        Minimum standard deviation.
    sigma_max : float, default=80.0
        Maximum standard deviation
    rho : float, default=7.0
        Schedule hyper-parameter.
    device : torch.device, default=None
        Device to generate the schedule/sigmas/boundaries/ts on.
    Returns
    -------
    Tensor
        Generated schedule/sigmas/boundaries/ts.
    """
    rho_inv = 1.0 / rho
    # Clamp steps to 1 so that we don't get nans
    steps = torch.arange(num_timesteps, device=device) / max(num_timesteps - 1, 1)
    sigmas = sigma_min**rho_inv + steps * (
        sigma_max**rho_inv - sigma_min**rho_inv
    )
    sigmas = sigmas**rho

    return sigmas


def skip_scaling(
    sigma: Tensor, sigma_data: float = 0.5, sigma_min: float = 0.002
) -> Tensor:
    """Computes the scaling value for the residual connection.

    Parameters
    ----------
    sigma : Tensor
        Current standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise from the karras schedule.

    Returns
    -------
    Tensor
        Scaling value for the residual connection.
    """
    return sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)


def output_scaling(
    sigma: Tensor, sigma_data: float = 0.5, sigma_min: float = 0.002
) -> Tensor:
    """Computes the scaling value for the model's output.

    Parameters
    ----------
    sigma : Tensor
        Current standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise from the karras schedule.

    Returns
    -------
    Tensor
        Scaling value for the model's output.
    """
    return (sigma_data * (sigma - sigma_min)) / (sigma_data**2 + sigma**2) ** 0.5


class ConsistencyTraining:

    """
    Implements the Consistency Training algorithm proposed in the paper.
    Parameters
    ----------
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_max : float, default=80.0
        Maximum standard deviation of the noise.
    rho : float, default=7.0
        Schedule hyper-parameter.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    initial_timesteps : int, default=2
        Schedule timesteps at the start of training.
    final_timesteps : int, default=150
        Schedule timesteps at the end of training.
    initial_ema_decay_rate : float, default=0.95
        EMA rate at the start of training.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 0.5,
        initial_timesteps: int = 2,
        final_timesteps: int = 150,
        with_energy=False,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.initial_timesteps = initial_timesteps
        self.final_timesteps = final_timesteps
        self.with_energy = with_energy

    def __call__(
        self,
        online_model: nn.Module,
        ema_model: nn.Module,
        energy_online_model: nn.Module,
        energy_ema_model: nn.Module,
        rfeats: Tensor,
        pfeats: Tensor,
        radjs: Tensor,
        padjs: Tensor,
        redges:Tensor,
        pedges:Tensor,
        rcoords:Tensor,
        pcoords:Tensor,
        xyzs: Tensor,
        masks: Tensor,
        current_training_step: int,
        total_training_steps: int,
    ) -> Tuple[Tensor, Tensor]:
        """Runs one step of the consistency training algorithm.

        Parameters
        ----------
        online_model : nn.Module
            Model that is being trained.
        ema_model : nn.Module
            An EMA of the online model.
        x : Tensor
            Clean data.
        current_training_step : int
            Current step in the training loop.
        total_training_steps : int
            Total number of steps in the training loop.
        **kwargs : Any
            Additional keyword arguments to be passed to the models.

        Returns
        -------
        (Tensor, Tensor)
            The predicted and target values for computing the loss.
        """
        num_timesteps = timesteps_schedule(
            current_training_step,
            total_training_steps,
            self.initial_timesteps,
            self.final_timesteps,
        )

        sigmas = karras_schedule(
            num_timesteps, self.sigma_min, self.sigma_max, self.rho, xyzs.device
        )

        noise = torch.randn_like(xyzs)

        timesteps = torch.randint(0, num_timesteps - 1, (xyzs.shape[0],), device=xyzs.device)

        current_sigmas = sigmas[timesteps]
        next_sigmas = sigmas[timesteps + 1]

        next_xyzs = xyzs + pad_dims_like(next_sigmas, xyzs) * noise
        next_xyzs = online_model(
            rfeats,pfeats,
            radjs,padjs,
            rcoords,pcoords,
            next_xyzs,
            next_sigmas,
            masks,
            self.sigma_data,
            self.sigma_min,
            )
        
        if self.with_energy:
            next_xyzs_clone = next_xyzs.clone().detach()
            next_energies, next_forces = energy_online_model( 
                rfeats,pfeats,
                redges,pedges,
                rcoords,pcoords,
                next_xyzs_clone,
                next_sigmas,
                masks)
        else:
            next_energies,next_forces=None,None

        with torch.no_grad():
            current_xyzs = xyzs + pad_dims_like(current_sigmas, xyzs) * noise
            current_xyzs = ema_model(
                rfeats,pfeats,
                radjs,padjs,
                rcoords,pcoords,
                current_xyzs,
                current_sigmas,
                masks,
                self.sigma_data,
                self.sigma_min,
            )
            if self.with_energy:
                current_xyzs_clone = current_xyzs.clone().detach()
                current_energies, current_forces = energy_ema_model(
                    rfeats,pfeats,
                    redges,pedges,
                    rcoords,pcoords,
                    current_xyzs_clone,
                    current_sigmas,
                    masks,
                )
            else:
                current_energies,current_forces=None,None
        
        return ( next_xyzs, next_energies, next_forces, current_xyzs, current_energies, current_forces)

class ConsistencySamplingAndEditing:
    """Implements the Consistency Sampling and Zero-Shot Editing algorithms.

    Parameters
    ----------
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    """
    def __init__(self, sigma_min: float = 0.002, sigma_data: float = 0.5,with_energy=False) -> None:
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        self.with_energy=with_energy

    def __call__(
        self,
        model: nn.Module,
        energy_model: nn.Module,
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
        **kwargs: Any,
    ) -> Tensor:
        
        """Runs the sampling/zero-shot editing loop.

        With the default parameters the function performs consistency sampling.

        Parameters
        ----------
        model : nn.Module
            Model to sample from.
        y : Tensor
            Reference sample e.g: a masked image or noise.
        sigmas : Iterable[Union[Tensor, float]]
            Decreasing standard deviations of the noise.
        transform_mask : Tensor, default=None
            A mask of zeros and ones with ones indicating where to edit. By
            default the whole sample will be edited. This is useful for sampling.
        transform_fn : Callable[[Tensor], Tensor], default=lambda x: x
            An invertible linear transformation. Defaults to the identity function.
        inverse_transform_fn : Callable[[Tensor], Tensor], default=lambda x: x
            Inverse of the linear transformation. Defaults to the identity function.
        start_from_y : bool, default=False
            Whether to use y as an initial sample and add noise to it instead of starting
            from random gaussian noise. This is useful for tasks like style transfer.
        add_initial_noise : bool, default=True
            Whether to add noise at the start of the schedule. Useful for tasks like interpolation
            where noise will alerady be added in advance.
        clip_denoised : bool, default=False
            Whether to clip denoised values to [-1, 1] range.
        verbose : bool, default=False
            Whether to display the progress bar.
        **kwargs : Any
            Additional keyword arguments to be passed to the model.
        Returns
        -------
        Tensor
            Edited/sampled sample.
        """

        # Set mask to all ones which is useful for sampling and style transfer
        if transform_mask is None:
            transform_mask = torch.ones_like(y)
        x_list=[]
        e_list=[]
        f_list=[]
        # Use y as an initial sample which is useful for tasks like style transfer
        # and interpolation where we want to use content from the reference sample
        x = y if start_from_y else torch.zeros_like(y)

        # Sample at the end of the schedule
        y = self.__mask_transform(x, y, transform_mask, transform_fn, inverse_transform_fn)
        # For tasks like interpolation where noise will already be added in advance we
        # can skip the noising process
        x = y + sigmas[0] * torch.randn_like(y) if add_initial_noise else y
        #x_list.append(x)
        sigma = torch.full((x.shape[0],), sigmas[0], dtype=x.dtype, device=x.device)
        
        print (rfeats.shape,pfeats.shape,radjs.shape,padjs.shape,rcoords.shape,pcoords.shape,x.shape,sigma.shape,masks.shape)
        x = model(
            rfeats, pfeats, radjs, padjs, rcoords, pcoords, x, sigma, masks,self.sigma_data, self.sigma_min
        )
        if self.with_energy:
            e,f = energy_model(
                rfeats, pfeats, redges, pedges, rcoords, pcoords, x, sigma, masks
            )

        #if clip_denoised:
        #    x = x.clamp(min=-1.0, max=1.0)
        x = self.__mask_transform(x, y, transform_mask, transform_fn, inverse_transform_fn)
        x_list.append(x)
        if self.with_energy:
            e_list.append(e)
            f_list.append(f)

        # Progressively denoise the sample and skip the first step as it has already
        # been run
        #pbar = tqdm(sigmas[1:], disable=(not verbose))

        pbar = tqdm(sigmas, disable=(not verbose))
        for sigma in pbar:
            pbar.set_description(f"sampling (σ={sigma:.4f})")            
            sigma = torch.full((x.shape[0],), sigma, dtype=x.dtype, device=x.device)
            x = x + pad_dims_like(
                (sigma**2 - self.sigma_min**2) ** 0.5, x
            ) * torch.randn_like(x)
            x = model(
                rfeats, pfeats, radjs, padjs, rcoords, pcoords, x, sigma, masks, self.sigma_data, self.sigma_min
            )
            if self.with_energy:
                e,f = energy_model(
                    rfeats, pfeats, redges, pedges, rcoords, pcoords, x, sigma, masks
                )
            #if clip_denoised:
            #    x = x.clamp(min=-1.0, max=1.0)
            x = self.__mask_transform(x, y, transform_mask, transform_fn, inverse_transform_fn)
            x_list.append(x)
            if self.with_energy:
                e_list.append(e)
                f_list.append(f)
        
        if not self.with_energy:
            e,f=None,None

        return x,e,f,x_list,e_list,f_list

    def __mask_transform(
        self,
        x: Tensor,
        y: Tensor,
        transform_mask: Tensor,
        transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        inverse_transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
    ) -> Tensor:
        return inverse_transform_fn(transform_fn(y) * (1.0 - transform_mask) + x * transform_mask)
    
def analyze_reaction_variables(radjs,padjs):
    radjs_=radjs.sum(dim=-1)
    padjs_=padjs.sum(dim=-1)
    #print ('adjs in analyze_reaction_variables',radjs_.shape,padjs_.shape)
    adjs_diff=(radjs_-padjs_)
    up_triu=torch.triu(adjs_diff,diagonal=1)
    #print ('triu shape',up_triu.shape)
    break_bonds=torch.stack(torch.where(up_triu>0),dim=0).permute(1,0)
    #print (break_bonds.shape)
    #print (torch.where(up_triu>0))

    bind_bonds=torch.stack(torch.where(up_triu<0),dim=0).permute(1,0)
    #print (bind_bonds.shape)
    return break_bonds,bind_bonds
    
def reaction_coords(x,break_bonds,bind_bonds,mask=None,eps=1e-7):
    batchsize=x.shape[0]
    nbreaks=break_bonds.shape[0]/batchsize
    nbinds=bind_bonds.shape[0]/batchsize
    break_dist=cal_bond_(x,break_bonds,mask,eps)
    bind_dist=cal_bond_(x,bind_bonds,mask,eps)
    break_dist=break_dist.reshape(batchsize,-1)    
    bind_dist=bind_dist.reshape(batchsize,-1)
    reaction_coords=torch.cat((break_dist,bind_dist),dim=-1)
    return reaction_coords

def argsort_reaction_points(r_reac_coords,p_reac_coords,reac_coords):
    dist_r=torch.norm(reac_coords-r_reac_coords,dim=-1,p=2)
    dist_p=torch.norm(reac_coords-p_reac_coords,dim=-1,p=2)
    rank=torch.argsort(dist_r-dist_p,dim=1)
    return rank 
    
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
    
    def init_reaction_path_variables(self,atoms,rfeats,pfeats,radjs,padjs,redges,pedges,rcoords,pcoords,masks,sigmas):
        self.atoms=atoms
        self.natoms=len(atoms)
        self.npathes=rfeats.shape[0]
        # init inputs for path predictions
        self.rcoords = rcoords.unsqueeze(1).repeat(1, 2*self.n_mid_states+1,1,1)
        self.pcoords = pcoords.unsqueeze(1).repeat(1,2*self.n_mid_states+1,1,1)
        self.rfeats = rfeats.unsqueeze(1).repeat(1,2*self.n_mid_states+1,1,1)
        self.pfeats = pfeats.unsqueeze(1).repeat(1,2*self.n_mid_states+1,1,1)
        self.radjs = radjs.unsqueeze(1).repeat(1,2*self.n_mid_states+1,1,1,1)
        self.padjs = padjs.unsqueeze(1).repeat(1,2*self.n_mid_states+1,1,1,1)
        self.redges = redges.unsqueeze(1).repeat(1,2*self.n_mid_states+1,1,1,1)
        self.pedges = pedges.unsqueeze(1).repeat(1,2*self.n_mid_states+1,1,1,1)
        self.masks = masks.unsqueeze(1).repeat(1,2*self.n_mid_states+1,1)

        self.rcoords=rearrange(self.rcoords,"b n c d -> (b n) c d")
        self.pcoords=rearrange(self.pcoords,"b n c d -> (b n) c d")
        self.rfeats=rearrange(self.rfeats,"b n c d -> (b n) c d")
        self.pfeats=rearrange(self.pfeats,"b n c d -> (b n) c d")
        self.radjs=rearrange(self.radjs,"b n c d e -> (b n) c d e")
        self.padjs=rearrange(self.padjs,"b n c d e -> (b n) c d e")
        self.redges=rearrange(self.redges,"b n c d e -> (b n) c d e")
        self.pedges=rearrange(self.pedges,"b n c d e -> (b n) c d e")
        self.masks=rearrange(self.masks,"b n c -> (b n) c")

        # init inputs for ts predictions
        self.rcoords_ts=rcoords
        self.pcoords_ts=pcoords
        self.rfeats_ts=rfeats
        self.pfeats_ts=pfeats
        self.radjs_ts=radjs
        self.padjs_ts=padjs
        self.redges_ts=redges
        self.pedges_ts=pedges
        self.masks_ts=masks
        # init variables for neb ode guidance
        self.Fn=torch.zeros_like(self.rcoords).to(self.rcoords)
        self.Fp=torch.zeros_like(self.rcoords).to(self.rcoords)
        self.Rn=torch.zeros(self.npathes).to(self.rcoords)
        self.Rp=torch.zeros(self.npathes).to(self.rcoords)
        self.En=torch.zeros((self.npathes,self.n_mid_states*2+1)).to(self.rcoords)
        self.Ep=torch.zeros((self.npathes,self.n_mid_states*2+1)).to(self.rcoords)
        self.h=torch.zeros(self.npathes).to(self.rcoords)
        self.converge_flags=torch.zeros(self.npathes).to(self.rcoords)
        self.failed_flags=torch.zeros(self.npathes).to(self.rcoords)

        self.sigmas=sigmas
        self.x=self.interpolate_init_path(method='linear')
        # init ts masks to select x for ts predictions
        self.ts_masks=torch.zeros((self.npathes,2*self.n_mid_states+1)).to(self.rcoords).long()
        self.ts_masks[:,self.n_mid_states]=1
        self.ts_masks=self.ts_masks.view(-1).bool()
        self.ts_x=self.x[self.ts_masks]

        self.ts_e=torch.zeros(self.npathes).to(self.rcoords)
        # init reaction variables
        #print (self.radjs.shape,self.padjs.shape)
        self.bind_bonds,self.break_bonds=analyze_reaction_variables(self.radjs,self.padjs)


        self.r_reac_vec=reaction_coords(self.rcoords,self.break_bonds,self.bind_bonds,self.masks)
        self.p_reac_vec=reaction_coords(self.pcoords,self.break_bonds,self.bind_bonds,self.masks)
        #print ('break_bonds:',self.break_bonds.shape)
        
        self.bind_bonds_ts,self.break_bonds_ts=analyze_reaction_variables(self.radjs_ts,self.padjs_ts)
        #print ('break_bonds_ts:',self.break_bonds_ts.shape)
        return 
    
    def interpolate_init_path(self,method='linear'):
        if method=='linear':
            rp_ratios=torch.linspace(0,1,2*self.n_mid_states+1).reshape(-1,1)
            #print (rp_ratios)
            rp_ratios=rp_ratios.unsqueeze(0).repeat(self.npathes,1,1)
            rp_ratios = rearrange(rp_ratios, "b n c -> (b n) c 1").to(self.rcoords)
            rp_middles = self.rcoords + rp_ratios * (self.pcoords - self.rcoords)
        else:
            rp_middles=torch.zeros_like(self.rcoords)
        return rp_middles
    
    def inject_ts_into_path(self, x, ts_x):
        x_reac_vec=reaction_coords(x,self.break_bonds,self.bind_bonds)

        x_reac_vec=rearrange(x_reac_vec,"(b n) c -> b n c", n=self.n_mid_states*2+1)
        r_reac_vec=rearrange(self.r_reac_vec,"(b n) c -> b n c", n=self.n_mid_states*2+1)
        p_reac_vec=rearrange(self.p_reac_vec,"(b n) c -> b n c", n=self.n_mid_states*2+1)
        rank=argsort_reaction_points(r_reac_vec,p_reac_vec,x_reac_vec)

        bids=torch.arange(rank.shape[0]).view(-1,1).tile((1,rank.shape[1])).to(rank)
        rank=rank+bids*(2*self.n_mid_states+1)
        rank=rank.view(-1)

        x_reac_vec=rearrange(x_reac_vec,"b n c -> (b n) c")
        x=torch.index_select(x,dim=0,index=rank)
        x_reac_vec=torch.index_select(x_reac_vec,dim=0,index=rank)
        #print ('ts_x.shape,',ts_x.shape)
        ts_reac_vec = reaction_coords(ts_x, self.break_bonds_ts, self.bind_bonds_ts)
        ts_reac_vec = ts_reac_vec.unsqueeze(1)
        x_reac_vec = rearrange(x_reac_vec, "(b n) c -> b n c", n=self.n_mid_states*2+1)

        dist = torch.norm(x_reac_vec - ts_reac_vec, dim=-1, p=2)
        ts_masks=torch.zeros_like(dist)
        ts_masks.scatter_(1,dist.argmin(dim=1, keepdim=True),1)
        ts_masks=ts_masks.view(-1).bool()
        x[ts_masks] = ts_x
        return x, ts_x, ts_masks
    
    def diff_ts_path_step(self,x, ts_x, sigma, sigma_ts):

        ts_x = self.ts_model(
                    self.rfeats_ts,self.pfeats_ts,self.radjs_ts,self.padjs_ts,self.rcoords_ts,self.pcoords_ts,ts_x,sigma_ts,self.masks_ts,self.sigma_data,self.sigma_min,
                )

        ts_e,_=self.energy_model(
                    self.rfeats_ts,self.pfeats_ts,self.redges_ts,self.pedges_ts,self.rcoords_ts,self.pcoords_ts,ts_x,sigma_ts,self.masks_ts
                )

        x = self.path_model(
            self.rfeats, self.pfeats,
            self.radjs, self.padjs, 
            self.rcoords, self.pcoords,
            x, sigma, self.masks, self.sigma_data, self.sigma_min,
        )

        x,ts_x,ts_masks = self.inject_ts_into_path(x,ts_x)
        return x,ts_x,ts_e,ts_masks
    
    def path_static_gen(
        self,
        atoms,
        rfeats: Tensor,pfeats: Tensor,
        radjs: Tensor,padjs: Tensor,
        redges:Tensor,pedges:Tensor,
        rcoords: Tensor,pcoords: Tensor,
        sigmas: Iterable[Union[Tensor, float]],
        masks: Tensor,):

        npaths=int(pcoords.shape[0]/(self.n_mid_states*2+1))
        self.init_reaction_path_variables(atoms,rfeats,pfeats,radjs,padjs,redges,pedges,rcoords,pcoords,masks,sigmas)
        rp_middles=rearrange(self.x,"(b n) c d -> b n c d", n=2*self.n_mid_states+1)
        self.init_neb_environ(rp_middles[0],rcoords[0],pcoords[0])

        with torch.no_grad():
            sigma = torch.full((self.x.shape[0],), sigmas[0], dtype=x.dtype, device=x.device)
            sigma_ts=sigma[self.ts_masks]
            
            self.x=self.x + sigmas[0] * torch.randn_like(self.x)
            self.ts_x=self.x[self.ts_masks]

            self.x,self.ts_x,self.ts_e,self.ts_masks=self.diff_ts_path_step(self.x,self.ts_x,sigma,sigma_ts)
        if GP.with_energy_guide and self.calc is not None:
            self.x,self.nn_e=self.NEB_static_step(self.x)
        
        pbar=tqdm(sigmas[1:-1],disable=(not False))

        for sid,sigma in tqdm(enumerate(pbar)):
            pbar.set_description(f"interpolating (σ={sigma:.4f})")
            with torch.no_grad():
                sigma=torch.full((self.x.shape[0],),sigma,dtype=self.x.dtype,device=x.device)
                sigma_ts=sigma[self.ts_masks]

                self.x = self.x + pad_dims_like((sigma**2-self.sigma_min**2)**0.5,self.x)*torch.randn_like(self.x)
                self.ts_x=self.x[self.ts_masks]

                self.x,self.ts_x,self.ts_e,self.ts_masks=self.mix_diff_ts_path_step(self.x,self.ts_x,sigma,sigma_ts)
            if GP.with_energy_guide and self.calc is not None:
                self.x,self.nn_e=self.NEB_static_step(self.x)
                
        return self.x,self.ts_x,self.ts_e,self.nn_e,self.ts_masks
        
    def init_neb_environ(self,rp_middles,rcoords,pcoords):
        from ase import Atoms
        from copy import deepcopy
        from ase.mep.neb import NEB,NEBOptimizer

        r_pos=rcoords.clone().detach().cpu().numpy()[:self.natoms]
        p_pos=pcoords.clone().detach().cpu().numpy()[:self.natoms]
        mid_pos=rp_middles.clone().detach().cpu().numpy()[:,:self.natoms]
        images=[Atoms(numbers=self.atoms,positions=r_pos)]+[Atoms(numbers=self.atoms,positions=mid_pos[i]) for i in range(self.n_mid_states*2+1)]+[Atoms(numbers=self.atoms,positions=p_pos)]

        for image in images[1:-1]:
            image.calc=deepcopy(self.calc)
        self.neb=NEB(images)
        self.opt=NEBOptimizer(self.neb,method='static',alpha=self.alpha)
        return 
    
    
    def NEB_forces_energies(self,coords):
        #try:
        #if True:
            coords_=coords.clone().detach().cpu().numpy()[:,:self.natoms].reshape(-1,3)
            self.opt.neb.set_positions(coords_)
            forces_=self.opt.force_function(coords_.reshape(-1)).reshape(-1,self.natoms,3)
            energies_=self.opt.neb.energies[1:-1]
            forces=torch.zeros_like(coords)
            forces[:,:self.natoms]=torch.tensor(forces_).to(coords)
            energies=torch.tensor(energies_).to(coords)
            fmax=self.opt.get_residual()

        #except:
        #    forces=torch.zeros_like(coords).to(coords)
        #    energies=torch.ones(coords.shape[0]).to(coords)*10000
        #    fmax=10000000

            return forces,fmax,energies

    def NEB_forces_energies_batch(self,batch_coords):
        neb_forces=[]
        neb_energies=[]

        batch_coords_=batch_coords.clone().detach()
        
        batch_coords_=rearrange(batch_coords_,"(b n) c d -> b n c d", n=2*self.n_mid_states+1)

        neb_forces=[]
        neb_energies=[] 
        neb_force_max=[]

        for i in range(batch_coords_.shape[0]):
            #print ('batch_coords_.shape',batch_coords_[i].shape)
            neb_force,force_max,neb_energy=self.NEB_forces_energies(batch_coords_[i])
            neb_forces.append(neb_force)
            neb_force_max.append(force_max)
            neb_energies.append(neb_energy)
            print (neb_force.shape,neb_energy.shape)
            rpstr=f'Neb  force calculation in 1th diffusion step of {i}th reaction path -- Force max: {force_max:.4f}'
            print (rpstr)
        #print (neb_force_max)

        with torch.no_grad():
            neb_forces=torch.stack(neb_forces,dim=0)
            neb_forces=rearrange(neb_forces,"b n c d -> (b n) c d")
            neb_force_max=torch.Tensor(neb_force_max)
            neb_energies=torch.stack(neb_energies,dim=0)
            neb_energies=rearrange(neb_energies,"b n -> (b n)")
        return neb_forces,neb_force_max,neb_energies

    def NEB_static_step(self,x):
        neb_forces,neb_max_forces,neb_energies=self.NEB_forces_energies_batch(x)
        x = x+neb_forces*self.alpha
        return x,neb_energies

    def NEB_predict_force_energy(self,coords_):
            self.opt.neb.set_positions(coords_.reshape(-1,3))
            forces_=self.opt.force_function(coords_.reshape(-1))
            energies_=self.opt.neb.energies[1:-1]
            fmax=self.opt.get_residual()
            return forces_,fmax,energies_
    
    def NEB_ode_step(self,x,Fn,Rn,En,Fp,Rp,Ep,h,fmax=0.05,
                     rtol=0.1,C1=0.01,C2=2,hmin=1e-10,max_tol=1e3):
        #print ('x in ode',x.shape)
        #print ('Fn in ode',Fn.shape)
        #print ('En in ode',En.shape)
        accept_flag=False
        reject_times=0
        while not accept_flag:        
            x_new=x+Fn*h
            Fn_new,Rn_new,En_new=self.NEB_predict_force_energy(x_new)
            Fp_new,Rp_new,Ep_new=Fn_new,Rn_new,En_new
            e=0.5 * h * (Fp_new - Fp)
            
            err=np.linalg.norm(e, torch.inf)
            accept = ((Rp_new <= Rp * (1 - C1 * h)) or
                      ((Rp_new <= Rp * C2) and err <= rtol))
            
            #print ('accept:',accept)
            y=Fp-Fp_new
            #print ('En_new',En_new.shape)
            h_ls = h * (Fp @ y) / (y @ y + 1e-10)
            if np.isnan(h_ls) or h_ls < hmin:  # Rejects if increment is too small
                h_ls = np.inf

            h_err = h * 0.5 * np.sqrt(rtol / err)

            if accept or reject_times>5:
                x=x_new
                Fn=Fn_new
                Rn=Rn_new
                En=En_new
                Fp=Fp_new
                Rp=Rp_new
                Ep=Ep_new
                h = max(0.25 * h,
                        min(4 * h, h_err, h_ls))
                #if Rp_new <= fmax:
                #    converge_flag=True
                #if Rp_new > max_tol:
                #    failed_flag=True
                accept_flag=True
            else:
                print ('Reject',Rp_new,Rp,err,rtol)
                h = max(0.1 * h, min(0.25 * h, h_err, h_ls),0.0002)
                reject_times+=1
                print (f'Adjust h to {h:.4f}')

        return x,Fn,Rn,En,Fp,Rp,Ep,h
    
    def NEB_ode_guide(self,x,Fn,Rn,En,Fp,Rp,Ep,h,fmax=0.05,):
        for i in range(self.npathes):
            #print ('x before ode',x.shape)
            x_i=x[i*(self.n_mid_states*2+1):(i+1)*(self.n_mid_states*2+1)].clone().detach().cpu().numpy()[:,:self.natoms].reshape(-1)

            Fn_i=Fn[i*(self.n_mid_states*2+1):(i+1)*(self.n_mid_states*2+1)].clone().detach().cpu().numpy()[:,:self.natoms].reshape(-1)
            Rn_i=Rn[i].clone().detach().cpu().numpy()
            En_i=En[i*(self.n_mid_states*2+1):(i+1)*(self.n_mid_states*2+1)].clone().detach().cpu().numpy()
            
            Fp_i=Fp[i*(self.n_mid_states*2+1):(i+1)*(self.n_mid_states*2+1)].clone().detach().cpu().numpy()[:,:self.natoms].reshape(-1)
            Rp_i=Rp[i].clone().detach().cpu().numpy()
            Ep_i=Ep[i*(self.n_mid_states*2+1):(i+1)*(self.n_mid_states*2+1)].clone().detach().cpu().numpy()

            h_i=h[i].clone().detach().cpu().numpy()
            x_i,Fn_i,Rn_i,En_i,Fp_i,Rp_i,Ep_i,h_i=self.NEB_ode_step(x_i,Fn_i,Rn_i,En_i,Fp_i,Rp_i,Ep_i,h_i,fmax)
            #print (Rn_i,En_i)

            x[i*(self.n_mid_states*2+1):(i+1)*(self.n_mid_states*2+1)][:,:self.natoms]=torch.tensor(x_i.reshape(-1,self.natoms,3)).to(x)
            Fn[i*(self.n_mid_states*2+1):(i+1)*(self.n_mid_states*2+1)][:,:self.natoms]=torch.tensor(Fn_i.reshape(-1,self.natoms,3)).to(Fn)
            Rn[i]=torch.tensor(Rn_i).to(Rn)
            En[i*(self.n_mid_states*2+1):(i+1)*(self.n_mid_states*2+1)]=torch.tensor(En_i).to(En)
            
            Fp[i*(self.n_mid_states*2+1):(i+1)*(self.n_mid_states*2+1)][:,:self.natoms]=torch.tensor(Fp_i.reshape(-1,self.natoms,3)).to(Fp)
            Rp[i]=torch.tensor(Rp_i).to(Rp)
            Ep[i*(self.n_mid_states*2+1):(i+1)*(self.n_mid_states*2+1)]=torch.tensor(Ep_i).to(Ep)
            h[i]=torch.tensor(h_i).to(h)
        print ('Fmax in ODE Guidance are ',self.Rn)
        return x,Fn,Rn,En,Fp,Rp,Ep,h

    def path_ode_gen(
        self,
        atoms,
        rfeats: Tensor,pfeats: Tensor,
        radjs: Tensor,padjs: Tensor,
        redges:Tensor,pedges:Tensor,
        rcoords: Tensor,pcoords: Tensor,
        sigmas: Iterable[Union[Tensor, float]],
        masks: Tensor,rtol=1e-1):

        self.init_reaction_path_variables(atoms,rfeats,pfeats,radjs,padjs,redges,pedges,rcoords,pcoords,masks,sigmas)
        rp_middles=rearrange(self.x,"(b n) c d -> b n c d", n=2*self.n_mid_states+1)
        self.init_neb_environ(rp_middles[0],rcoords[0],pcoords[0])

        with torch.no_grad():
            sigma = torch.full((self.x.shape[0],), sigmas[0], dtype=self.x.dtype, device=self.x.device)
            sigma_ts=sigma[self.ts_masks]
            self.x=self.x + sigmas[0] * torch.randn_like(self.x)
            self.ts_x=self.x[self.ts_masks]
            self.x,self.ts_x,self.ts_e,self.ts_masks=self.diff_ts_path_step(self.x,self.ts_x,sigma,sigma_ts)
        if GP.with_energy_guide:
            self.Fn,self.Rn,self.En=self.NEB_forces_energies_batch(self.x)
            #print (self.En)
            self.Fp,self.Rp,self.Ep=self.Fn,self.Rn,self.En
            #print ('First',self.Fn.shape,self.Rn.shape,self.En.shape)
            self.h= 0.5 * rtol ** 0.5 / self.Rp
            self.h=torch.where(self.h>1e-10,self.h,1e-10)
            print (self.h)
            #print (self.x.shape)
            self.x,self.Fn,self.Rn,self.En,self.Fp,self.Rp,self.Ep,self.h=self.NEB_ode_guide(self.x,self.Fn,self.Rn,self.En,self.Fp,self.Rp,self.Ep,self.h)

        pbar=tqdm(sigmas[1:-1],disable=(not False))

        for sid,sigma in tqdm(enumerate(pbar)):            
            pbar.set_description(f"interpolating (σ={sigma:.4f})")
            with torch.no_grad():
                sigma=torch.full((self.x.shape[0],),sigma,dtype=self.x.dtype,device=self.x.device)
                sigma_ts=sigma[self.ts_masks]

                #self.x = self.x + pad_dims_like((sigma**2-self.sigma_min**2)**0.5,self.x)*torch.randn_like(self.x)
                #self.ts_x=self.x[self.ts_masks]

                self.x,self.ts_x,self.ts_e,self.ts_masks=self.diff_ts_path_step(self.x,self.ts_x,sigma,sigma_ts)
            if GP.with_energy_guide:
                self.x,self.Fn,self.Rn,self.En,self.Fp,self.Rp,self.Ep,self.h=self.NEB_ode_guide(self.x,self.Fn,self.Rn,self.En,self.Fp,self.Rp,self.Ep,self.h)
                
        self.Fn,self.Rn,self.En=self.NEB_forces_energies_batch(self.x)
        self.nn_e=self.En[self.ts_masks]
        self.nn_max_e=torch.max(self.En.reshape(-1,self.n_mid_states*2+1),dim=1).values
        print (self.nn_e)
        return self.x,self.ts_x,self.ts_e,self.nn_e,self.nn_max_e,self.ts_masks

    def ode12r(f, X0, h=None, verbose=1, fmax=1e-6, maxtol=1e3, steps=100,
           rtol=1e-1, C1=1e-2, C2=2.0, hmin=1e-10, extrapolate=3,
           callback=None, apply_precon=None, converged=None, residual=None):
 
        X = X0
        Fn = f(X)
        
        if callback is None:
            def callback(X):
                pass
 
        if residual is None:
            def residual(F, X):
                return np.linalg.norm(F, np.inf)
        Rn = residual(Fn, X)
 
        if apply_precon is None:
            def apply_precon(F, X):
                return F, residual(F, X)
        Fp, Rp = apply_precon(Fn, X)
 
        def log(*args):
            if verbose >= 1:
                print(*args)
 
        def debug(*args):
            if verbose >= 2:
                print(*args)
 
        if converged is None:
            def converged(F, X):
                return residual(F, X) <= fmax
 
        if converged(Fn, X):
            log("ODE12r terminates successfully after 0 iterations")
            return X
        if Rn >= maxtol:
            raise OptimizerConvergenceError(f"ODE12r: Residual {Rn} is too large "
                                            "at iteration 0")
 
        # computation of the initial step
        r = residual(Fp, X)  # pick the biggest force
        if h is None:
            h = 0.5 * rtol ** 0.5 / r  # Chose a stepsize based on that force
            h = max(h, hmin)  # Make sure the step size is not too big
 
        for nit in range(1, steps + 1):
            Xnew = X + h * Fp  # Pick a new position
            Fn_new = f(Xnew)  # Calculate the new forces at this position
            Rn_new = residual(Fn_new, Xnew)
            Fp_new, Rp_new = apply_precon(Fn_new, Xnew)
 
            e = 0.5 * h * (Fp_new - Fp)  # Estimate the area under the forces curve
            err = np.linalg.norm(e, np.inf)  # Error estimate
 
            # Accept step if residual decreases sufficiently and/or error acceptable
            accept = ((Rp_new <= Rp * (1 - C1 * h)) or
                      ((Rp_new <= Rp * C2) and err <= rtol))
 
            # Pick an extrapolation scheme for the system & find new increment
            y = Fp - Fp_new
            if extrapolate == 1:  # F(xn + h Fp)
                h_ls = h * (Fp @ y) / (y @ y)
            elif extrapolate == 2:  # F(Xn + h Fp)
                h_ls = h * (Fp @ Fp_new) / (Fp @ y + 1e-10)
            elif extrapolate == 3:  # min | F(Xn + h Fp) |
                h_ls = h * (Fp @ y) / (y @ y + 1e-10)
            else:
                raise ValueError(f'invalid extrapolate value: {extrapolate}. '
                                 'Must be 1, 2 or 3')
            if np.isnan(h_ls) or h_ls < hmin:  # Rejects if increment is too small
                h_ls = np.inf
 
            h_err = h * 0.5 * np.sqrt(rtol / err)
 
            # Accept the step and do the update
            if accept:
                X = Xnew
                Rn = Rn_new
                Fn = Fn_new
                Fp = Fp_new
                Rp = Rp_new
                callback(X)
 
                # We check the residuals again
                if Rn >= maxtol:
                    raise OptimizerConvergenceError(
                        f"ODE12r: Residual {Rn} is too "
                        f"large at iteration number {nit}")
 
                if converged(Fn, X):
                    log("ODE12r: terminates successfully "
                        f"after {nit} iterations.")
                    return X
 
                # Compute a new step size.
                # Based on the extrapolation and some other heuristics
                h = max(0.25 * h,
                        min(4 * h, h_err, h_ls))  # Log steep-size analytic results
 

            else:
                # Compute a new step size.
                h = max(0.1 * h, min(0.25 * h, h_err,
                                     h_ls))

 
            # abort if step size is too small
            if abs(h) <= hmin:
                raise OptimizerConvergenceError('ODE12r terminates unsuccessfully'
                                                f' Step size {h} too small')
