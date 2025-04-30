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

def cal_bond_(x,bond_indexes,mask=None,eps=1e-7):
    batch_num=x.shape[0]
    max_atom_num=x.shape[1]
    it=bond_indexes[:,0]*max_atom_num+bond_indexes[:,1]
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
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.initial_timesteps = initial_timesteps
        self.final_timesteps = final_timesteps

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
        next_energies, next_forces = energy_online_model( 
            rfeats,pfeats,
            redges,pedges,
            rcoords,pcoords,
            next_xyzs,
            next_sigmas,
            masks)

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
            current_energies, current_forces = energy_ema_model(
                rfeats,pfeats,
                redges,pedges,
                rcoords,pcoords,
                current_xyzs,
                current_sigmas,
                masks,
            )
        
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
    def __init__(self, sigma_min: float = 0.002, sigma_data: float = 0.5) -> None:
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data

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
        e,f = energy_model(
            rfeats, pfeats, redges, pedges, rcoords, pcoords, x, sigma, masks
        )

        #if clip_denoised:
        #    x = x.clamp(min=-1.0, max=1.0)
        x = self.__mask_transform(x, y, transform_mask, transform_fn, inverse_transform_fn)
        x_list.append(x)
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
            e,f = energy_model(
                rfeats, pfeats, redges, pedges, rcoords, pcoords, x, sigma, masks
            )
            #if clip_denoised:
            #    x = x.clamp(min=-1.0, max=1.0)
            x = self.__mask_transform(x, y, transform_mask, transform_fn, inverse_transform_fn)
            x_list.append(x)
            e_list.append(e)
            f_list.append(f)

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
            x = y + sigmas[0] * torch.randn_like(y) if add_initial_noise else y

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
            x=self.NEB_guide(x,atoms,rcoords,pcoords,guide_steps_interval=1)
            
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
                x=self.NEB_guide(x,atoms,rcoords,pcoords,guide_steps_interval=1)
        
            x_list.append(x)

        if not sample_path_only:
            ts_states=x[ts_masks]
            return x,ts_states,ts_e,x_list
        else:
            return x,None,None,x_list

    def NEB_guide(self,x,atoms,rcoords,pcoords,guide_steps_interval=1):
        for guide_step in range(guide_steps_interval):
            neb_forces=[]
            neb_energies=[]

            x_=x.clone().detach()
            
            x_=rearrange(x_,"(b n) c d -> b n c d", n=2*self.n_mid_states+1)
            
            neb_forces=[]
            neb_energies=[] 

            for i in range(x_.shape[0]):
                neb_force,force_max,neb_energy=NEB_step(atoms,x_[i],self.calc,rcoords[0],pcoords[0])
                neb_forces.append(neb_force)
                neb_energies.append(neb_energy)
                rpstr=f'Neb {guide_step} th force calculation in 1th diffusion step of {i}th reaction path -- Force max: {force_max:.4f}'
                print (rpstr)

            with torch.no_grad():
                neb_forces=torch.stack(neb_forces,dim=0)
                neb_forces=rearrange(neb_forces,"b n c d -> (b n) c d")
                x = x+neb_forces*self.alpha
        return x
    
    def __mask_transform(
        self,
        x: Tensor,
        y: Tensor,
        transform_mask: Tensor,
        transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        inverse_transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
    ) -> Tensor:
        return inverse_transform_fn(transform_fn(y) * (1.0 - transform_mask) + x * transform_mask)
    
