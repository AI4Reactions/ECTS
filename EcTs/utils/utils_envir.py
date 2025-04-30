import torch 
import numpy as np 
from random import random
from torch import distributed as dist

def Set_all_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def Summarize_model(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})

def Set_gpu_envir(rank):
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    return 

def Set_model_rank(model,rank):
    model.cuda(rank)
    torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[rank],output_device=rank,find_unused_parameters=True,broadcast_buffers=False)
    return model