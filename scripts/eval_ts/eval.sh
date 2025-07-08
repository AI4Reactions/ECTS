CUDA_VISIBLE_DEVICES=3 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 1 --rdzv_id 1 eval.py -i ctrl_sample.json  --steps 1
CUDA_VISIBLE_DEVICES=3 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 1 --rdzv_id 1 eval.py -i ctrl_sample.json  --steps 2
CUDA_VISIBLE_DEVICES=3 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 1 --rdzv_id 1 eval.py -i ctrl_sample.json  --steps 5
CUDA_VISIBLE_DEVICES=3 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 1 --rdzv_id 1 eval.py -i ctrl_sample.json  --steps 10
CUDA_VISIBLE_DEVICES=3 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 1 --rdzv_id 1 eval.py -i ctrl_sample.json  --steps 15
CUDA_VISIBLE_DEVICES=3 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 1 --rdzv_id 1 eval.py -i ctrl_sample.json  --steps 25
