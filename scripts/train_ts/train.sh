#CUDA_VISIBLE_DEVICES=2,3 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 2 --rdzv_id 1 train_ts_and_e.py -i ctrl_ts_e.json
#CUDA_VISIBLE_DEVICES=2,3 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 2 --rdzv_id 1 train_ts.py -i ctrl_ts.json
CUDA_VISIBLE_DEVICES=2,3 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 2 --rdzv_id 1 train_e.py -i ctrl_e.json
