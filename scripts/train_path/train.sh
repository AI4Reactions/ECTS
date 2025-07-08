#CUDA_VISIBLE_DEVICES=2,3 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 2 --rdzv_id 1 train_ts_and_e.py -i ctrl_ts_e.json
CUDA_VISIBLE_DEVICES=2,3 torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nnodes 1 --nproc_per_node 2 --rdzv_id 1 train_path.py -i ctrl_path.json
cd EcTs_Model/model 
mv online_model_perepoch.cpk path_online_model_perepoch.cpk
