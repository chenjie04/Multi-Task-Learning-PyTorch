#!/usr/bin/env bash 
date

python get_flops.py --config_env configs/env.yml --config_exp configs/pascal/squadnet/squadnet_base.yml --shape 512 512 # nudy 480 640

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

torchrun --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29501 \
    main_pcgrad_ddp.py --config_env configs/env.yml \
    --config_exp configs/pascal/squadnet/squadnet_base.yml

date
