#!/bin/bash
#source activate venv
source ~/.bashrc
conda activate venv
python train3d.py \
    --dataroot /ocean/projects/asc170022p/rohit33 \
    --dataset_mode copdpatchpartition \
    --gpu_ids 0,1 \
    --model cycle_gan_patch \
    --patchfloat 1 \
    --augment 1 \
    --batch_size 100 \
    --save_latest_freq 100 \
    --lambda_identity 0.2 \
    --display_freq 40 \
    --pool_size 500 \
    --netG resnet3d_9blocks_noadd \
    --netD patchgan_3d \
    --input_nc 1  \
    --output_nc 1 \
    --norm batch3d \
    --subroot emphysemafileindex \
    --partitions 5 \
    --name emphysema_resnet9_part5

