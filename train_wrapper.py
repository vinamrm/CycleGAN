import os
import sys
from .config import Config

def run(command):
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)


if __name__ == '__main__':
    
    c = Config()

    c.dataroot = sys.argv[1]
    run(f'python train.py --model cycle_gan --dataroot {c.dataroot} --name {c.name} --gpu_ids {c.gpu_ids} --checkpoints_dir {c.checkpoints_dir} 
        --model {c.model} --input_nc {c.input_nc} --output_nc {c.output_nc} --ngf {c.ngf} --ndf {c.ndf} --netD {c.netD} --netG {c.netG} 
        --n_layers_D {c.n_layers_D} --norm {c.norm} --init_type {c.init_type} --init_gain {c.init_gain} {'--no_dropout' if c.no_dropout else ''} 
        --dataset_mode multispectral --direction {c.direction} {'--serial_batches' if c.serial_batches else ''} --num_threads {c.num_threads} 
        --batch_size {c.batch_size} --load_size {c.load_size} --crop_size {c.crop_size} {f'--max_dataset_size {c.max_dataset_size}' if c.max_dataset_size else ''} 
        --preprocess {c.preprocess} {'--no_flip' if c.no_flip else ''} --display_winsize {c.display_winsize} --epoch {c.epoch} --load_iter {c.load_iter} --verbose
        --suffix {c.suffix} {'--use_wandb' if c.use_wandb else ''} --wandb_project_name {c.wandb_project_name} --save_latest_freq {c.save_latest_freq} 
        --save_epoch_freq {c.save_epoch_freq} {'--save_by_iter' if c.save_by_iter else ''} {'--continue_train' if c.continue_train else ''} --epoch_count {c.epoch_count} 
        --phase {c.phase} --n_epochs {c.n_epochs} --n_epochs_decay {c.n_epochs_decay} --beta1 {c.beta1} --lr {c.lr} --gan_mode {c.gan_mode} --pool_size {c.pool_size} 
        --lr_policy {c.lr_policy} --lr_decay_iters {c.lr_decay_iters}')
