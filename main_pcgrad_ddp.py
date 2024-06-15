#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import numpy as np
import sys
import torch

from utils.config import create_config
from utils.common_config import get_train_dataset, get_transformations,\
                                get_val_dataset, get_train_dataloader_ddp, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate,\
                                get_criterion
from utils.logger import Logger
from train.train_utils import train_pcgrad_amp_ddp
from evaluation.evaluate_utils import eval_model, validate_results, save_model_predictions,\
                                    eval_all_results
from termcolor import colored
import torch.distributed as dist
import torch.nn as nn

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# Parser
parser = argparse.ArgumentParser(description='Vanilla Training')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

def main():

    torch.manual_seed(42)  # 设置随机种子以确保可重复性
    # 设置所有GPU的随机种子
    torch.cuda.manual_seed_all(42)

    dist.init_process_group(backend='nccl')
    dist.barrier()
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()
    print(f"Device id: {device_id}")

    # Retrieve config file
    cv2.setNumThreads(0)
    p = create_config(args.config_env, args.config_exp)
    if device_id == 0:
        sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'))
        print(colored(p, 'red'))

    # Get model
    if device_id == 0:
        print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    model.apply(weights_init)
    # create model and move it to GPU with id rank
    
    model = model.to(device_id)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id],
                                                      output_device=device_id, find_unused_parameters=True)

    # Get criterion
    if device_id == 0:
        print(colored('Get loss', 'blue'))
    criterion = get_criterion(p)
    criterion.to(device_id)
    if device_id == 0:
        print(criterion)

    # CUDNN
    if device_id == 0:
        print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Optimizer
    if device_id == 0:
        print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    if device_id == 0:
        print(optimizer)

    # Dataset
    if device_id == 0:
        print(colored('Retrieve dataset', 'blue'))
    
    # Transforms 
    train_transforms, val_transforms = get_transformations(p)
    train_dataset = get_train_dataset(p, train_transforms)
    val_dataset = get_val_dataset(p, val_transforms)
    true_val_dataset = get_val_dataset(p, None) # True validation dataset without reshape 
    train_dataloader, train_sampler = get_train_dataloader_ddp(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    if device_id == 0:
        print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
        print('Train transformations:')
        print(train_transforms)
        print('Val transformations:')
        print(val_transforms)
    
    # Resume from checkpoint
    if os.path.exists(p['checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['checkpoint']), 'blue'))
        checkpoint = torch.load(p['checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best_result = checkpoint['best_result']

    else:
        print(colored('No checkpoint file at {}'.format(p['checkpoint']), 'blue'))
        start_epoch = 0
        save_model_predictions(p, val_dataloader, model)
        best_result = eval_all_results(p)
    

    # Main loop
    if device_id == 0:
        print(colored('Starting main loop', 'blue'))

    for epoch in range(start_epoch, p['epochs']):
        if device_id == 0:
            print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
            print(colored('-'*10, 'yellow'))

        train_sampler.set_epoch(epoch)

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)

        if device_id == 0:
            print('Adjusted learning rate to {:.8f}'.format(lr))
            # Train 
            print('Train ...')

        eval_train = train_pcgrad_amp_ddp(p, train_dataloader, model, criterion, optimizer, epoch, device_id)

        # Evaluate
            # Check if need to perform eval first
        if 'eval_final_10_epochs_only' in p.keys() and p['eval_final_10_epochs_only']: # To speed up -> Avoid eval every epoch, and only test during final 10 epochs.
            if epoch + 1 > p['epochs'] - 10:
                eval_bool = True
            else:
                eval_bool = False
        else:
            if 'eval_freq' in p.keys():
                if (epoch + 1) % p['eval_freq'] == 0:
                    eval_bool = True
                else:
                    eval_bool = False
            else:
                if (epoch + 1) % 10 == 0:
                    eval_bool = True
                else:
                    eval_bool = False


        # Perform evaluation
        if eval_bool:
            print('Evaluate ...')
            save_model_predictions(p, val_dataloader, model)
            curr_result = eval_all_results(p)
            improves, best_result = validate_results(p, curr_result, best_result)
            if improves:
                print('Save new best model')
                torch.save(model.state_dict(), p['best_model'])

        # Checkpoint
        
        if device_id == 0:
            print('Checkpoint ...')
            torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1, 'best_result': best_result}, p['checkpoint'])

    # Evaluate best model at the end
    if device_id == 0:
        print(colored('Evaluating best model at the end', 'blue'))
        model.load_state_dict(torch.load(p['checkpoint'])['model'])
        save_model_predictions(p, val_dataloader, model)
        eval_stats = eval_all_results(p)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
