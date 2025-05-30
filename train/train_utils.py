#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
from evaluation.evaluate_utils import PerformanceMeter
from utils.utils import AverageMeter, ProgressMeter, get_output


def get_loss_meters(p):
    """ Return dictionary with loss meters to monitor training """
    all_tasks = p.ALL_TASKS.NAMES
    tasks = p.TASKS.NAMES


    if p['model'] == 'mti_net': # Extra losses at multiple scales
        losses = {}
        for scale in range(4):
            for task in all_tasks:
                losses['scale_%d_%s' %(scale, task)] = AverageMeter('Loss scale-%d %s ' %(scale+1, task), ':.4e')
        for task in tasks:
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')


    elif p['model'] == 'pad_net': # Extra losses because of deepsupervision
        losses = {}
        for task in all_tasks:
            losses['deepsup_%s' %(task)] = AverageMeter('Loss deepsup %s' %(task), ':.4e')
        for task in tasks:
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')


    else: # Only losses on the main task.
        losses = {task: AverageMeter('Loss %s' %(task), ':.4e') for task in tasks}


    losses['total'] = AverageMeter('Loss Total', ':.4e')
    return losses


def train_vanilla(p, train_loader, model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights """
    losses = get_loss_meters(p)
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
        output = model(images)
        
        # Measure loss and performance
        loss_dict = criterion(output, targets)
        for k, v in loss_dict.items():
            losses[k].update(v.item())
        performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES}, 
                                 {t: targets[t] for t in p.TASKS.NAMES})
        
        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results

def train_fp16(p, train_loader, model, criterion, optimizer, epoch, scaler, aux_loss=False):
    """ Vanilla training with fixed loss weights """
    losses = get_loss_meters(p)
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))

    model.train()

    if epoch == 0:
        for i, batch in enumerate(train_loader):
            # Forward pass
            images = batch['image'].cuda(non_blocking=True)
            targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
            with torch.cuda.amp.autocast():
                output = model(images)
            
                # Measure loss and performance
                loss_dict = criterion(output, targets)

            for k, v in loss_dict.items():
                losses[k].update(v.item())
            performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES}, 
                                    {t: targets[t] for t in p.TASKS.NAMES})
            
            # Backward

            optimizer.zero_grad()
            task_id = 0
            num_tasks = len(p.TASKS.NAMES)
            for task in p.TASKS.NAMES:
                if task_id < num_tasks - 1:
                    scaler.scale(loss_dict[task]).backward(retain_graph=True)
                else:
                    scaler.scale(loss_dict[task]).backward()
                task_id += 1

                print("task:", task)
                for name, param in model.named_parameters():
                    print(name, param.grad.shape)

            scaler.step(optimizer)
            scaler.update()

            if i % 25 == 0:
                progress.display(i)
    else:
        for i, batch in enumerate(train_loader):
            # Forward pass
            images = batch['image'].cuda(non_blocking=True)
            targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
            with torch.cuda.amp.autocast():
                output = model(images)
            
                # Measure loss and performance
                loss_dict = criterion(output, targets)

            for k, v in loss_dict.items():
                losses[k].update(v.item())
            performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES}, 
                                    {t: targets[t] for t in p.TASKS.NAMES})
            
            # Backward

            optimizer.zero_grad()
            if aux_loss:
                # 因为模型并行训练，所以aux_loss等于gpu的数量,所以求和之后要除以gpu数量
                loss_dict['total'] += output['aux_loss'].sum()/torch.cuda.device_count()
            scaler.scale(loss_dict['total']).backward()
            scaler.step(optimizer)

            scaler.update()

            if i % 25 == 0:
                progress.display(i)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results

from train.pcgrad import PCGrad
def train_pcgrad(p, train_loader, model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights """
    pc_optimizer = PCGrad(optimizer)
    losses = get_loss_meters(p)
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
        output = model(images)
        
        # Measure loss and performance
        loss_dict = criterion(output, targets)
        for k, v in loss_dict.items():
            losses[k].update(v.item())
        performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES}, 
                                 {t: targets[t] for t in p.TASKS.NAMES})
        
        # Backward
        pc_optimizer.zero_grad()
        pc_optimizer.pc_backward([loss_dict[t] for t in p.TASKS.NAMES])
        pc_optimizer.step()

        if i % 25 == 0:
            progress.display(i)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results

from train.pcgrad_amp import PCGradAMP
def train_pcgrad_amp_ddp(p, train_loader, model, criterion, optimizer, epoch, device_id):
    """ Vanilla training with fixed loss weights """
    scaler = torch.cuda.amp.GradScaler()
    num_tasks = len(p.TASKS.NAMES)
    pc_optimizer = PCGradAMP(num_tasks, optimizer, scaler=scaler, reduction='sum', cpu_offload= False)
    losses = get_loss_meters(p)
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].to(device_id)
        targets = {task: batch[task].to(device_id) for task in p.ALL_TASKS.NAMES}

        with torch.cuda.amp.autocast():
            output = model(images)        
            # Measure loss and performance
            loss_dict = criterion(output, targets)

        for k, v in loss_dict.items():
            losses[k].update(v.item())
        performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES}, 
                                 {t: targets[t] for t in p.TASKS.NAMES})
        
        # Backward
        pc_optimizer.zero_grad()
        pc_optimizer.backward([loss_dict[t] for t in p.TASKS.NAMES])
        pc_optimizer.step()

        if device_id == 0 and i % 25 == 0:
            progress.display(i)

    if device_id == 0:
        verbose = True
    else:
        verbose = False
        
    eval_results = performance_meter.get_score(verbose = verbose)

    return eval_results

def train_pcgrad_amp(p, train_loader, model, criterion, optimizer, epoch, scaler):
    """ Vanilla training with fixed loss weights """
    num_tasks = len(p.TASKS.NAMES)
    pc_optimizer = PCGradAMP(num_tasks, optimizer, scaler=scaler, reduction='sum', cpu_offload= False)
    losses = get_loss_meters(p)
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}

        with torch.cuda.amp.autocast():
            output = model(images)        
            # Measure loss and performance
            loss_dict = criterion(output, targets)

        for k, v in loss_dict.items():
            losses[k].update(v.item())
        performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES}, 
                                 {t: targets[t] for t in p.TASKS.NAMES})
        
        # Backward
        pc_optimizer.backward([loss_dict[t] for t in p.TASKS.NAMES])
        pc_optimizer.step()

        if i % 25 == 0:
            progress.display(i)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results

from train.gradvac_amp import GradVacAMP
def train_gradvac_amp(p, train_loader, model, criterion, optimizer, epoch, scaler):
    """ Vanilla training with fixed loss weights """
    num_tasks = len(p.TASKS.NAMES)
    DEVICE = next(model.parameters()).device
    vac_optimizer = GradVacAMP(num_tasks, optimizer, DEVICE, scaler = scaler, beta = 1e-2, reduction='sum', cpu_offload = False)
    losses = get_loss_meters(p)
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}

        with torch.cuda.amp.autocast():
            output = model(images)        
            # Measure loss and performance
            loss_dict = criterion(output, targets)

        for k, v in loss_dict.items():
            losses[k].update(v.item())
        performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES}, 
                                 {t: targets[t] for t in p.TASKS.NAMES})
        
        # Backward
        vac_optimizer.backward([loss_dict[t] for t in p.TASKS.NAMES])
        vac_optimizer.step()

        if i % 25 == 0:
            progress.display(i)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results