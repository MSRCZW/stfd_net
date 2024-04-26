# version 4
import time

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from mmcv import Config

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
import collections

from samplers import DistributedSampler
import os
from utils import *
from network import *
from configs import *
from math import cos, pi
import math
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from pycls.datasets import RepeatDataset
from pycls.datasets import PoseDataset
from pycls.models.build import BACKBONES
from pycls.models.heads import HEADS
from mmcv.utils import build_from_cfg
from mmcv.runner import DistSamplerSeedHook, EpochBasedRunner, OptimizerHook, build_optimizer, get_dist_info
import torch.distributed as dist
import numpy as np
import random
from functools import partial
from mmcv.parallel import collate
import mmcv
from torch.nn.parallel import DistributedDataParallel as DDP
from pycls.models.utils import Graph
from test import test_backbone, stgcn_test, stgcn_likestn_test,test_ge

parser = argparse.ArgumentParser(description='Training code for GFST-Net')

parser.add_argument('--work_dirs', default='./output', type=str,
                    help='path to save log and checkpoints')
parser.add_argument('--resume', default='', type=str,
                    help='path to the checkpoint for resuming')
parser.add_argument('--local_rank', type=int,
                    help='node rank for distributed training')
parser.add_argument('--config', default='', type=str,
                    help='path to config file')
args = parser.parse_args()
cfg = Config.fromfile(args.config)

dist.init_process_group(backend='nccl')
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def stgcn_train(model,fc,memory, ppo, optimizer, train_loader, criterion, epoch, graph, args):
    """def train(model_prime, model, fc, memory, ppo, optimizer, train_loader, criterion,
          print_freq, epoch, batch_size, record_file, train_configuration, args):"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(train_loader)
    prog_bar = mmcv.ProgressBar(train_batches_num)

    model.train()
    fc.train()

    end = time.time()
    sum_iter = epoch * train_batches_num
    max_iter = cfg.total_epochs * train_batches_num
    for i, (x, target) in enumerate(train_loader):
        batch_size = len(x)
        # torch.autograd.set_detect_anomaly(True)
        current_iter = sum_iter + i

        adjust_lr(optimizer, cfg.optimizer.lr, 0, current_iter, max_iter)
        assert x.shape[1] == 1
        x = x[:, 0]
        x = x.cuda()
        target_var = target.cuda()

        input = x


        output,_ = model(input)
        output = fc(output)

        loss = criterion(output, target_var)

        losses.update(loss.data.item(), x.size(0))
        acc = accuracy(output, target_var, topk=(1,))
        top1.update(acc.sum(0).mul_(100.0 / batch_size).data.item(), x.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        memory.clear_memory()

        batch_time.update(time.time() - end)
        end = time.time()

        prog_bar.update()

def stgcn_val(model, fc, memory, val_loader, criterion, graph, record_file, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    fc.eval()
    prog_bar = mmcv.ProgressBar(len(val_loader))

    fd = open(record_file, 'a+')
    with torch.no_grad():
        for data in val_loader:
            assert data['keypoint'].shape[1] == 1
            input = data['keypoint'][:, 0].cuda()
            target_var = data['label'].cuda()
            batch_size = len(input)
            x = input
            output,_ = model(x)
            output = fc(output)
            loss_prime = criterion(output, target_var)
            losses.update(loss_prime.data.item(), input.size(0))
            acc = accuracy(output, target_var, topk=(1,))
            top1.update(acc.sum(0).mul_(100.0 / batch_size).data.item(), input.size(0))
            memory.clear_memory()
            prog_bar.update()
        print('\n')
        save_string = ('Epoch: [{0}]\t'
                       'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                       'Loss {loss.value:.4f} ({loss.ave:.4f})\t'.format(
            epoch, batch_time=batch_time, loss=losses))
        print(save_string)
        fd.write(save_string + '\n')

        _acc = top1.ave
        print('accuracy :')
        print(_acc)
        fd.write(str(_acc) + '\n')

    fd.close()
    result = top1.ave

    return result

def stgcn_main():
    """
    create record file path
    """
    graph = Graph(**cfg.model.backbone.graph_cfg)
    record_path = cfg.work_dir + '/GF-' + f"{cfg.model.backbone.type}"
    if not os.path.exists(record_path):
        mkdir_p(record_path)
    record_file = record_path + '/record.txt'

    """
        create model
    """
    rank, world_size = get_dist_info()
    model = build_from_cfg(cfg.model.backbone, BACKBONES)


    fc = build_from_cfg(cfg.model.cls_head, HEADS)


    optimizer = torch.optim.SGD(
        [{'params': model.parameters()},
         {'params': fc.parameters()}, ],
        lr=cfg.optimizer.lr,
        momentum=cfg.optimizer.momentum,
        nesterov=cfg.optimizer.nesterov,
        weight_decay=cfg.optimizer.weight_decay)
    training_epoch_num = cfg.total_epochs


    criterion = nn.CrossEntropyLoss().cuda()


    model = DDP(model.cuda(), device_ids=[int(os.environ['LOCAL_RANK'])],)

    fc = DDP(fc.cuda(), device_ids=[int(os.environ['LOCAL_RANK'])],)
    batch_size = cfg.data.videos_per_gpu

    train_set = RepeatDataset(**cfg.data.train)
    test_set = PoseDataset(**cfg.data.test, test_mode=True)
    val_set = PoseDataset(**cfg.data.val, test_mode=True)



    train_sampler = DistributedSampler(
        train_set, world_size, rank, shuffle=True, seed=None)
    test_sampler = DistributedSampler(
        test_set, world_size, rank, shuffle=False, seed=None)
    val_sampler = DistributedSampler(
        val_set, world_size, rank, shuffle=False, seed=None)

    init_fn = partial(
        worker_init_fn, num_workers=16, rank=rank,
        seed=0)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=8,
        collate_fn=partial(collate, samples_per_gpu=8),
        pin_memory=True,
        shuffle=False,
        worker_init_fn=init_fn,
        drop_last=False,
        prefetch_factor=2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=8,
        collate_fn=partial(collate, samples_per_gpu=16),
        pin_memory=True,
        shuffle=False,
        worker_init_fn=init_fn,
        drop_last=False,
        prefetch_factor=2,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        sampler=test_sampler,
        num_workers=8,
        collate_fn=partial(collate, samples_per_gpu=16),
        pin_memory=True,
        shuffle=False,
        worker_init_fn=init_fn,
        persistent_workers=False,
        drop_last=False,
        prefetch_factor=2,
    )

    ppo = None
    memory = Memory()
    if args.resume:
        resume_ckp = torch.load(args.resume)
        start_epoch = resume_ckp['epoch']
        print('resume from epoch: {}'.format(start_epoch))
        model.load_state_dict(resume_ckp['model_state_dict'])
        fc.load_state_dict(resume_ckp['fc'])
        if optimizer:
            optimizer.load_state_dict(resume_ckp['optimizer'])
        best_acc = resume_ckp['best_acc']
    else:
        start_epoch = 0
        best_acc = 0
    for epoch in range(start_epoch, training_epoch_num):
        train_sampler.set_epoch(epoch)
        stgcn_train(model, fc, memory, ppo, optimizer, train_loader, criterion, epoch, graph, args)
        result = stgcn_val(model=model,
                                fc=fc,
                                memory=memory,
                                val_loader=val_loader,
                                criterion=criterion,
                                graph=graph,
                                record_file=record_file,
                                epoch=epoch + 1)


        if result > best_acc:
            best_acc = result
            is_best = True
        else:
            is_best = False

        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'fc': fc.state_dict(),
            'result_list': result,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict() if optimizer else None,
            'ppo_optimizer': ppo.optimizer.state_dict() if ppo else None,
            'policy': ppo.policy.state_dict() if ppo else None,
        }, result, is_best, epoch + 1, checkpoint=record_path)
    model_path = record_path+'/model_best.pth.tar'
    test_ge(model, fc, memory, test_loader,model_path,record_path, args)


def adjust_lr(optimizer, base_lr, target_lr, current_iter, max_iter, weight=1):
    factor = current_iter / max_iter
    cos_out = cos(pi * factor) + 1
    cur_lr = target_lr + 0.5 * weight * (base_lr - target_lr) * cos_out
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Init the random seed for various workers."""
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# os.environ['CUDA_VISIBLE_DEVICES']='1'
if __name__ == '__main__':
    stgcn_main()
