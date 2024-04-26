from utils import *
import mmcv
from mmcv import Config
from mmcv.cnn import get_model_complexity_info

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
from pycls.models.build import BACKBONES as PYCLS_BACKBONES
from pycls.models.heads import HEADS as PYCLS_HEADS

# from pyskl.models.builder import BACKBONES as PYSKL_BACKBONES
# from pyskl.models.heads import HEADS as PYSKL_HEADS

from mmcv.utils import build_from_cfg
from mmcv.runner import DistSamplerSeedHook, EpochBasedRunner, OptimizerHook, build_optimizer, get_dist_info
import torch.distributed as dist
import numpy as np
import random
from functools import partial
from mmcv.parallel import collate
import mmcv
from pycls.models.utils import Graph


parser = argparse.ArgumentParser(description='Training code for GFST-Net')

parser.add_argument('--data_url', default='./data', type=str,
                    help='path to the dataset (ntu60rgb+d)')

parser.add_argument('--work_dirs', default='./output', type=str,
                    help='path to save log and checkpoints')

parser.add_argument('--train_stage', default=3, type=int,
                    help='select training stage, see our paper for details \
                          stage-1 : warm-up \
                          stage-2 : learn to select patches with RL \
                          stage-3 : finetune GCNs')

parser.add_argument('--model_arch', default='', type=str,
                    help='architecture of the model to be trained \
                         stgcn++pooling')

parser.add_argument('--T', default=6, type=int,
                    help='maximum length of the sequence of Glance + Focus')

parser.add_argument('--resume', default='', type=str,
                    help='path to the checkpoint for resuming')

parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')

parser.add_argument('--config', default='', type=str,
                    help='path to config file')


parser.add_argument('--GE_model_path', default='', type=str,
                    help='path to GE')

parser.add_argument('--last_stage_checkpoint', default='', type=str,
                    help='path to checkpoint from last stage')

args = parser.parse_args()
cfg = Config.fromfile(args.config)

def average_clip(cls_score):
    """Averaging class score over multiple clips.

    Using different averaging types ('score' or 'prob' or None, which defined in test_cfg) to computed the final
    averaged class score. Only called in test mode. By default, we use 'prob' mode.

    Args:
        cls_score (torch.Tensor): Class score to be averaged.

    Returns:
        torch.Tensor: Averaged class score.
    """
    assert len(cls_score.shape) == 3  # * (Batch, NumSegs, Dim)
    average_clips = 'prob'
    if average_clips not in ['score', 'prob', None]:
        raise ValueError(f'{average_clips} is not supported. Supported: ["score", "prob", None]')

    if average_clips is None:
        return cls_score

    if average_clips == 'prob':
        return F.softmax(cls_score, dim=2).mean(dim=1)
    elif average_clips == 'score':
        return cls_score.mean(dim=1)


def stgcn_test(model, pmodel_list, fc, memory, ppo, test_loader, criterion, graph, record_path, args):
    record_file = record_path + '/record.txt'
    cfg = Config.fromfile(args.config)
    save_path = record_path + '/model_best.pth.tar'
    resume_ckp = torch.load(save_path)
    start_epoch = resume_ckp['epoch']
    print('resume from epoch: {}'.format(start_epoch))
    model.load_state_dict(resume_ckp['model_state_dict'])
    pmodel_list.load_state_dict(resume_ckp['pmodel_1_state_dict'])
    fc.load_state_dict(resume_ckp['fc'])

    if ppo:
        ppo.policy.load_state_dict(resume_ckp['policy'])
        ppo.policy_old.load_state_dict(resume_ckp['policy'])
        ppo.optimizer.load_state_dict(resume_ckp['ppo_optimizer'])

    best_acc = resume_ckp['best_acc']


    batch_time = AverageMeter()
    losses = [AverageMeter() for _ in range(args.T)]
    top1 = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]



    model.eval()
    for m in pmodel_list:
        m.eval()
    fc.eval()


    dataset = test_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    fd = open(record_file, 'a+')
    with torch.no_grad():
        for data_id, data in enumerate(test_loader):

            input = data['keypoint'].cuda()
            target_var = data['label'].cuda()
            bs, nc = input.shape[:2]
            input = input.reshape((bs * nc,) + input.shape[2:])

            batch_size = len(input)
            loss_cla = []
            output, state = model(input)
            output = fc(output, restart_batch=True)
            output = output.reshape(bs, nc, output.shape[-1])
            output = average_clip(output)

            acc = accuracy(output, target_var, topk=(1,))
            step0_acc = acc
            top1[0].update(acc.sum(0).mul_(100.0 / bs).data.item(), bs)

            confidence_last = torch.gather(F.softmax(output.detach(), 1), dim=1, index=target_var.view(-1, 1)).view(1,
                                                                                                                    -1)

            for patch_step in range(1, args.T):
                focus_size = 0#patch_step - 1
                part_count = len(graph.part_node[focus_size])
                if args.train_stage == 1:
                    s_action = torch.floor(torch.rand(input.size(0), 1).cuda().squeeze() * part_count).int()
                    t_action = torch.rand(input.size(0), 1).cuda()
                else:
                    if patch_step == 1:
                        s_action, t_action = ppo.select_action(state.to(0), memory, restart_batch=True, training=False)

                    else:
                        s_action, t_action = ppo.select_action(state.to(0), memory, training=False)

                patches_list, partid_list = get_crop_pmodel(input, s_action, t_action, graph.part_node, focus_size)

                patches = torch.cat(patches_list, 0)
                output, state = pmodel_list[focus_size](
                    patches.clone())
                output = fc(output, restart_batch=False)
                output = output.reshape(bs, nc, output.shape[-1])
                output = average_clip(output)

                loss = criterion(output, target_var)
                loss_cla.append(loss)
                losses[patch_step].update(loss.data.item(), input.size(0))

                acc = accuracy(output, target_var, topk=(1,))
                if acc[0].item() > step0_acc[0].item():
                    print(data_id)
                top1[patch_step].update(acc.sum(0).mul_(100.0 / bs).data.item(), bs)

                confidence = torch.gather(F.softmax(output.detach(), 1), dim=1, index=target_var.view(-1, 1)).view(1,
                                                                                                                   -1)
                reward = confidence - confidence_last
                confidence_last = confidence

                reward_list[patch_step - 1].update(reward.data.mean(), input.size(0))
                memory.rewards.append(reward)

            memory.clear_memory()
            batch_size = len(output)
            for _ in range(batch_size):
                prog_bar.update()


        _acc = [acc.ave for acc in top1]
        print('accuracy of each step:')
        print(_acc)
        fd.write('accuracy of each step:\n')
        fd.write(str(_acc) + '\n')

        _reward = [reward.ave for reward in reward_list]
        print('reward of each step:')
        print(_reward)
        fd.write('reward of each step:\n')
        fd.write(str(_reward) + '\n')

    fd.close()

def test_ge(model, fc, memory, test_loader,model_path,record_path, args):
    cfg = Config.fromfile(args.config)
    resume_ckp = torch.load(model_path)
    start_epoch = resume_ckp['epoch']
    print('resume from epoch: {}'.format(start_epoch))
    model.load_state_dict(resume_ckp['model_state_dict'])
    fc.load_state_dict(resume_ckp['fc'])

    batch_time = AverageMeter()

    top1 = [AverageMeter() for _ in range(1)]
    if record_path is not None:
        record_file = record_path + '/record.txt'
        fd = open(record_file+'', 'a+')

    model.eval()

    fc.eval()


    dataset = test_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    test_logit = []
    with torch.no_grad():
        for data in test_loader:

            input = data['keypoint'].cuda()
            target_var = data['label'].cuda()
            bs, nc = input.shape[:2]
            input = input.reshape((bs * nc,) + input.shape[2:])
            output,_ = model(input)
            output = fc(output)
            output = output.reshape(bs, nc, output.shape[-1])
            output = average_clip(output)
            test_logit.append(output)
            acc = accuracy(output, target_var, topk=(1,))
            top1[0].update(acc.sum(0).mul_(100.0 / bs).data.item(), bs)
            memory.clear_memory()
            batch_size = len(output)
            for _ in range(batch_size):
                prog_bar.update()

        test_logit = torch.cat(test_logit,0)
        torch.save(test_logit,record_path+'/test_logit.pt')
        _acc = [acc.ave for acc in top1]
        print('accuracy of each step:')
        print(_acc)
        if record_path is not None:
            fd.write('accuracy of each step:\n')
            fd.write(str(_acc) + '\n')

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



if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    test_main()