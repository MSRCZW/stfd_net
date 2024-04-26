# version 4
import time
from mmcv import Config
from samplers import DistributedSampler
from utils import *
from network import *
from math import cos, pi
import argparse
from pycls.datasets import RepeatDataset
from pycls.datasets import PoseDataset
from pycls.models.build import BACKBONES
from pycls.models.heads import HEADS
from mmcv.utils import build_from_cfg
from mmcv.runner import get_dist_info
import torch.distributed as dist
import numpy as np
import random
from functools import partial
from mmcv.parallel import collate
import mmcv
from pycls.models.utils import Graph
from test import stgcn_test

parser = argparse.ArgumentParser(description='Training code for GFNet')


parser.add_argument('--work_dirs', default='./output', type=str,
                    help='path to save log and checkpoints')

parser.add_argument('--train_stage', default=-1, type=int,
                    help='select training stage, see our paper for details \
                          stage-1 : warm-up \
                          stage-2 : learn to select patches with RL \
                          stage-3 : finetune CNNs')

parser.add_argument('--model_arch', default='', type=str,
                    help='name of the model to be trained')

parser.add_argument('--T', default=3, type=int,
                    help='maximum length of the sequence of Glance + Focus')
parser.add_argument('--resume', default='', type=str,
                    help='path to the checkpoint for resuming')
parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--config', default='', type=str,
                    help='path to backbone')
parser.add_argument('--GE_model_path', default='', type=str,
                    help='path to GE model')
parser.add_argument('--last_stage_checkpoint', default='', type=str,
                    help='path to checkpoint from last stage')
args = parser.parse_args()
cfg = Config.fromfile(args.config)

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)



def stgcn_train(model, model_list, fc, dsn_fc, memory, ppo, optimizer, train_loader, criterion, epoch, graph, args):
    """def train(model_prime, model, fc, memory, ppo, optimizer, train_loader, criterion,
          print_freq, epoch, batch_size, record_file, train_configuration, args):"""

    # Initialize various metrics for monitoring progress
    batch_time = AverageMeter()
    losses = [AverageMeter() for _ in range(args.T)]
    top1 = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]

    train_batches_num = len(train_loader)
    prog_bar = mmcv.ProgressBar(train_batches_num)

    # Set the model to evaluation mode if in stage 2, else set to training mode
    if args.train_stage == 2:
        model.eval()
        fc.eval()
        for m in model_list:
            m.eval()
        dsn_fc.eval()
    else:
        if args.GE_model_path:
            model.eval()
        else:
            model.train()
        fc.train()
        for m in model_list:
            m.train()
        dsn_fc.train()
    # Initialize time and iteration counters
    end = time.time()
    sum_iter = epoch * train_batches_num
    max_iter = cfg.total_epochs * train_batches_num
    # Loop over the training data
    for i, (x, target) in enumerate(train_loader):
        # Prepare data and target
        batch_size = len(x)
        loss_cla = []
        loss_list_dsn = []
        current_iter = sum_iter + i
        if args.train_stage != 2:
            adjust_lr(optimizer, cfg.optimizer.lr, 0, current_iter, max_iter)
        assert x.shape[1] == 1
        x = x[:, 0]
        x = x.cuda()

        target_var = target.cuda()
        input = x

        # Forward pass of glimpse step
        if args.train_stage != 2:
            if args.GE_model_path:
                with torch.no_grad():
                    output, state = model(input)
            else:
                output, state = model(input)
            dsn_output = dsn_fc.module[0](output)
            output = fc(output, restart_batch=True)

        else:
            with torch.no_grad():
                output, state = model(input)
                dsn_output = dsn_fc.module[0](output)
                output = fc(output, restart_batch=True)
        # Loss computation and backpropagation
        loss = criterion(output, target_var)
        loss_cla.append(loss)
        loss_dsn = criterion(dsn_output, target_var)
        loss_list_dsn.append(loss_dsn)
        losses[0].update(loss.data.item(), x.size(0))
        acc = accuracy(output, target_var, topk=(1,))
        top1[0].update(acc.sum(0).mul_(100.0 / batch_size).data.item(), x.size(0))
        part_count = len(graph.part_node[0])
        # Forward pass of zoom step
        for patch_step in range(1, args.T):

            focus_size = 0
            if args.train_stage == 1:
                s_action = torch.floor(torch.rand(x.size(0), 1).cuda().squeeze()*part_count).int()
                t_action = torch.rand(x.size(0), 1).cuda()
            else:
                if patch_step == 1:
                    s_action, t_action = ppo.select_action(state.to(0), memory, restart_batch=True)
                else:
                    s_action, t_action = ppo.select_action(state.to(0), memory)
            patches_list, partid_list = get_crop_pmodel(x, s_action, t_action, graph.part_node, focus_size)

            if args.train_stage != 2:
                patches = torch.cat(patches_list, 0)
                output, state = model_list[focus_size](
                    patches.clone())
                dsn_output = dsn_fc.module[focus_size + 1](output)
                output = fc(output, restart_batch=False)
            else:
                with torch.no_grad():
                    random_s_action = torch.floor(torch.rand(x.size(0), 1).cuda().squeeze() * part_count).int()
                    random_t_action = torch.rand(x.size(0), 1).cuda()
                    random_patches_list, random_partid_list = get_crop_pmodel(x, random_s_action, random_t_action, graph.part_node,focus_size)
                    random_patches = torch.cat(random_patches_list, 0)
                    random_output, random_state = model_list[focus_size](
                        random_patches.clone())
                    random_output = fc.module.confirm_forward(random_output, restart_batch=False)

                    patches = torch.cat(patches_list, 0)
                    output, state = model_list[focus_size](
                        patches.clone())
                    output = fc(output, restart_batch=False)
            # Loss computation and backpropagation
            if args.train_stage != 2:
                loss = criterion(output, target_var)
                loss_cla.append(loss)
                losses[patch_step].update(loss.data.item(), x.size(0))

                loss_dsn = criterion(dsn_output, target_var)
                loss_list_dsn.append(loss_dsn)
            else:
                confidence = torch.gather(F.softmax(output.detach(), 1), dim=1, index=target_var.view(-1, 1)).view(1,
                                                                                                                   -1)
                random_confidence = torch.gather(F.softmax(random_output.detach(), 1), dim=1,
                                                 index=target_var.view(-1, 1)).view(1, -1)
                reward = confidence - random_confidence
                reward_list[patch_step - 1].update(reward.data.mean(), x.size(0))
                memory.rewards.append(reward)

        if args.train_stage != 2:
            loss = (sum(loss_cla) + 0.7 * sum(loss_list_dsn)) / args.T
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            ppo.update(memory)
        memory.clear_memory()
        batch_time.update(time.time() - end)
        end = time.time()
        prog_bar.update()

def stgcn_val(model, pmodel_list, fc, dsn_fc, memory, ppo, val_loader, criterion, graph, record_file, epoch):
    batch_time = AverageMeter()
    losses = [AverageMeter() for _ in range(args.T)]
    top1 = [AverageMeter() for _ in range(args.T)]
    dsn_top1 = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]

    model.eval()
    for m in pmodel_list:
        m.eval()
    fc.eval()

    prog_bar = mmcv.ProgressBar(len(val_loader))

    fd = open(record_file, 'a+')
    with torch.no_grad():
        for i,data in enumerate(val_loader):
            assert data['keypoint'].shape[1] == 1
            input = data['keypoint'][:, 0].cuda()
            target_var = data['label'].cuda()
            batch_size = len(input)
            loss_cla = []

            x = input
            output, state = model(x)

            dsn_output = dsn_fc.module[0](output)

            output = fc(output, restart_batch=True)

            loss_prime = criterion(output, target_var)
            loss_cla.append(loss_prime)

            dsn_acc = accuracy(dsn_output, target_var, topk=(1,))
            dsn_top1[0].update(dsn_acc.sum(0).mul_(100.0 / batch_size).data.item(), input.size(0))

            losses[0].update(loss_prime.data.item(), input.size(0))
            acc = accuracy(output, target_var, topk=(1,))
            top1[0].update(acc.sum(0).mul_(100.0 / batch_size).data.item(), input.size(0))
            part_count = len(graph.part_node[0])
            for patch_step in range(1, args.T):
                focus_size = 0
                if args.train_stage == 1:

                    s_action = torch.floor(torch.rand(input.size(0), 1).cuda().squeeze() * part_count).int()
                    t_action = torch.rand(input.size(0), 1).cuda()

                else:
                    if patch_step == 1:
                        s_action, t_action = ppo.select_action(state.to(0), memory, restart_batch=True, training=False)

                    else:
                        s_action, t_action = ppo.select_action(state.to(0), memory, training=False)
                if i == 1:
                    print(s_action)
                patches_list, partid_list = get_crop_pmodel(input, s_action, t_action, graph.part_node, focus_size)


                patches = torch.cat(patches_list, 0)

                with torch.no_grad():
                    random_s_action = torch.floor(torch.rand(input.size(0), 1).cuda().squeeze() * part_count).int()
                    random_t_action = torch.rand(input.size(0), 1).cuda()
                    random_patches_list, random_partid_list = get_crop_pmodel(x, random_s_action,
                                                                              random_t_action,
                                                                              graph.part_node, focus_size)
                    random_patches = torch.cat(random_patches_list, 0)
                    random_output, random_state = pmodel_list[focus_size](
                        random_patches.clone())
                    random_output = fc.module.confirm_forward(random_output, restart_batch=False)
                if args.train_stage != 2:
                    output, state = pmodel_list[focus_size](
                        patches.clone())
                    dsn_output = dsn_fc.module[focus_size + 1](output)
                    output = fc(output, restart_batch=False)

                else:
                    with torch.no_grad():
                        output, state = pmodel_list[focus_size](
                            patches.clone())
                        dsn_output = dsn_fc.module[focus_size + 1](output)
                        output = fc(output, restart_batch=False)

                acc = accuracy(output, target_var, topk=(1,))
                top1[patch_step].update(acc.sum(0).mul_(100.0 / batch_size).data.item(), input.size(0))

                dsn_acc = accuracy(dsn_output, target_var, topk=(1,))
                dsn_top1[patch_step].update(dsn_acc.sum(0).mul_(100.0 / batch_size).data.item(), input.size(0))

                confidence = torch.gather(F.softmax(output.detach(), 1), dim=1, index=target_var.view(-1, 1)).view(1,
                                                                                                                   -1)
                random_confidence = torch.gather(F.softmax(random_output.detach(), 1), dim=1,
                                                 index=target_var.view(-1, 1)).view(1,
                                                                                    -1)
                reward = confidence - random_confidence

                reward_list[patch_step - 1].update(reward.data.mean(), input.size(0))
                memory.rewards.append(reward)

            memory.clear_memory()
            prog_bar.update()
        print('\n')
        save_string = ('Epoch: [{0}]\t'
                       'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                       'Loss {loss.value:.4f} ({loss.ave:.4f})\t'.format(
            epoch, batch_time=batch_time, loss=losses[-1]))
        print(save_string)
        fd.write(save_string + '\n')

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
    result_list = [d.ave for d in top1]
    dsn_result_list = [d.ave for d in dsn_top1]
    for i, r in enumerate(dsn_result_list):
        print(" dsn_step {} :ACC = {}\n".format(i, r))
    return result_list

def stgcn_main():
    graph = Graph(**cfg.model.backbone.graph_cfg)
    record_path = cfg.work_dir + '/GF-' + str(args.model_arch) +"-k400"
    if args.train_stage == 2:
        record_path = record_path + '-stage2'
    elif args.train_stage == 3:
        record_path = record_path + '-stage3'

    if not os.path.exists(record_path):
        mkdir_p(record_path)
    record_file = record_path + '/record.txt'


    pmodel_list = nn.ModuleList()
    model = build_from_cfg(cfg.model.backbone, BACKBONES)

    cfg.model.pbackbone.focus_size = 0
    pmodel = build_from_cfg(cfg.model.pbackbone, BACKBONES)
    pmodel_list.append(pmodel)

    fc = build_from_cfg(cfg.model.cls_head, HEADS)
    dsn_fc = nn.ModuleList()
    dsn_fc.append(build_from_cfg(cfg.model.prime_cls_head, HEADS))
    dsn_fc.append(build_from_cfg(cfg.model.p1_cls_head, HEADS))


    if args.train_stage != 2:

        optimizer = torch.optim.SGD(
            [{'params': model.parameters()}, {'params': fc.parameters()}, ] + [{'params': m.parameters()} for m in
                                                                               pmodel_list] + [
                {'params': dsn_fc.parameters()}],
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            nesterov=cfg.optimizer.nesterov,
            weight_decay=cfg.optimizer.weight_decay)
        training_epoch_num = cfg.total_epochs
    else:
        optimizer = None
        training_epoch_num = 15

    criterion = nn.CrossEntropyLoss().cuda()

    model = nn.parallel.DataParallel(model.cuda(), device_ids=[args.local_rank],
                                     output_device=args.local_rank,
                                     )
    for i in range(len(pmodel_list)):
        pmodel_list[i] = nn.parallel.DataParallel(pmodel_list[i].cuda(), device_ids=[args.local_rank],
                                                  output_device=args.local_rank,
                                                  )
    fc = nn.parallel.DataParallel(fc.cuda(), device_ids=[args.local_rank],
                                  output_device=args.local_rank,
                                  )
    dsn_fc = nn.parallel.DataParallel(dsn_fc.cuda(), device_ids=[args.local_rank],
                                      output_device=args.local_rank,
                                      )
    batch_size = cfg.data.videos_per_gpu

    train_set = RepeatDataset(**cfg.data.train)
    test_set = PoseDataset(**cfg.data.test, test_mode=True)
    val_set = PoseDataset(**cfg.data.val, test_mode=True)

    rank, world_size = get_dist_info()

    train_sampler = DistributedSampler(
        train_set, world_size, rank, shuffle=True, seed=None)
    test_sampler = DistributedSampler(
        test_set, world_size, rank, shuffle=False, seed=None)
    val_sampler = DistributedSampler(
        val_set, world_size, rank, shuffle=False, seed=None)

    init_fn = partial(
        worker_init_fn, num_workers=2, rank=rank,
        seed=0)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        collate_fn=partial(collate, samples_per_gpu=8),
        pin_memory=True,
        shuffle=False,
        worker_init_fn=init_fn,
        drop_last=False,
        prefetch_factor=4,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=16,
        collate_fn=partial(collate, samples_per_gpu=8),
        pin_memory=True,
        shuffle=False,
        worker_init_fn=init_fn,
        drop_last=False,
        prefetch_factor=4,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        sampler=test_sampler,
        num_workers=16,
        collate_fn=partial(collate, samples_per_gpu=8),
        pin_memory=True,
        shuffle=False,
        worker_init_fn=init_fn,
        persistent_workers=False,
        drop_last=False,
        prefetch_factor=4,
    )

    if args.train_stage != 1:
        state_dim = cfg.model.cls_head.in_channels
        hidden_dim = cfg.model.cls_head.in_channels * 2
        ppo = PPO(cfg.model.cls_head.in_channels, state_dim,
                  hidden_dim,graph=graph)
        resume_ckp = torch.load(args.last_stage_checkpoint)
        s_epoch = resume_ckp['epoch']
        print('resume from epoch: {}'.format(s_epoch))
        model.load_state_dict(resume_ckp['model_state_dict'])
        pmodel_list.load_state_dict(resume_ckp['pmodel_1_state_dict'])
        dsn_fc.module.load_state_dict(resume_ckp['dsn_fc_state_dict'])
        fc.load_state_dict(resume_ckp['fc'])
        if args.train_stage == 3:
            ppo.policy.load_state_dict(resume_ckp['policy'])
            ppo.policy_old.load_state_dict(resume_ckp['policy'])
    else:
        ppo = None
    memory = Memory()

    print('Training Stage: {}'.format(args.train_stage))
    if args.GE_model_path:
        ge_resume_ckp = torch.load(args.GE_model_path)
        model.load_state_dict(ge_resume_ckp['model_state_dict'])
        start_epoch = 0
        best_acc = 0
    if args.resume:
        resume_ckp = torch.load(args.resume)
        start_epoch = resume_ckp['epoch']
        print('resume from epoch: {}'.format(start_epoch))
        model.load_state_dict(resume_ckp['model_state_dict'])
        pmodel_list.load_state_dict(resume_ckp['pmodel_1_state_dict'])
        dsn_fc.module.load_state_dict(resume_ckp['dsn_fc_state_dict'])
        fc.load_state_dict(resume_ckp['fc'])
        if optimizer:
            optimizer.load_state_dict(resume_ckp['optimizer'])
        if ppo:
            ppo.policy.load_state_dict(resume_ckp['policy'])
            ppo.policy_old.load_state_dict(resume_ckp['policy'])
            ppo.optimizer.load_state_dict(resume_ckp['ppo_optimizer'])
        best_acc = resume_ckp['best_acc']
    else:
        start_epoch = 0
        best_acc = 0

    for epoch in range(start_epoch, training_epoch_num):
        train_sampler.set_epoch(epoch)

        stgcn_train(model, pmodel_list, fc, dsn_fc, memory, ppo, optimizer, train_loader, criterion, epoch, graph, args)
        result_list = stgcn_val(model, pmodel_list, fc, dsn_fc, memory, ppo, val_loader, criterion, graph, record_file,
                                epoch + 1)

        for i, r in enumerate(result_list):
            print(" step {} :ACC = {}\n".format(i, r))

        if result_list[args.T - 1] > best_acc:
            best_acc = result_list[args.T - 1]
            is_best = True
        else:
            is_best = False

        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'pmodel_1_state_dict': pmodel_list.state_dict(),
            'dsn_fc_state_dict': dsn_fc.module.state_dict(),
            'fc': fc.state_dict(),
            'result_list': result_list,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict() if optimizer else None,
            'ppo_optimizer': ppo.optimizer.state_dict() if ppo else None,
            'policy': ppo.policy.state_dict() if ppo else None,
        }, result_list[args.T - 1], is_best, epoch + 1, checkpoint=record_path)
    stgcn_test(model, pmodel_list, fc, memory, ppo, test_loader, criterion, graph, record_path, args)


def adjust_lr(optimizer, base_lr, target_lr, current_iter, max_iter, weight=1):
    factor = current_iter / max_iter
    cos_out = cos(pi * factor) + 1
    cur_lr = target_lr + 0.5 * weight * (base_lr - target_lr) * cos_out
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

def adjust_weight(base_weight, target_weight, current_iter, max_iter, weight=1):
    factor = current_iter / max_iter
    cos_out = cos(pi * factor) + 1
    cur_weight = target_weight + 0.5 * weight * (base_weight - target_weight) * cos_out
    return cur_weight

def worker_init_fn(worker_id, num_workers, rank, seed):
    """Init the random seed for various workers."""
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    stgcn_main()
