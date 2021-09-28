# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/9/20 15:08

import datetime
import os.path

import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import math
import sys

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma, accuracy

from datasets import get_train_loader, get_val_loader
from losses import DistillationLoss
import utils
from utils import get_hp_dict
from admm import ADMM


@torch.no_grad()
def evaluate(data_loader, model, device, print_freq=100):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Evaluation Result: Acc@1 {top1.global_avg:.3f}%, Acc@5 {top5.global_avg:.3f}%, loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval(model, args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    model.to(device)
    # val_loader = get_data_loader(False, args)
    data_loader_val = get_val_loader(args)
    evaluate(data_loader_val, model, device, args.print_freq)


def train(model, args):
    utils.init_distributed_mode(args)
    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)
    model.to(device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    data_loader_train = get_train_loader(args)
    data_loader_val = get_val_loader(args)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    if args.distributed:
        print('*INFO: Distributed training.')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpus])
        model_without_ddp = model.module
    elif args.parallel:
        print('*INFO: Parallel training.')
        model = torch.nn.DataParallel(model, device_ids=list(args.gpus))
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 256.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=True,
            num_classes=args.num_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    output_dir = Path(args.output_dir)
    timestamp = time.strftime('%m%d-%H%M%S')
    if args.save_model or args.save_log:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    # ADMM initialization
    if args.admm:
        admm = ADMM(model_without_ddp, get_hp_dict(args.model, args.ratio, args.format), args.format,
                    device, verbose=args.verbose, log=args.log)
        file_name = '{}_{}_admm_{}_{}'.format(args.model, args.dataset, args.format, timestamp)
    else:
        admm = None
        file_name = '{}_{}_{}'.format(args.model, args.dataset, timestamp)

    print(f"Start training for {args.epochs} epochs")
    max_acc1 = 0.
    max_acc5 = 0.
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}/{}]'.format(epoch+1, args.epochs)

        if args.admm:
            admm.update()
        for samples, targets in metric_logger.log_every(data_loader_train, args.print_freq, header):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(samples, outputs, targets)
                if args.admm:
                    loss = admm.append_admm_loss(args.rho, loss)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                        parameters=model.parameters(), create_graph=is_second_order)

            torch.cuda.synchronize()
            if model_ema is not None:
                model_ema.update(model)

            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        lr_scheduler.step(epoch)
        if args.save_model:
            checkpoint_path = os.path.join(output_dir, file_name + '.pth')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'model_ema': get_state_dict(model_ema),
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device)
        max_acc1 = max(max_acc1, test_stats["acc1"])
        max_acc5 = max(max_acc5, test_stats["acc5"])
        print('Max acc1: {:.3f}%, max acc5: {:.3f}%'.format(max_acc1, max_acc5))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch+1,
                     'n_parameters': n_parameters}

        if args.save_log and utils.is_main_process():
            log_path = os.path.join(output_dir, file_name + '.log')
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_stats) + "\n")

        elapsed_time = time.time() - start_time
        remaining_time = (args.epochs - epoch - 1) * elapsed_time
        print('| Elapsed time : {}'.format(str(datetime.timedelta(seconds=int(elapsed_time)))))
        print('| Remaining time : {}'.format(str(datetime.timedelta(seconds=int(remaining_time)))))
