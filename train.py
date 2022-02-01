import shutil
import time
import numpy as np
import os
import random

import torch
from torch.nn.utils import clip_grad_norm_
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torchvision

from dataset import video_dataset
from models import team_net, transforms
from train_options import parser

from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import pdb

SAVE_FREQ = 40
PRINT_FREQ = 20
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.gpus


local_args = dict()
local_args['recover_from_checkpoint'] = None
local_args['pretrained'] = None
local_args['logdir'] = 'log/{data_name}/{arch}_{num_seg}f'\
                        .format(data_name=args.data_name,
                        arch=args.arch,
                        num_seg=args.num_segments
                        )


def main():
    seed = 1
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    global best_prec1
    global best_prec5
    best_prec1 = 0
    best_prec5 = 0

    print('Training arguments:')
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))

    if args.data_name == 'ucf101':
        num_class = 101
    elif args.data_name == 'hmdb51':
        num_class = 51
    elif args.data_name == 'kinetic400':
        num_class = 400

    model = team_net.TEAM_Net(num_class, args.num_segments,
                                 base_model=args.arch, is_shift=args.is_shift, 
                                 shift_div=8, dropout=args.dropout)

    cudnn.benchmark = True

    if args.is_train:
        # continue training from the last checkpoint
        if local_args['recover_from_checkpoint'] is not None:
            print('loading last trained checkpoint')
            model_checkpoint = torch.load(local_args['recover_from_checkpoint'], map_location='cpu')
            model_dict = model.state_dict()
            for k, v in model_checkpoint['state_dict'].items():
                if 'module' in k:
                    k = k.replace('module.', '')
                    model_dict.update({k: v})
                else:
                    model_dict.update({k: v})
            model.load_state_dict(model_dict)
            print("model epoch {} best prec@1: {}".format(model_checkpoint['epoch'], model_checkpoint['best_prec1']))

        # load pretrained model
        if local_args['pretrained'] is not None:
            print('loading pretrained weights')
            model_checkpoint = torch.load(local_args['pretrained'], map_location='cpu')
            model_dict = model.state_dict()
            for k, v in model_checkpoint['state_dict'].items():
                if 'new_fc' in k:
                    continue
                else:
                    if 'module' in k:
                        k = k.replace('module.', '')
                        model_dict.update({k: v})
                    else:
                        model_dict.update({k: v})
            model.load_state_dict(model_dict)
            print("model epoch {} best prec@1: {}".format(model_checkpoint['epoch'], model_checkpoint['best_prec1']))



        policies = model.get_optim_policies()
        for param_group in policies:
            param_group['lr'] = args.lr * param_group['lr_mult']
            param_group['weight_decay'] = args.weight_decay * param_group['decay_mult']
        optimizer = torch.optim.SGD(policies, momentum=0.9)
        
        criterion = torch.nn.CrossEntropyLoss().cuda()


        train_loader = torch.utils.data.DataLoader(
                video_dataset.CoviarDataSet(
                    args.data_root,
                    args.data_name,
                    video_list=args.train_list,
                    num_segments=args.num_segments,
                    frame_transform=model.I_model.get_augmentation(args.data_name),
                    motion_transform=model.MV_model.get_augmentation(args.data_name),
                    residual_transform=model.R_model.get_augmentation(args.data_name),
                    is_train=True,
                    accumulate=(not args.no_accumulation),
                    dense_sample=False
                    ),
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            video_dataset.CoviarDataSet(
                args.data_root,
                args.data_name,
                video_list=args.test_list,
                num_segments=args.num_segments,
                frame_transform=torchvision.transforms.Compose([
                    transforms.GroupScale(int(model.I_model.scale_size)),
                    transforms.GroupCenterCrop(model.I_model.crop_size),
                    ]),
                motion_transform=torchvision.transforms.Compose([
                    transforms.GroupScale(int(model.MV_model.scale_size)),
                    transforms.GroupCenterCrop(model.MV_model.crop_size),
                    ]),
                residual_transform=torchvision.transforms.Compose([
                    transforms.GroupScale(int(model.R_model.scale_size)),
                    transforms.GroupCenterCrop(model.R_model.crop_size),
                    ]),
                is_train=False,
                accumulate=(not args.no_accumulation),
                dense_sample=False
                ),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)




        # create log file for tensorboard
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        logdir = os.path.join(local_args['logdir'], cur_time)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        writer = SummaryWriter(log_dir=logdir)

        model = torch.nn.DataParallel(model).cuda()
        for epoch in trange(args.epochs):
            cur_lr = adjust_learning_rate(optimizer, epoch, args.lr_steps, args.lr_decay)

            train(train_loader, model, criterion, optimizer, epoch, cur_lr, writer)

            if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
                prec1 = validate(val_loader, model, criterion, epoch, writer)

                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                if is_best or epoch % SAVE_FREQ == 0:
                    save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': model.module.state_dict(),
                            'best_prec1': best_prec1,
                        },
                        is_best,
                        filename='checkpoint.pth.tar')
                print('Best Testing Results: {}'.format(best_prec1))
    else:      
        criterion = torch.nn.CrossEntropyLoss().cuda()
        checkpoint = torch.load(args.teacher_weight)
        print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
        model.load_state_dict(checkpoint['state_dict'])
        val_loader = torch.utils.data.DataLoader(
            dataset_fuse.CoviarDataSet(
                args.data_root,
                args.data_name,
                video_list=args.test_list,
                num_segments=args.num_segments,
                frame_transform=torchvision.transforms.Compose([
                    transforms.GroupScale(int(model.I_model.scale_size)),
                    transforms.GroupCenterCrop(model.I_model.crop_size),
                    ]),
                motion_transform=torchvision.transforms.Compose([
                    transforms.GroupScale(int(model.MV_model.scale_size)),
                    transforms.GroupCenterCrop(model.MV_model.crop_size),
                    ]),
                residual_transform=torchvision.transforms.Compose([
                    transforms.GroupScale(int(model.R_model.scale_size)),
                    transforms.GroupCenterCrop(model.R_model.crop_size),
                    ]),
                is_train=False,
                accumulate=(not args.no_accumulation),
                ),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)  
        model = torch.nn.DataParallel(model).cuda()
        prec1 = validate(val_loader, model, criterion)


def train(train_loader, model, criterion, optimizer, epoch, cur_lr, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        input_i, input_mv, input_r = input[0].cuda(non_blocking=True), \
                                    input[1].cuda(non_blocking=True), \
                                    input[2].cuda(non_blocking=True)


        output = model(input_i, input_mv, input_r)
        loss = criterion(output, target) 

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input_i.size(0))
        top1.update(prec1.item(), input_i.size(0))
        top5.update(prec5.item(), input_i.size(0))

        optimizer.zero_grad()
        loss.backward()
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader),
                       batch_time=batch_time,
                       data_time=data_time,
                       loss=losses,
                       top1=top1,
                       top5=top5,
                       lr=cur_lr)))

    writer.add_scalar('train_loss_epoch', losses.avg, epoch)
    writer.add_scalar('train_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('train_top5_acc_epoch', top5.avg, epoch)


def validate(val_loader, model, criterion, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input_i, input_mv, input_r = input[0].cuda(non_blocking=True), \
                                        input[1].cuda(non_blocking=True), \
                                        input[2].cuda(non_blocking=True)

            output = model(input_i, input_mv, input_r)
            loss = criterion(output, target) 

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input_i.size(0))
            top1.update(prec1.item(), input_i.size(0))
            top5.update(prec5.item(), input_i.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % PRINT_FREQ == 0:
                print(('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5)))

    writer.add_scalar('val_loss_epoch', losses.avg, epoch)
    writer.add_scalar('val_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('val_top5_acc_epoch', top5.avg, epoch)

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1, top5=top5, loss=losses)))
    return top1.avg


def save_checkpoint(state, is_best, filename):
    filename = '_'.join((args.arch, '{}f'.format(args.num_segments), filename))
    folder = os.path.join('checkpoints', args.data_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder , filename)
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.arch, 'tsn_{}f.pth.tar'.format(args.num_segments)))
        best_name = os.path.join(folder, best_name)
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps, lr_decay):
    decay = lr_decay ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    wd = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = wd * param_group['decay_mult']
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
