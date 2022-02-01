"""Run testing given a trained model."""

import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
import torchvision

from dataset import dataset_fuse
from models import team_net, transforms

import os
from tqdm import tqdm, trange
import pdb

parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--data-name', type=str)
parser.add_argument('--representation', type=str, choices=['iframe', 'residual', 'mv'])
parser.add_argument('--no-accumulation', action='store_true',
                    help='disable accumulation of motion vectors and residuals.')
parser.add_argument('--data-root', type=str)
parser.add_argument('--test-list', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--arch', type=str)
parser.add_argument('--save-scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--test-crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--num_clips', type=int, default=10)
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of workers for data loader.')
parser.add_argument('--gpus', type=str, default='2',
                    help='gpu ids.')
parser.add_argument('--is_shift', action='store_true',
                    help='enable TSM')

args = parser.parse_args()


if args.data_name == 'ucf101':
    num_class = 101
elif args.data_name == 'hmdb51':
    num_class = 51
elif args.data_name == 'kinetic400':
    num_class = 400


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


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpus
    net = team_net.TEAM_Net(num_class, args.test_segments,
                                base_model=args.arch, is_shift=args.is_shift)

    checkpoint = torch.load(args.weights)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    net.load_state_dict(checkpoint['state_dict'])

    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            transforms.GroupScale(256),
            transforms.GroupCenterCrop(256),
        ])
    elif args.test_crops == 3:
        cropping = torchvision.transforms.Compose([
            transforms.GroupFullResSample(256, 256, flip = False)
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            transforms.GroupOverSample(256, 256, is_mv = True)
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported, but got {}.".format(args.test_crops))


    data_loader = torch.utils.data.DataLoader(
        dataset_fuse.CoviarDataSet_inference(
            args.data_root,
            args.data_name,
            video_list=args.test_list,
            num_segments=args.test_segments,
            frame_transform=cropping,
            motion_transform=cropping,
            residual_transform=cropping,
            accumulate=(not args.no_accumulation),
            test_crops = args.test_crops,
            num_clips = args.num_clips),
        batch_size=1, shuffle=False, pin_memory=False,
        num_workers=args.workers)





    net = torch.nn.DataParallel(net.cuda())
    net.eval()


    total_num = len(data_loader.dataset)
    output = []

    def forward_video(data):
        with torch.no_grad():
            # input size: nb, clips*crops, t, c, h, w
            input_i, input_mv, input_r = data[0].cuda(non_blocking=True), \
                                         data[1].cuda(non_blocking=True), \
                                         data[2].cuda(non_blocking=True) 
            input_i = input_i.view((-1,)+input_i.size()[2:])
            input_mv = input_mv.view((-1,)+input_mv.size()[2:])
            input_r = input_r.view((-1,)+input_r.size()[2:])
            scores = net(input_i, input_mv, input_r)
            scores = scores.mean(0, keepdims=True)
        return scores.data.cpu().numpy().copy()



    top1 = AverageMeter()
    cnt_time_avg = AverageMeter()
    for i, (data, label) in enumerate(tqdm(data_loader)):
        proc_start_time = time.time()
        video_scores = forward_video(data)
        output.append((video_scores, label[0]))
        cnt_time = time.time() - proc_start_time
        video_pred = np.argmax(video_scores)
        acc = float(video_pred == label) * 100.0
        top1.update(acc, 1)
        cnt_time_avg.update(cnt_time, 1)
        if (i + 1) % 100 == 0:
            print('video {} done, total {}/{}, average {} sec/video, top-1 {:.02f}%'.format(i, i+1,
                                                                                      total_num,
                                                                                    #   float(cnt_time) / (i+1),
                                                                                      cnt_time_avg.avg,
                                                                                      top1.avg
                                                                                      ))
    video_pred = [np.argmax(x[0]) for x in output]
    video_labels = [x[1] for x in output]

    print('Accuracy {:.02f}% ({})'.format(
        float(np.sum(np.array(video_pred) == np.array(video_labels))) / len(video_pred) * 100.0,
        len(video_pred)))


    if args.save_scores is not None:

        name_list = [x.strip().split()[0] for x in open(args.test_list)]
        order_dict = {e:i for i, e in enumerate(sorted(name_list))}

        reorder_output = [None] * len(output)
        reorder_label = [None] * len(output)
        reorder_name = [None] * len(output)

        for i in range(len(output)):
            idx = order_dict[name_list[i]]
            reorder_output[idx] = output[i]
            reorder_label[idx] = video_labels[i]
            reorder_name[idx] = name_list[i]

        np.savez(args.save_scores, scores=reorder_output, labels=reorder_label, names=reorder_name)


if __name__ == '__main__':
    main()
