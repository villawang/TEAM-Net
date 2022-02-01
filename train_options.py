"""Training options."""

import argparse

parser = argparse.ArgumentParser(description="CoViAR")

# Data.
parser.add_argument('--data-name', type=str,
                    help='dataset name.') #choices=['ucf101', 'hmdb51']
parser.add_argument('--data-root', type=str,
                    help='root of data directory.')
parser.add_argument('--train-list', type=str,
                    help='training example list.')
parser.add_argument('--test-list', type=str,
                    help='testing example list.')

# Model.
parser.add_argument('--arch', type=str, default="resnet152",
                    help='base architecture.')
parser.add_argument('--num_segments', type=int, default=3,
                    help='number of TSN segments.')
parser.add_argument('--no-accumulation', action='store_true',
                    help='disable accumulation of motion vectors and residuals.')
parser.add_argument('--dropout', '--dropout', default=0.5, type=float,
                    help='dropout.')
parser.add_argument('--is_shift', action='store_true',
                    help='enable TSM')


# Training.
parser.add_argument('--is_train', action='store_true',
                    help='training flag.')
parser.add_argument('--epochs', default=500, type=int,
                    help='number of training epochs.')
parser.add_argument('--batch-size', default=40, type=int,
                    help='batch size.')
parser.add_argument('--lr', default=0.001, type=float,
                    help='base learning rate.')
parser.add_argument('--lr-steps', default=[200, 300, 400], type=float, nargs="+",
                    help='epochs to decay learning rate.')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    help='lr decay factor.')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay.')
parser.add_argument('--clip_gradient', '--gd', type=int, default=20, help='gradient clip')
parser.add_argument("--local_rank", type=int,
                    help='local rank for DistributedDataParallel')

# Log.
parser.add_argument('--eval-freq', default=5, type=int,
                    help='evaluation frequency (epochs).')
parser.add_argument('--workers', default=20, type=int,
                    help='number of data loader workers.')
parser.add_argument('--model-prefix', type=str, default="model",
                    help="prefix of model name.")
parser.add_argument('--gpus', type=str, default='2',
                    help='gpu ids.')

