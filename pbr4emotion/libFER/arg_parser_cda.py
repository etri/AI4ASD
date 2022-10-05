""" 
   * Source: libFER.arg_parser_cda.py
   * License: PBR License (Dual License)
   * Modified by ByungOk Han <byungok.han@etri.re.kr>
   * Date: 13 Mar 2021, ETRI
   * Copyright 2022. ETRI all rights reserved. 

"""

import argparse


def argument_parser_cda():

    """argument_parser_cda function

    Note: argument_parser_cda function

    """

    parser = argparse.ArgumentParser(description = 'Pytorch Cross-dataset Adaptation Training')

    parser.add_argument('--cuda', '-c', default=True)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (usefult on restarts)')
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '-wd', default=2e-4, type=float,
                        metavar='W', help='weight decay (default: 2e-4)')
    parser.add_argument('--print_freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--model', default='', type=str, metavar='Model',
                        help='model type: resnet')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--root_path', default=None, type=str, metavar='PATH',
                        help='path to root path of images (default: None)')
    parser.add_argument('--train_list', default=None, type=str, metavar='PATH',
                        help='path to training list (default: None)')
    parser.add_argument('--val_list', default=None, type=str, metavar='PATH',
                        help='path to validation list (default: None)')
    parser.add_argument('--save_path', default=None, type=str, metavar='PATH',
                        help='path to save checkpoint (default: None)')
    parser.add_argument('--num_classes', default=7, type=int,
                        metavar='N', help='number of emotion classes (default: 7)')
    parser.add_argument('--num_ds_classes', default=3, type=int,
                        metavar='N', help='number of dataset classes (default: 3)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help = 'number of train epochs (default: 200)')
    parser.add_argument('--resnet_version', default=2, type=int, metavar='N',
                        help = 'version of ResNet (default: 2)')
    parser.add_argument('--optim', default='sgd', type=str, metavar='OPTIM',
                        help='optimizer type: sgd, adam  (default: sgd)')
    parser.add_argument('--emo_loss', default='cce', type=str, metavar='LOSS',
                        help='loss type: cce, ace (default: cce)')
    parser.add_argument('--ds_loss', default='cce', type=str, metavar='LOSS',
                        help='loss type: cce, ace (default: cce)')
    parser.add_argument('--ace_rate', default=[0.1, 0.5, 0.9, 1.0], nargs='+', type=float, metavar='R',
                        help='ace rate for adaptive cross entropy loss')
    parser.add_argument('--ds_alpha', default=[0.1, 0.5, 0.9, 1.0], nargs='+', type=float,
                        metavar='R', help='alpha value of the GRL in dataset classifier')
    parser.add_argument('--classifier_type', default='conv', type=str, metavar='TYPE',
                        help='classifier type for emotion and dataset classifiers: conv, 1-fc, 3-fc (default: conv)')
    parser.add_argument('--learning_rate', default=[0.1, 0.01, 0.001, 0.0001], nargs='+', type=float,
                        metavar='LR', help='learning rate (default: 0.1, 0.01, 0.001, 0.0001)')
    parser.add_argument('--lr_boundary', default=[50, 100, 150], nargs='+', type=int,
                        metavar='N', help='change boundary for lr')
    parser.add_argument('--boundary', default=[50, 100, 150], nargs='+', type=int,
                        metavar='N', help='change boundary for ds_alpha, ace_rate')

    #args = parser.parse_args()
    args, _ = parser.parse_known_args()

    return args
