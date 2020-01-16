#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import os.path as osp


data_dir = '/home/jovyan/lujie/end2end/anno_file'   # gpu


def training_args():
    parser = argparse.ArgumentParser(description='Trainging Anomaly Detector')

    # -- env
    parser.add_argument('--use_gpu',  type=bool, default=True)
    parser.add_argument('--gpu_ids',  type=list, default=[0, 1])
    parser.add_argument('--workers',  type=int,  default=0)

    # -- dataset
    parser.add_argument('--train_file', type=str, default=osp.join(data_dir, 'multitask_1530_1203_train.txt'))
    parser.add_argument('--eval_file',  type=str, default=osp.join(data_dir, 'multitask_1530_1203_test.txt'))
    parser.add_argument('--in_size',    type=int, default=112)   # default=224
    parser.add_argument('--batchsize',  type=int, default=128)   # default=256
    
    # -- model
    parser.add_argument('--backbone',   type=str,   default='resnet101')
    parser.add_argument('--num_classes',type=int,   default=6)
    parser.add_argument('--regular',    type=float, default=1.0)
    parser.add_argument('--weights',    type=list,  default=[1.1, 1.0])
    
    #  -- optimizer
    parser.add_argument('--base_lr',     type=float, default=1e-3)
    parser.add_argument('--lr_steps',    type=list,  default=[18, 25])
    parser.add_argument('--gamma',       type=float, default=0.5)
    parser.add_argument('--weight_decay',type=float, default=5e-4)

    # -- epoch
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--end_epoch',   type=int, default=30)
    parser.add_argument('--print_freq',  type=int, default=200)

    # -- snapshot
    parser.add_argument('--save_freq',type=int, default=5)
    parser.add_argument('--resume',   type=str, default='') # ../checkpoint/resnet101_aug/sota.pth.tar
    parser.add_argument('--save_to',  type=str, default='../checkpoint/resnet101_flip_aug')


    args = parser.parse_args()
    return args
