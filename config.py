#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import os.path as osp


data_dir = '/home/jovyan/gpu3-data2/lujie/imgs_occ/filter_data_1113'   # gpu


def training_args():
    parser = argparse.ArgumentParser(description='Trainging Practical Facial Landmark Detector')

    # env
    parser.add_argument('--use_gpu',  type=bool, default=True)
    parser.add_argument('--gpu_ids',  type=list, default=[0,1])
    parser.add_argument('--workers',  type=int,  default=0)

    # --dataset
    parser.add_argument('--train_file', type=str, default=osp.join(data_dir, 'akuface_train_1115.txt'))
    parser.add_argument('--eval_file',  type=str, default=osp.join(data_dir, 'akuface_test_1115.txt'))
    parser.add_argument('--in_size',    type=int, default=224)
    parser.add_argument('--batchsize',  type=int, default=32)   # default=256

    ##  -- optimizer
    parser.add_argument('--base_lr',       type=float, default=1e-3)
    parser.add_argument('--gamma',         type=float, default=0.5)
    parser.add_argument('--weight_decay',  type=float, default=5e-4)

    # -- lr
    parser.add_argument("--lr_patience", type=int, default=40)

    # -- epoch
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--end_epoch',   type=int, default=30)
    parser.add_argument('--print_freq',  type=int, default=100)

    # -- snapshot
    parser.add_argument('--save_freq',type=int, default=5)
    parser.add_argument('--resume',   type=str, default='')
    parser.add_argument('--snapshot', type=str, default='checkpoint/snapshot/')


    args = parser.parse_args()
    return args
