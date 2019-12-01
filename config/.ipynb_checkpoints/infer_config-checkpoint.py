#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import os.path as osp


# data_dir = '/home/jovyan/gpu3-data2/lujie/imgs_occ/filter_data_1113'   # gpu
save_dir = '/home/jovyan/gpu3-data2/lujie/imgs_occ/akuface_50k'

def infer_args():
    parser = argparse.ArgumentParser(description='Inference of OccDetector')

    # --env
    parser.add_argument('--use_gpu',  type=bool, default=True)
    parser.add_argument('--gpu_ids',  type=list, default=[0,1])
    parser.add_argument('--workers',  type=int,  default=0)

    # --dataset
    parser.add_argument('--eval_file',  type=str, default=osp.join(save_dir, '50k_akuface.txt'))  # akuface_test_1115.txt
    parser.add_argument('--in_size',    type=int, default=112)   # default=224
    parser.add_argument('--batchsize',  type=int, default=128)   # default=256

    # --snapshot
    parser.add_argument('--resume',   type=str, default='../checkpoint/resnet/sota.pth.tar')
    
    # --save
    parser.add_argument('--save_dir', type=str, default=save_dir)


    args = parser.parse_args()
    return args
