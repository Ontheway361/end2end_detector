#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import os.path as osp


# data_dir = '/home/jovyan/gpu3-data2/lujie/imgs_occ/filter_data_1113'   # gpu
# save_dir = '/home/jovyan/gpu3-data2/lujie/imgs_occ/akuface_50k'
save_dir = '/home/jovyan/lujie/end2end'


def infer_args():
    parser = argparse.ArgumentParser(description='Inference of Anomaly Detector')

    # --env
    parser.add_argument('--use_gpu',  type=bool, default=True)
    parser.add_argument('--gpu_ids',  type=list, default=[0])
    parser.add_argument('--workers',  type=int,  default=0)
    
    # -- model
    parser.add_argument('--backbone',   type=str, default='resnet101')
    parser.add_argument('--num_classes',type=int, default=6)
    parser.add_argument('--weights',     type=list,  default=[1.0, 1.0])
    
    # --dataset
    parser.add_argument('--eval_file',  type=str, default=osp.join(save_dir, 'unknow.txt'))  # aku_face_hstest.txt
    parser.add_argument('--in_size',    type=int, default=112)   # default=224
    parser.add_argument('--batchsize',  type=int, default=128)   # default=256

    # --snapshot
    parser.add_argument('--resume',   type=str, default='../checkpoint/resnet101/sota_0.94_0.94.pth.tar')
    
    # --inference
    parser.add_argument('--out_file', type=str, default=osp.join(save_dir, '1203_select.txt'))
    parser.add_argument('--save_dir', type=str, default=osp.join(save_dir, '1203_4w/maybe_neg'))


    args = parser.parse_args()
    return args
