#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import cv2
import sys
import numpy as np
import torchvision
# from .transforms import *
from torch.utils import data

from IPython import embed

def aku_trans():

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform


class AkuDataset(data.Dataset):

    def __init__(self, in_file, in_size = 224, is_aug = False):

        self.transforms = aku_trans()
        self.in_size    = in_size
        self.is_aug     = is_aug
        self.max_cnt    = 10          # TODO
        
        with open(in_file, 'r') as f:
            self.lines  = f.readlines()
        f.close()
        self._data_aug()
    
    
    def _data_aug(self):
        
        lines = []
        if self.is_aug:
            for row in self.lines:
                if ('1 0 1\n' in row) or ('0 0 0\n' in row):
                    aug = row[:-2] + 2 * row[-2] + '\n'
                    lines.append(aug)
        self.lines.extend(lines)
        print('After data augmentation, %3d rows added.' % len(lines))
    
    
    def __getitem__(self, index):

        try:
            info = self.lines[index].strip().split(' ')
            img  = cv2.resize(cv2.imread(info[0]), (self.in_size, self.in_size))
            occ, hos, anno  = info[1:]
        except Exception as e:
            print(e)
            cnt_try = 0
            while cnt_try < self.max_cnt:
                try:
                    idx  = np.random.randint(0, len(self.lines))
                    info = self.lines[idx].strip().split(' ')
                    img  = cv2.resize(cv2.imread(info[0]), (self.in_size, self.in_size))
                    occ, hos, anno  = info[1:]
                except Exception as e:
                    print(e)
                    cnt_try += 1
                else:
                    break        
        
        if len(anno) == 2:
            img  = cv2.flip(img, 1)
            anno = anno[0]
            
        if self.transforms is not None:
            img = self.transforms(img)

        return (img, [int(occ), int(hos), int(anno)], info[0])


    def __len__(self):
        return len(self.lines)
