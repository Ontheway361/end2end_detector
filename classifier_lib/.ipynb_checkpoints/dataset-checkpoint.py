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

    def __init__(self, file_list, in_size = 224):

        self.transforms = aku_trans()
        self.in_size    = in_size
        with open(file_list, 'r') as f:
            self.lines  = f.readlines()

    def __getitem__(self, index):

        try:
            img_info   = self.lines[index].strip().split()
            self.img   = cv2.resize(cv2.imread(img_info[0]), (self.in_size, self.in_size))
        except Exception as e:
            print(e)
            idx = np.random.randint(0, len(self.lines))
            img_info = self.lines[idx].strip().split()
            self.img   = cv2.resize(cv2.imread(img_info[0]), (self.in_size, self.in_size))
        
        self.label = int(img_info[1])
        self.img   = self.transforms(self.img)
            
            
        return (self.img, self.label, img_info[0])


    def __len__(self):
        return len(self.lines)
