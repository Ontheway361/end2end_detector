#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn

from IPython import embed

class AnomalyLoss(nn.Module):

    def __init__(self, weights = [1.0, 1.0], regular = 1.0, use_gpu = True):
        
        super(AnomalyLoss, self).__init__()
        
        weights  = torch.tensor(weights, dtype=torch.float32)
        if use_gpu:
            weights = weights.cuda()
        self.reweiCEL = nn.CrossEntropyLoss(weight=weights)
        self.CrossEnt = nn.CrossEntropyLoss()
        self.regular  = regular

        
    def forward(self, pred_score, gt_label):
        
        occ_loss = self.reweiCEL(pred_score[:, :2], gt_label[0])
        hos_loss = self.CrossEnt(pred_score[:, 2:], gt_label[1])
        ano_loss = self.regular * occ_loss +  hos_loss
        return ano_loss
