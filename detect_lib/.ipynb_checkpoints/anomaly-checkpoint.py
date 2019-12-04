#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn

from IPython import embed

class AnomalyLoss(nn.Module):

    def __init__(self, weights = [0.5, 0.5]):
        
        super(AnomalyLoss, self).__init__()
        
        self.CrossEnt = nn.CrossEntropyLoss()
        self.weights  = weights

        
    def forward(self, pred_score, gt_label):
        
        occ_loss = self.CrossEnt(pred_score[:, :2], gt_label[0])
        hos_loss = self.CrossEnt(pred_score[:, 2:], gt_label[1])
        ano_loss = self.weights[0] * occ_loss + self.weights[1] * hos_loss
        return ano_loss
