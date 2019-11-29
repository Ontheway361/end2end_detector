#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import os
import sys
import time
import torch
import numpy as np
import torchvision
from sklearn import metrics
from torch.utils.data import DataLoader

from config import training_args
from classifier_lib import MobileNetV2, AkuDataset

from IPython import embed


class OccDetetor(object):

    def __init__(self, args):

        self.args   = args
        self.model  = dict()
        self.data   = dict()
        self.softmax= torch.nn.Softmax(dim=1)
        self.device = args.use_gpu and torch.cuda.is_available()


    def _report_settings(self):
        ''' Report the settings '''

        str = '-' * 16
        print('%sEnvironment Versions%s' % (str, str))
        print("- Python: {}".format(sys.version.strip().split('|')[0]))
        print("- PyTorch: {}".format(torch.__version__))
        print("- TorchVison: {}".format(torchvision.__version__))
        print("- device: {}".format(self.device))
        print('-'*52)


    def _model_loader(self):

        if self.device:
            self.model['backbone'] = MobileNetV2(num_classes=2).cuda()
        else:
            self.model['backbone'] = MobileNetV2(num_classes=2)
        # self.model['criterion'] = torch.nn.BCELoss()
        self.model['criterion'] = torch.nn.CrossEntropyLoss()
        self.model['optimizer'] = torch.optim.Adam(
                                      [{'params': self.model['backbone'].parameters()}],
                                      lr=self.args.base_lr,
                                      weight_decay=self.args.weight_decay)
        self.model['scheduler'] = torch.optim.lr_scheduler.MultiStepLR(
                                      self.model['optimizer'], milestones=[25, 40], gamma=self.args.gamma)

        if self.device and len(self.args.gpu_ids) > 1:
            self.model['backbone'] = torch.nn.DataParallel(self.model['backbone'], device_ids=self.args.gpu_ids)
            torch.backends.cudnn.benchmark = True
            print('Parallel mode was going ...')
        elif self.device:
            print('Single-gpu mode was going ...')
        else:
            print('CPU mode was going ...')

        if len(self.args.resume) > 2:
            checkpoint = torch.load(self.args.resume, map_location=lambda storage, loc: storage)
            self.args.start_epoch = checkpoint['epoch']
            self.model['backbone'].load_state_dict(checkpoint['backbone'])
            print('Resuming the train process at %3d epoches ...' % self.args.start_epoch)
        print('Model loading was finished ...')


    def _data_loader(self):

        self.data['train_loader'] = DataLoader(AkuDataset(self.args.train_file, self.args.in_size),
                                        batch_size=self.args.batchsize,
                                        shuffle=True,
                                        num_workers=self.args.workers,
                                        drop_last=False)
        self.data['eval_loader']  = DataLoader(AkuDataset(self.args.eval_file, self.args.in_size),
                                        batch_size=self.args.batchsize,
                                        shuffle=False,
                                        num_workers=self.args.workers)
        print('Data loading was finished ...')


    def _calculate_acc(self, gt_label, pred_score):
        
        pred_score = self.softmax(torch.from_numpy(np.array(pred_score)))
        pred_label = pred_score.argmax(dim=1).numpy().tolist()
        auc        = metrics.roc_auc_score(gt_label, pred_label)
        acc        = metrics.accuracy_score(gt_label, pred_label)
        recall     = metrics.recall_score(gt_label, pred_label)
        f1_score   = metrics.f1_score(gt_label, pred_label)
        precision  = metrics.precision_score(gt_label, pred_label)
        print(metrics.classification_report(gt_label, pred_label, digits=4))
        print(metrics.confusion_matrix(gt_label, pred_label))
        return precision, recall


    def _model_train(self, epoch = 0):

        self.model['backbone'].train()

        loss_recorder = []
        gt_label_list, pred_label_list = [], []
        for idx, (img, gt_label) in enumerate(self.data['train_loader']):

            img.requires_grad      = False
            gt_label.requires_grad = False

            if self.device:
                img      = img.cuda()
                gt_label = gt_label.cuda()
            score = self.model['backbone'](img)
            loss = self.model['criterion'](score, gt_label)
            self.model['optimizer'].zero_grad()
            loss.backward()
            self.model['optimizer'].step()
            loss_recorder.append(loss.item())
            gt_label_list.extend(gt_label.cpu().detach().numpy().tolist())
            pred_label_list.extend(score.cpu().detach().numpy().tolist())
            if (idx + 1) % self.args.print_freq == 0:
                ave_loss = np.mean(loss_recorder)
                print('cur_epoch : %3d|%2d|%2d, loss : %.4f' % \
                      (idx+1, epoch, self.args.end_epoch, ave_loss))
        self._calculate_acc(gt_label_list, pred_label_list)
        return ave_loss


    def _model_eval(self):

        self.model['backbone'].eval()
        losses = []
        with torch.no_grad():
            gt_label_list, pred_label_list = [], []
            for idx, (img, gt_label) in enumerate(self.data['eval_loader']):

                img.requires_grad      = False
                gt_label.requires_grad = False

                if self.device:
                    img      = img.cuda()
                    gt_label = gt_label.cuda()
                score = self.model['backbone'](img)
                loss = self.model['criterion'](score, gt_label)
                losses.append(loss.item())
                gt_label_list.extend(gt_label.cpu().detach().numpy().tolist())
                pred_label_list.extend(score.cpu().detach().numpy().tolist())
            eval_loss = np.mean(losses)
            self._calculate_acc(gt_label_list, pred_label_list)
            print('eval_loss : %.4f' % eval_loss)
        return eval_loss


    def _main_loop(self):

        min_loss = 1e6
        for epoch in range(self.args.start_epoch, self.args.end_epoch + 1):
            
            start_time = time.time()
            train_loss = self._model_train(epoch)
            end_time   = time.time()
            print('Single epoch cost time : %.2f mins' % ((end_time - start_time)/60))
            self.model['scheduler'].step()
            val_loss   = self._model_eval()
            if val_loss < min_loss:
                min_loss = val_loss
                filename = os.path.join(self.args.snapshot, 'sota.pth.tar')
                torch.save({
                    'epoch'   : epoch,
                    'backbone': self.model['backbone'].state_dict(),
                    'loss'    : val_loss
                }, filename)

            if (epoch + 1) % self.args.save_freq == 0:
                filename = os.path.join(self.args.snapshot, 'checkpoint_epoch_'+str(epoch)+'.pth.tar')
                torch.save({
                    'epoch'   : epoch,
                    'backbone': self.model['backbone'].state_dict(),
                    'loss'    : val_loss
                }, filename)


    def train_runner(self):

        self._report_settings()

        self._model_loader()

        self._data_loader()

        self._main_loop()


if __name__ == "__main__":

    occdetector = OccDetetor(training_args())
    occdetector.train_runner()
