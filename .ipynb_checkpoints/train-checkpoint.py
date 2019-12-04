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

import detect_lib as lib
from config import training_args

from IPython import embed


class AnomalyDetetor(object):

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
        
        if self.args.backbone == 'mobilenet':
            self.model['backbone'] = lib.MobileNetV2(self.args.num_classes)
        elif self.args.backbone == 'resnet34':
            self.model['backbone'] = lib.resnet34(self.args.num_classes)
        elif self.args.backbone == 'resnet101':
            self.model['backbone'] = lib.resnet101(self.args.num_classes)
        else:
            raise TypeError('unknow backbone, please check out!')
        if self.device:
            self.model['backbone'] = self.model['backbone'].cuda()
           
        # self.model['criterion'] = torch.nn.BCELoss()
        # self.model['criterion'] = torch.nn.CrossEntropyLoss()
        self.model['criterion'] = lib.AnomalyLoss(weights=self.args.weights)
        self.model['optimizer'] = torch.optim.Adam(
                                      [{'params': self.model['backbone'].parameters()}],
                                      lr=self.args.base_lr,
                                      weight_decay=self.args.weight_decay)
        self.model['scheduler'] = torch.optim.lr_scheduler.MultiStepLR(
                                      self.model['optimizer'], milestones=[10, 15], gamma=self.args.gamma)

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

        self.data['train_loader'] = DataLoader(lib.AkuDataset(self.args.train_file, self.args.in_size, True),
                                        batch_size=self.args.batchsize,
                                        shuffle=True,
                                        num_workers=self.args.workers,
                                        drop_last=False)
        self.data['eval_loader']  = DataLoader(lib.AkuDataset(self.args.eval_file, self.args.in_size, False),
                                        batch_size=self.args.batchsize,
                                        shuffle=False,
                                        num_workers=self.args.workers)
        print('Data loading was finished ...')


    def _calculate_acc(self, gt_occ, gt_hos, gt_img, scores, verbose = False):
        
        npy_score = np.array(scores)
        occ_prob  = self.softmax(torch.from_numpy(npy_score[:, :2]))
        hos_prob  = self.softmax(torch.from_numpy(npy_score[:, 2:]))
        
        occ_label = occ_prob.argmax(dim=1).numpy()
        hos_label = hos_prob.argmax(dim=1).numpy()
        pred_info = ((occ_label == 1) & (hos_label == 0)) * 1
        
        # auc        = metrics.roc_auc_score(gt_label, pred_label)
        acc        = metrics.accuracy_score(gt_img, pred_info)
        recall     = metrics.recall_score(gt_img, pred_info)
        f1_score   = metrics.f1_score(gt_img, pred_info)
        precision  = metrics.precision_score(gt_img, pred_info)
        print('acc : %.4f, precision : %.4f, recall : %.4f, f1_score : %.4f' % \
              (acc, precision, recall, f1_score))
        
        if verbose:
            print('%s gt vs. pred %s' % ('-' * 36, '-' * 36))
            print(metrics.classification_report(gt_occ, occ_label, digits=4))
            print(metrics.confusion_matrix(gt_occ, occ_label))
            print('-' * 85)
            print(metrics.classification_report(gt_hos, hos_label, digits=4))
            print(metrics.confusion_matrix(gt_hos, hos_label))
            print('-' * 85)
        return f1_score
        
        
    def _model_train(self, epoch = 0):

        self.model['backbone'].train()

        loss_recorder = []
        gt_occ, gt_hos, gt_img, pred_score = [], [], [], []
        for idx, (img, gt_label, _) in enumerate(self.data['train_loader']):

            img.requires_grad         = False
            gt_label[0].requires_grad = False  # occ_label
            gt_label[1].requires_grad = False  # hos_label
            gt_label[2].requires_grad = False  # img_label
            
            if self.device:
                img = img.cuda()
                gt_label[0] = gt_label[0].cuda()
                gt_label[1] = gt_label[1].cuda()
                gt_label[2] = gt_label[2].cuda()
                
            score = self.model['backbone'](img)
            loss = self.model['criterion'](score, gt_label)
            self.model['optimizer'].zero_grad()
            loss.backward()
            self.model['optimizer'].step()
            loss_recorder.append(loss.item())
            gt_occ.extend(gt_label[0].cpu().detach().numpy().tolist())
            gt_hos.extend(gt_label[1].cpu().detach().numpy().tolist())
            gt_img.extend(gt_label[2].cpu().detach().numpy().tolist())
            pred_score.extend(score.cpu().detach().numpy().tolist())
            if (idx + 1) % self.args.print_freq == 0:
                print('epoch : %2d|%2d, iter : %3d|%3d,  loss : %.4f' % \
                      (epoch, self.args.end_epoch, idx+1, len(self.data['train_loader']), np.mean(loss_recorder)))
        self._calculate_acc(gt_occ, gt_hos, gt_img, pred_score, False)
        train_loss = np.mean(loss_recorder)
        print('train_loss : %.4f' % train_loss)


    def _model_eval(self):

        self.model['backbone'].eval()
        losses = []
        with torch.no_grad():
            gt_occ, gt_hos, gt_img, pred_score = [], [], [], []
            for idx, (img, gt_label, _) in enumerate(self.data['eval_loader']):

                img.requires_grad         = False
                gt_label[0].requires_grad = False  # occ_label
                gt_label[1].requires_grad = False  # hos_label
                gt_label[2].requires_grad = False  # hos_label

                if self.device:
                    img = img.cuda()
                    gt_label[0] = gt_label[0].cuda()
                    gt_label[1] = gt_label[1].cuda()
                    gt_label[2] = gt_label[2].cuda()
                    
                score = self.model['backbone'](img)
                loss = self.model['criterion'](score, gt_label)
                losses.append(loss.item())
                gt_occ.extend(gt_label[0].cpu().detach().numpy().tolist())
                gt_hos.extend(gt_label[1].cpu().detach().numpy().tolist())
                gt_img.extend(gt_label[2].cpu().detach().numpy().tolist())
                pred_score.extend(score.cpu().detach().numpy().tolist())
            eval_loss = np.mean(losses)
            print('eval_loss : %.4f' % eval_loss)
            f1_score = self._calculate_acc(gt_occ, gt_hos, gt_img, pred_score, verbose=True)
            
        return eval_loss, f1_score


    def _main_loop(self):

        min_loss = 100
        max_f1_score = 0.0
        for epoch in range(self.args.start_epoch, self.args.end_epoch + 1):
            
            start_time = time.time()
            self._model_train(epoch)
            self.model['scheduler'].step()
            val_loss, f1_score = self._model_eval()
            end_time = time.time()
            print('Single epoch cost time : %.2f mins' % ((end_time - start_time)/60))
            
            if not os.path.exists(self.args.snapshot):
                os.mkdir(self.args.snapshot)
                
            if (min_loss > val_loss) & (max_f1_score < f1_score):
                print('%snew SOTA was found%s' % ('*'*16, '*'*16))
                min_loss = val_loss
                filename = os.path.join(self.args.snapshot, 'sota.pth.tar')
                torch.save({
                    'epoch'   : epoch,
                    'backbone': self.model['backbone'].state_dict(),
                    'f1_score': f1_score
                }, filename)

            if epoch % self.args.save_freq == 0:
                filename = os.path.join(self.args.snapshot, 'epoch_'+str(epoch)+'.pth.tar')
                torch.save({
                    'epoch'   : epoch,
                    'backbone': self.model['backbone'].state_dict(),
                    'f1_score': f1_score
                }, filename)


    def train_runner(self):

        self._report_settings()

        self._model_loader()

        self._data_loader()

        self._main_loop()


if __name__ == "__main__":

    anodetector = AnomalyDetetor(training_args())
    anodetector.train_runner()
