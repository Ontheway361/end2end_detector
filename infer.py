#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import os
import sys
import time
import torch
import shutil
import numpy as np
import torchvision
from sklearn import metrics
from torch.utils.data import DataLoader

from config import infer_args
from classifier_lib import MobileNetV2, AkuDataset, resnet34

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
            # self.model['backbone'] = MobileNetV2(num_classes=2).cuda()
            # self.model['backbone'] = torchvision.models.AlexNet(num_classes=2).cuda()
            self.model['backbone'] = resnet34(num_classes=2).cuda()
        else:
            self.model['backbone'] = resnet34(num_classes=2)
            # self.model['backbone'] = MobileNetV2(num_classes=2)
            # self.model['backbone'] = torchvision.models.AlexNet(num_classes=2)

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
            
            cpu_model = self.model['backbone'].module
            torch.save(cpu_model.state_dict(), 'cpu_mode.pth')
            print('Resuming the train process at %3d epoches ...' % self.args.start_epoch)
        print('Model loading was finished ...')


    def _data_loader(self):

        self.data['eval_loader']  = DataLoader(AkuDataset(self.args.eval_file, self.args.in_size),
                                        batch_size=self.args.batchsize,
                                        shuffle=False,
                                        num_workers=self.args.workers)
        print('Data loading was finished ...')


    def _calculate_acc(self, gt_label, pred_score):
        
        pred_score = self.softmax(torch.from_numpy(np.array(pred_score)))
        pred_label = pred_score.argmax(dim=1).numpy().tolist()
        print(metrics.classification_report(gt_label, pred_label, digits=4))
        print(metrics.confusion_matrix(gt_label, pred_label))


    def _model_eval(self):

        self.model['backbone'].eval()
        losses = []
        with torch.no_grad():
            gt_label_list, pred_label_list, img_name_list = [], [], []
            for idx, (img, gt_label, _) in enumerate(self.data['eval_loader']):

                img.requires_grad      = False
                gt_label.requires_grad = False
                if self.device:
                    img      = img.cuda()
                    gt_label = gt_label.cuda()
                score = self.model['backbone'](img)
                img_name_list.extend(img_name)
                gt_label_list.extend(gt_label.cpu().detach().numpy().tolist())
                pred_label_list.extend(score.cpu().detach().numpy().tolist())
            self._calculate_acc(gt_label_list, pred_label_list)

    
    def _model_infer(self):
        
        self.model['backbone'].eval()
        with torch.no_grad():
            pred_label_list, img_path_list = [], []
            for idx, (img, gt_label, img_path) in enumerate(self.data['eval_loader']):

                img.requires_grad      = False
                gt_label.requires_grad = False
                if self.device:
                    img      = img.cuda()
                    gt_label = gt_label.cuda()
                score = self.model['backbone'](img)
                img_path_list.extend(img_path)
                pred_label_list.extend(score.cpu().detach().numpy().tolist())
                
                if (idx + 1) % 50 == 0:
                    print('already precessed %3d, total %3d ...' % (idx+1, len(self.data['eval_loader'])))
            save_info = []
            pred_score = self.softmax(torch.from_numpy(np.array(pred_label_list)))
            pred_probs = pred_score.numpy().tolist()
            for idx, (abs_name, pred) in enumerate(zip(img_path_list, pred_probs)):
                
                pred_prob = pred[1]
                img_name  = abs_name.split('/')[-1]
                if pred_prob > 0.9:
                    img_dir = os.path.join(self.args.save_dir, 'pos', img_name)
                elif pred_prob < 0.1:
                    img_dir = os.path.join(self.args.save_dir, 'neg', img_name)
                else:
                    img_dir = os.path.join(self.args.save_dir, 'unknow', img_name)
                save_info.append([abs_name, pred_prob])
                shutil.copyfile(abs_name, img_dir)
                
                if (idx+1) % 1000 ==0:
                    print('already precessed %3d images, total %3d' % (idx+1, len(img_path_list)))
            np.save(os.path.join(self.args.save_dir, 'infer_result.npy'), save_info)
            print('Inference was finished ...')
                
                
    def _main_loop(self):
        
        start_time = time.time()
        # self._model_eval()
        self._model_infer()
        end_time   = time.time()
        print('Inference cost time : %.2f mins' % ((end_time - start_time)/60))
            
            
    def train_runner(self):

        self._report_settings()

        self._model_loader()

        # self._data_loader()

        # self._main_loop()


if __name__ == "__main__":

    occdetector = OccDetetor(infer_args())
    occdetector.train_runner()
