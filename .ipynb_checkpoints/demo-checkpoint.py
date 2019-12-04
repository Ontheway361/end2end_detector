#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import os
import cv2
import sys
import time
import torch
import shutil
import numpy as np
import torchvision
from sklearn import metrics
from torch.utils.data import DataLoader

import detect_lib as lib
from config import infer_args


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
            #self.model['backbone'].load_state_dict(checkpoint)
            print('Resuming the train process at %d epoches ...' % self.args.start_epoch)
        print('Model loading was finished ...')


    def _data_loader(self):

        self.transform = torchvision.transforms.Compose([
                             torchvision.transforms.ToTensor(),
                             torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], \
                                                              std=[0.5, 0.5, 0.5]),
                         ])
        print('Data loading was finished ...')


    def _calculate_acc(self, gt_occ, gt_hos, gt_img, scores, verbose = False):
        
        npy_score = np.array(scores)
        occ_prob  = self.softmax(torch.from_numpy(npy_score[:, :2]))
        hos_prob  = self.softmax(torch.from_numpy(npy_score[:, 2:]))
        
        occ_label = occ_prob.argmax(dim=1).numpy()
        hos_label = hos_prob.argmax(dim=1).numpy()
        pred_info = ((occ_label == 1) & (hos_label == 0)) * 1
        
        print('There are %3d gt_pos, %3d pred_pos' % (sum(gt_img), sum(pred_info)))
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
            self._calculate_acc(gt_occ, gt_hos, gt_img, pred_score, verbose=True)
            print('eval_loss : %.4f' % eval_loss)
        return eval_loss


    def _model_infer(self):

        self.model['backbone'].eval()
        with torch.no_grad():
            
            # inference
            pred_score_list, img_path_list = [], []
            for idx, (img, _, img_path) in enumerate(self.data['eval_loader']):

                img.requires_grad      = False
                if self.device:
                    img = img.cuda()
                score = self.model['backbone'](img)
                img_path_list.extend(img_path)
                pred_score_list.extend(score.cpu().detach().numpy().tolist())

                if (idx + 1) % 10 == 0:
                    print('already precessed %3d, total %3d ...' % (idx+1, len(self.data['eval_loader'])))
                 
            save_info = []
            npy_score = np.array(pred_score_list)
            occ_prob  = self.softmax(torch.from_numpy(npy_score[:, :2]))
            hos_prob  = self.softmax(torch.from_numpy(npy_score[:, 2:]))
            
            occ_label = occ_prob.argmax(dim=1).numpy()
            hos_label = hos_prob.argmax(dim=1).numpy()
            occ_prob  = occ_prob.numpy()[:,1]
            # pred_info = ((occ_label == 1) & (hos_label == 0)) * 1
            
            # predict or analysis
            select_data = []
            cnt_hos = 0
            for idx, (abs_name, occ, hos) in enumerate(zip(img_path_list, occ_prob, hos_label)):   # TODO
                
                # img_name  = abs_name.split('/')[-1].split('.')[0] + '_' + str(pred) + '_.jpg'
                if hos == 0:
                    try:
                        img = cv2.resize(cv2.imread(abs_name), (self.args.in_size, self.args.in_size))
                    except:
                        continue
                    else:
                        if occ > 0.95:
                            select_data.append(abs_name + ' 1\n')
                        elif occ < 0.05:
                            select_data.append(abs_name + ' 0\n')
                else:
                    cnt_hos += 1
                    '''
                    try:
                        img = cv2.resize(cv2.imread(abs_name), (self.args.in_size, self.args.in_size))
                    except:
                        print('Errors occurs in %s' % abs_name)
                    else:
                        save_name = os.path.join(self.args.save_dir, img_name)
                        if cv2.imwrite(save_name, img):
                            cnt_hos += 1
                    '''
                '''
                pred_prob = pred[1]
                if pred_prob > 0.6:
                    img_dir = os.path.join(self.args.save_dir, 'pos', img_name)
                elif pred_prob < 0.4:
                    img_dir = os.path.join(self.args.save_dir, 'neg', img_name)
                else:
                    img_dir = os.path.join(self.args.save_dir, 'unknow', img_name)
                save_info.append([abs_name, pred_prob])
                shutil.copyfile(abs_name, img_dir)
                '''
                
                if (idx+1) % 1000 ==0:
                    print('already precessed %3d images, total %3d' % (idx+1, len(img_path_list)))
            # np.save(os.path.join(self.args.save_dir, 'infer_result.npy'), save_info)
            with open(self.args.out_file, 'w') as f:
                f.writelines(select_data)
            f.close()
            print('There are %3d hos-images among %3d images, ratio : %.4f' % \
                  (cnt_hos, len(img_path_list), cnt_hos/len(img_path_list)))
            print('There are %3d images selected from %3d images' % \
                  (len(select_data), len(img_path_list)))
            print('Inference was finished ...')


    def _main_loop(self):

        start_time = time.time()
        self._model_eval()
        # self._model_infer()
        end_time   = time.time()
        print('Inference cost time : %.2f mins' % ((end_time - start_time)/60))


    def runner(self):

        self._report_settings()

        self._model_loader()

        self._data_loader()

        self._main_loop()


if __name__ == "__main__":

    detector = AnomalyDetetor(infer_args())
    detector.runner()
