#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import os
import cv2
import argparse
import numpy as np
import pandas as pd

from IPython import embed

class HorizontalScreen(object):

    def __init__(self, args):

        self.args = args


    def _rotate_img(self, img, mode = 0):
        ''' clockwise rotate img in 90', 180', 270 '''

        return cv2.rotate(img, mode)

    
    
    def _padding_img(self, img, trans_flag = False):

        height, width, _ = img.shape
        pad_img = np.zeros((self.args.in_size, self.args.in_size, 3), dtype=img.dtype)
        if height >= width:
            map_width = int(width * self.args.in_size / height)
            padding   = int((self.args.in_size - map_width) / 2)
            try:
                pad_start = np.random.randint(0, self.args.in_size - map_width)
            except Exception as e:
                pad_start = 0
            resz_img  = cv2.resize(img, (map_width, self.args.in_size))
            if trans_flag:
                pad_img[:, pad_start:(map_width+pad_start), :] = resz_img
            else:
                pad_img[:, padding:(map_width+padding), :] = resz_img
        else:
            map_height = int(height * self.args.in_size / width)
            padding    = int((self.args.in_size - map_height) / 2)
            pad_start = np.random.randint(0, self.args.in_size - map_height)
            resz_img  = cv2.resize(img, (self.args.in_size, map_height))
            if trans_flag:
                pad_img[pad_start:(map_height+pad_start), :, :] = resz_img
            else:
                pad_img[padding:(map_height+padding), :, :] = resz_img

        return pad_img
    
    
    def _generator(self):

        with open(self.args.file, 'r') as f:
            img_list = f.readlines()
        f.close()
        '''
        try:
            df_data = pd.read_csv(self.args.file)
        except Exception as e:
            print(e)
            raise TypeError('errors occurs in %s' % self.args.file)
        '''
        print('There are %3d images in %s' % (len(img_list), self.args.file.split('/')[-1]))  # TODO | img_list <--> df_data
        
        out_data = []
        for i, img_info in enumerate(img_list):
        # for i, row in df_data.iterrows():
            
            img_info  = img_info.strip().split(' ')
            abs_path  = img_info[0]
            occ_label = img_info[1]
            # abs_path  = row['img_path']
            # occ_label = row['anno_label']
            try:
                img  = cv2.resize(cv2.imread(abs_path), (self.args.in_size, self.args.in_size))
                # img  = cv2.imread(abs_path)
                # img  = self._padding_img(img)
                mode = np.random.choice(4) - 1
                if mode != -1:
                    img = self._rotate_img(img, mode)
            except Exception as e:
                print('Errors occurs in %s' % abs_path)
                continue
            else:
                img_name  = abs_path.split('/')[-1] 
                save_path = os.path.join(self.args.save_dir, img_name)
                hos_label = mode + 1
                img_label = ((int(occ_label) == 1) & (mode == -1)) * 1
                if cv2.imwrite(save_path, img):
                    out_data.append([save_path, int(occ_label), hos_label, img_label])
            if (i + 1) % 1000 == 0:
                print('already processed %3d, total %3d images ...' % (i+1, len(img_list)))   # TODO | img_list <--> df_data
        df_out = pd.DataFrame(out_data, columns=['img_path', 'occ_label', 'hos_label', 'img_label'])
        df_out.to_csv(self.args.out_file, index=None)
        print('data generation was finished, there are %4d samples was generated.' % len(df_out))



# root_dir = '/home/jovyan/lujie/benchmark_imgs/akuface/akuface_1115_1.2w'
root_dir = '/home/jovyan/lujie/benchmark_imgs/akuface'


def aug_args():

    parser = argparse.ArgumentParser(description='Data Augmentation for AkuFace')

    # file
    parser.add_argument('--in_size',  type=int, default=112)
    # parser.add_argument('--root_dir', type=str, default=root_dir)
    parser.add_argument('--file',     type=str, default=os.path.join(root_dir, '5w_select.txt'))
    parser.add_argument('--save_dir', type=str, default=os.path.join(root_dir, 'temp'))
    parser.add_argument('--out_file', type=str, default=os.path.join(root_dir, 'akuface_anno/multitask_1203.csv'))

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    face_engine = HorizontalScreen(aug_args())
    face_engine._generator()
