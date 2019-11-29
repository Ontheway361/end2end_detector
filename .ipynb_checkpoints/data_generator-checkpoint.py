#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import os
import cv2
import argparse
import numpy as np
from mtcnn import MtcnnDetector

from IPython import embed

class AugAkuFace(object):

    def __init__(self, args):

        self.args = args


    def _padding_img(self, img, trans_flag = False):

        height, width, _ = img.shape
        pad_img = np.zeros((self.args.in_size, self.args.in_size, 3), dtype=img.dtype)
        if height > width:
            map_width = int(width * self.args.in_size / height)
            padding   = int((self.args.in_size - map_width) / 2)
            pad_start = np.random.randint(0, self.args.in_size - map_width)
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


    def _flip_img(self, img):

        return self._padding_img(cv2.flip(img, 1))


    def _translate_img(self, img):

        return self._padding_img(img, True)


    @staticmethod
    def _rotate_mat(angle, center):

        rad = angle * np.pi / 180.0
        cos_r, sin_r = np.cos(rad), np.sin(rad)
        r_matrix = np.zeros((2,3), dtype=np.float32)
        r_matrix[0, 0], r_matrix[0, 1] = cos_r, sin_r
        r_matrix[1, 0], r_matrix[1, 1] = -sin_r, cos_r
        r_matrix[0, 2] = (1-cos_r) * center[0] - sin_r * center[1]
        r_matrix[1, 2] = sin_r * center[0] + (1-cos_r) * center[1]

        return r_matrix


    def _rotate_img(self, img):

        height, width, _ = img.shape
        center = (int(self.args.in_size/2), int(self.args.in_size/2))
        loose_size = (self.args.in_size, self.args.in_size)
        cnt_rotate, img_list = 0, []
        pad_img    = self._padding_img(img)
        while cnt_rotate < self.args.num_rotate:

            angle = np.random.randint(self.args.r_angle[0], self.args.r_angle[1])
            r_matrix   = self._rotate_mat(angle, center)
            if np.random.randint(2) > 0:
                pad_img = cv2.flip(pad_img, 1)
            rotate_img = cv2.warpAffine(pad_img, r_matrix, loose_size)
            img_list.append(rotate_img)
            cnt_rotate += 1
        return img_list

    
    def aug_single(self, img, anno_info):
        
        data = []
        img_name = anno_info[0].split('/')[-1].split('.')[0]
        try:
            img_list = [self._padding_img(img)]
            img_list.append(self._translate_img(img))
            if self.args.aug_flag:
                img_list.append(self._flip_img(img))
                img_list.extend(self._rotate_img(img))
            for i, aimg in enumerate(img_list):
                aimg_name = os.path.join(self.args.save_dir, img_name + '_' + str(i) + '.jpg')
                if cv2.imwrite(aimg_name, aimg):
                    data.append(aimg_name + ' ' + anno_info[1] + '\n')
        except Exception as e:
            print(e)
            print('error occurs in %s.jpg' % img_name)
        return data 
            
            
    
    
    def aug_runner(self, file=None):

        if file is None:
            with open(self.args.anno_file, 'r') as f:
                file = f.readlines()
            f.close()

        data = []
        for idx, row in enumerate(file):

            row = row.strip().split(' ')
            img = cv2.imread(row[0])
            data.extend(self.aug_single(img, row))
            if (idx + 1) % 500 == 0:
                print('already precessed %4d|%4d' % (idx+1, len(file)))
        with open(self.args.save_file, 'w') as f:
            f.writelines(data)
        f.close()
        print('Data augmentation was finished ...')


class CropFace(AugAkuFace):
    
    def __init__(self, args):
        
        AugAkuFace.__init__(self, args)
        self.args  = args
        self.mtcnn = MtcnnDetector(use_gpu=args.use_gpu)
    
    
    def _face_detector(self, img):
        
        bboxes, landmarks = self.mtcnn.detect_face(img)
        area = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1) * -1
        area_index = area.argsort()
        bbox     = bboxes[area_index[0]].astype(np.int32)
        # landmark = landmarks[area_index[0]].astype(np.int32)
        ratio = - area[area_index[0]] / np.prod(img.shape[:-1])
        return bbox, ratio
    
    
    def _crop_face(self, img):
        
        bbox, ratio = self._face_detector(img)
        if ratio > self.args.ratio_thres:
            face = img[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1, :]
        else:
            face = img  # bug-image, no-face or card-face was detected
        return face
    
    
    def face_runner(self, anno_file = None):
        
        if anno_file is None:
            with open(self.args.anno_file, 'r') as f:
                anno_file = f.readlines()
            f.close()
        
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        anno_data = []
        for idx, row in enumerate(anno_file):
            
            row = row.strip().split(' ')
            try:
                img  = cv2.imread(row[0])
                face = self._crop_face(img)
            except Exception as e:
                print(e)
                continue
            else:
                anno_data.extend(self.aug_single(face, row))
            if(idx+1) % 500 == 0:
                print('alread processed %4d images ...' % (idx+1))
        # np.save('ratio_list.npy', ratio_list)
        with open(self.args.save_file, 'w') as f:
            f.writelines(anno_data)
        f.close()
        print('Crop face was finished ...')
                
        
root_dir = '/home/jovyan/gpu3-data2/lujie/imgs_occ/filter_data_1113'

def aug_args():

    parser = argparse.ArgumentParser(description='Data Augmentation for AkuFace')
    
    # env
    parser.add_argument('--use_gpu',   type=str, default=False)
    
    # file
    parser.add_argument('--anno_file',   type=str, default=os.path.join(root_dir, 'akuface_test_1115.txt'))
    parser.add_argument('--in_size',     type=int, default=112)
    parser.add_argument('--aug_flag',    type=int, default=False)
    parser.add_argument('--num_rotate',  type=int, default=10)        # TODO
    parser.add_argument('--r_angle',     type=list,default=[-30, 30])
    parser.add_argument('--ratio_thres', type=float,default=0.0079)        # 2*sigma | p[u-2sigma, u+2sigma]=0.9544
    parser.add_argument('--save_dir',    type=str, default=os.path.join(root_dir, 'end2end_test_1129'))
    parser.add_argument('--save_file',   type=str, default=os.path.join(root_dir,'end2end_test_1129.txt'))

    args = parser.parse_args()
    return args


if __name__ == '__main__':

#     aug_engine = AugAkuFace(aug_args())
#     aug_engine.aug_runner()
    face_engine = CropFace(aug_args())
    face_engine.face_runner()
