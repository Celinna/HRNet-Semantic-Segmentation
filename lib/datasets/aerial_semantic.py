# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset

class AerialSemantic(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=9,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=64, 
                 crop_size=(256, 384), 
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.0, 0.0, 0.0], 
                 std=[1.0, 1.0, 1.0]):

        super(AerialSemantic, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.label_mapping = {-1: ignore_label, 0: ignore_label, 
                                  1: 0, 2: 1, 3: 2, 4: 3, 
                                  5: 4, 6: 5, 7: 6, 8: 7, 
                                  9: 8, 10: 9, 11: 10, 12: 11, 
                                  13: 12, 14: 13, 15: 14, 
                                  16: 15, 17: 16, 18: 17, 
                                  19: 18, 20: 19, 21: 20, 
                                  22: 21, 23: ignore_label}
        
        self.class_weights = torch.FloatTensor([0.1183, 1.3586, 0.2262, 0.6143, 
                                                2.1583, 5.9558, 9.07124, 0.6617, 
                                                0.6600, 1.6541, 9.5407, 170.7932, 
                                                5.07813, 73.2289, 4.1526, 310.7591, 
                                                5.5883, 22.4242, 1.9594, 3.9591, 
                                                18.9286, 1.2774]).cuda()
 
            
    
    def read_files(self):
        files = []
#         if 'test' in self.list_path:
#             for item in self.img_list:
#                 image_path = item
#                 name = os.path.splitext(os.path.basename(image_path[0]))[0]
#                 files.append({
#                     "img": image_path[0],
#                     "name": name,
#                 })
#         else:
        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name,
                "weight": 1
            })
        return files
        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        
        image = cv2.imread(os.path.join(self.root, item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape

#         if 'test' in self.list_path:
#             image = self.input_transform(image)
#             image = image.transpose((2, 0, 1))

#             return image.copy(), np.array(size), name

        label = cv2.imread(os.path.join(self.root, item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        image, label = self.gen_sample(image, label, 
                                self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]

            preds = F.interpolate(
                preds, (ori_height, ori_width), 
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )            
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        # palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
#             save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))
