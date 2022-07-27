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

class SwissOkutama(BaseDataset):
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
                 mean=[0.39313033, 0.48066333, 0.45113695], # for BGR channels
                 std=[0.1179, 0.1003, 0.1139],
                 inference=False):

        super(SwissOkutama, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        if not inference:
            self.img_list = [line.strip().split() for line in open(list_path)]

            self.files = self.read_files()
            if num_samples:
                self.files = self.files[:num_samples]
                
            self.class_weights = torch.FloatTensor([6.39009734, 0.70639623,
                                                    0.59038726, 0.50078311, 
                                                    17.07579971,  0.42550586,
                                                    7.51839971, 13.64782882]).cuda()    

        self.label_mapping = {0: ignore_label, 
                              1: 0, 2: 1, 
                              3: 2, 4: 3, 
                              5: 4, 6: 5, 
                              7: 6, 8: 7, 
                              9: ignore_label}
        
           
        self.colormap = {
                        # in RGB order
                        "Background": [0, 0, 0], 
                        "Outdoor structures": [237, 237, 237],
                        "Buildings": [181, 0, 0],
                        "Paved ground": [135, 135, 135],
                        "Non-paved ground": [189, 107, 0],
                        "Train tracks": [128, 0, 128],
                        "Plants": [31, 123, 22],
                        "Wheeled vehicles": [6, 0, 130],
                        "Water": [0, 168, 255],
                        "People": [240, 255, 0]
                    }
        self.idx2color = {k:v for k,v in enumerate(list(self.colormap.values()))}
 
            
    
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
                new_img = np.expand_dims(new_img, axis=0).astype(np.float32)
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
        

    def category2mask(self, img):
        """ Convert a category image to color mask """
        if len(img) == 3:
            if img.shape[2] == 3:
                img = img[:, :, 0]
    
        mask = np.zeros(img.shape[:2] + (3, ), dtype='uint8')
    
        for category, mask_color in self.idx2color.items():
            locs = np.where(img == category)
            mask[locs] = mask_color
        
        return mask
    

    def save_pred(self, preds, sv_path, name, rgb=False):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            if rgb:
                mask_rgb = self.category2mask(pred)
                save_img = Image.fromarray(mask_rgb)
                save_img.save(os.path.join(sv_path, name +'.png'))
            else:
                save_img = Image.fromarray(pred)
                save_img.save(os.path.join(sv_path, name[i]+'.png'))
            
