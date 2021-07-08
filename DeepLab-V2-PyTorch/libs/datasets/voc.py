#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   08 February 2019

from __future__ import absolute_import, print_function

import os.path as osp

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data

from .base import _BaseDataset


class VOC(_BaseDataset):
    """
    PASCAL VOC Segmentation dataset
    """

    def __init__(self, year=2012, **kwargs):
        self.year = year
        super(VOC, self).__init__(**kwargs)

    def _set_files(self):
        self.root = osp.join(self.root, "VOC{}".format(self.year))
        self.image_dir = osp.join(self.root, "JPEGImages")
        self.label_dir = osp.join(self.root, "SegmentationClass")

        if self.split in ["train", "trainval", "val", "test"]:
            file_list = osp.join(
                self.root, "ImageSets/Segmentation", self.split + ".txt"
            )
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files = file_list
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index]
        image_path = osp.join(self.root, self.image_dir, image_id + ".jpg")
        label_path = osp.join(self.root, self.label_dir, image_id + ".png")
        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        return image_id, image, label


class VOCAug(_BaseDataset):
    """
    PASCAL VOC Segmentation dataset with extra annotations
    """

    def __init__(self, year=2012, **kwargs):
        self.year = year
        super(VOCAug, self).__init__(**kwargs)

    def _set_files(self):
        self.root = osp.join(self.root, "VOC{}".format(self.year))
        self.image_dir_path = osp.join(self.root, 'JPEGImages')
        self.label_dir_path = osp.join(self.root, self.gt_path)
    
        self.datalist_file = osp.join(self.root, "ImageSets/Segmentation", self.split + ".txt")
        print(self.datalist_file)
        self.image_ids, self.cls_labels = self.read_labeled_image_list(self.root, self.datalist_file)
        
    def _load_data(self, index):
        image_id = self.image_ids[index]
        image_path = osp.join(self.image_dir_path, image_id + '.jpg')
        label_path = osp.join(self.label_dir_path, image_id + '.png')
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32) if osp.exists(label_path) else np.zeros((100, 100))
        cls_label = self.cls_labels[index]
        
        return image_id, image, label, cls_label

    def read_labeled_image_list(self, data_dir, data_list):
        #img_dir = os.path.join(data_dir, "JPEGImages")
        
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        
        for line in lines:
            fields = line.strip().split()
            
            labels = np.zeros((21,), dtype=np.float32)
            labels[0] = 1. #background
            
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index+1] = 1.
                
            #img_name_list.append(os.path.join(img_dir, image))
            img_name_list.append(fields[0])
            img_labels.append(labels)
            
        return img_name_list, img_labels
    
    
if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import torchvision
    import yaml
    from torchvision.utils import make_grid
    from tqdm import tqdm

    kwargs = {"nrow": 10, "padding": 50}
    batch_size = 100

    dataset = VOCAug(
        root="/home/lianghuizhu/data/VOC2012/VOCdevkit",
        split="train_aug",
        ignore_label=255,
        mean_bgr=(104.008, 116.669, 122.675),
        year=2012,
        augment=True,
        base_size=None,
        crop_size=513,
        scales=(0.5, 0.75, 1.0, 1.25, 1.5),
        flip=True,
    )
    print(dataset)

    loader = data.DataLoader(dataset, batch_size=batch_size)

    for i, (image_ids, images, labels) in tqdm(
        enumerate(loader), total=np.ceil(len(dataset) / batch_size), leave=False
    ):
        if i == 0:
            mean = torch.tensor((104.008, 116.669, 122.675))[None, :, None, None]
            images += mean.expand_as(images)
            image = make_grid(images, pad_value=-1, **kwargs).numpy()
            image = np.transpose(image, (1, 2, 0))
            mask = np.zeros(image.shape[:2])
            mask[(image != -1)[..., 0]] = 255
            image = np.dstack((image, mask)).astype(np.uint8)

            labels = labels[:, np.newaxis, ...]
            label = make_grid(labels, pad_value=255, **kwargs).numpy()
            label_ = np.transpose(label, (1, 2, 0))[..., 0].astype(np.float32)
            label = cm.jet_r(label_ / 21.0) * 255
            mask = np.zeros(label.shape[:2])
            label[..., 3][(label_ == 255)] = 0
            label = label.astype(np.uint8)

            tiled_images = np.hstack((image, label))
            # cv2.imwrite("./docs/datasets/voc12.png", tiled_images)
            plt.imshow(np.dstack((tiled_images[..., 2::-1], tiled_images[..., 3])))
            plt.show()
            break
