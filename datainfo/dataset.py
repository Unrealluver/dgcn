import os
from torchvision import transforms
from randaugment import *
import os.path as osp
import sys
import numpy as np
import random
import collections
import torch
import torchvision
from scipy.ndimage import zoom
import cv2
from torch.utils import data
from tool.cues_reader import CuesReader
import imutils
import imageio


IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
IGNORE = 255

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

N_CAT = len(CAT_LIST)

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

cls_labels_dict = np.load("datainfo/list/cls_labels.npy", allow_pickle=True).item()

def decode_int_filename(int_filename):
    s = str(int(int_filename))
    return s[:4] + '_' + s[4:]

def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    elem_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME, decode_int_filename(img_name) + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((N_CAT), np.float32)

    for elem in elem_list:
        cat_name = elem.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab

def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]

def load_image_label_list_from_npy(img_name_list):

    return np.array([cls_labels_dict[img_name] for img_name in img_name_list])

def get_img_path(img_name, voc12_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')

def load_img_name_list(dataset_path):

    img_name_list = np.loadtxt(dataset_path, dtype=np.int32)

    return img_name_list


class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img


class TorchvisionMeanReduce():
    def __init__(self, mean=(122.67891434,116.66876762,104.00698793)):
        self.mean = mean

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] - self.mean[0])
        proc_img[..., 1] = (imgarr[..., 1] - self.mean[1])
        proc_img[..., 2] = (imgarr[..., 2] - self.mean[2])

        return proc_img


class VOCDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean_rgb=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=255,cues_dir=None, cues_name=None):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        # self.ignore_label = 0
        self.mean_rgb = mean_rgb
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split(' ') for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.cues_reader = CuesReader(cues_dir, cues_name)
        self.files = []
        for name,ids in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s" % name)
            mask = osp.join(self.root, "SegmentationClassAug/%s.png" % name.split('.')[0])
            self.files.append({
                "img": img_file,
                "id": ids,
                "mask" : mask
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label, cues):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        cues = cv2.resize(cues, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label, cues

    """输入为：样本的size和生成的随机lamda值"""

    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        """1.论文里的公式2，求出B的rw,rh"""
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        """2.论文里的公式2，求出B的rx,ry（bbox的中心点）"""
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        # 限制坐标区域不超过样本大小

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        """3.返回剪裁B区域的坐标值"""
        return bbx1, bby1, bbx2, bby2


    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        image = image[:,:,[2, 1, 0]]
        mask = cv2.imread(datafiles["mask"], cv2.IMREAD_GRAYSCALE)
        image_id = datafiles["id"]
        labels, cues= self.cues_reader.get_forward(image_id)
        size = image.shape

        # construct the markers for cues in image size
        markers_new = np.ones((41, 41))*255
        pos = np.where(cues == 1)
        markers_new[pos[1], pos[2]] = pos[0]
        img_h, img_w, _ = image.shape
        markers = zoom(markers_new, (float(img_h) / markers_new.shape[0], float(img_w) / markers_new.shape[1]), order=0)

        # resize long operation
        if self.scale:
            # image, mask, markers = self.generate_scale_label(image, mask, markers)
            image = imutils.random_resize_long(image, 320, 640)
            # rescale only apply for cam generation
            # image = imutils.random_scale(image, scale_range=self.rescale, order=3)
            markers = zoom(markers,
                           (float(image.shape[0]) / markers.shape[0], float(image.shape[1]) / markers.shape[1]),
                           order=0)
            mask = zoom(mask, (float(image.shape[0]) / mask.shape[0], float(image.shape[1]) / mask.shape[1]), order=0)

        image = Image.fromarray(image)
        # Color Transform
        color_transform = transforms.Compose(
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        )
        image = color_transform.transforms(image)

        # Do the RandAug only apply for img
        random_transform = transforms.Compose(
            RandAugmentOI(n=2, m=10)
        )
        image = random_transform.transforms(image)

        # Do the Both Aug RandAug
        # do nothing now!!!!!
        mask = Image.fromarray(mask)
        markers = Image.fromarray(markers)
        # multi_random_transform = transforms.Compose(
        #     RandAugmentMA(n=2, m=10)
        # )
        # list = multi_random_transform.transforms([image, mask, markers])
        # image = list[0]
        # mask = list[1]
        # markers = list[2]


        image = np.array(image)
        mask = np.array(mask)
        markers = np.array(markers)

        # Cutmix Augmentation
        # C_BETA = 1
        # # generate mixed sample
        # """1.设定lamda的值，服从beta分布"""
        # lam = np.random.beta(C_BETA, C_BETA)
        # """2.找到两个随机样本"""
        # rand_index = torch.randperm(input.size()[0]).cuda()
        # target_a = target  # 一个batch
        # target_b = target[rand_index]  # batch中的某一张
        # """3.生成剪裁区域B"""
        # bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        # """4.将原有的样本A中的B区域，替换成样本B中的B区域"""
        # input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        # # adjust lambda to exactly match pixel ratio
        # """5.根据剪裁区域坐标框的值调整lam的值"""
        # lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))


        # Norm op for image
        IRNetNormFlag = False
        if IRNetNormFlag:
            img_norm = TorchvisionNormalize()
            image = img_norm(image)
        else:
            image = np.asarray(image, np.float32)
            image -= self.mean_rgb

        img_h, img_w, _ = image.shape

        # do the padding operation before Crop
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(mask, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
            markers_pad = cv2.copyMakeBorder(markers, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad, markers_pad = image, mask, markers
        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        # image = zoom(image, ((321.0) / image.shape[0], (321.0) / image.shape[1]), order=0)

	    # resize to 321 to avoid CUDA OOM
        # image_ = np.ndarray(shape=(321, 321, 3))
        # for i in range(3):
        #     image_[:, :, i] = (zoom(image[:, :, i], ((321.0) / (image.shape[0]), (321.0) / (image.shape[1])), order=0))
        # image = image_
        # label = zoom(label, (float(321.0) / label.shape[0], float(321.0) / label.shape[1]), order=0)

        # reconstruct the 41 x 41 cues
        markers_pad = np.asarray(markers_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        markers_pad = zoom(markers_pad, (41.0 / markers_pad.shape[0], 41.0 / markers_pad.shape[1]), order=0) 
        cues_new = np.zeros(cues.shape)
        for class_i in range(cues.shape[0]):
            pos = np.where(markers_pad == class_i)
            if len(pos)==0:
                continue
            cues_new[class_i,pos[0],pos[1]] = 1

        # change to CHW
        image = image.transpose((2, 0, 1))

        # do the mirror operation in Horizontal
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            cues_new = cues_new[:, :, ::flip]
            label = label[:, ::flip]
        return image.copy(), cues_new.copy(), labels, label.copy()


class VOC12ImageDataset(data.Dataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionMeanReduce(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = np.asarray(imageio.imread(get_img_path(name_str, self.voc12_root)))

        if self.resize_long:
            img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img = imutils.random_scale(img, scale_range=self.rescale, order=3)

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img = imutils.random_lr_flip(img)

        if self.crop_size:
            if self.crop_method == "random":
                img = imutils.random_crop(img, self.crop_size, 0)
            else:
                img = imutils.top_left_crop(img, self.crop_size, 0)

        if self.to_torch:
            img = imutils.HWC_to_CHW(img)

        return {'name': name_str, 'img': img}

class VOC12ClassificationDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionMeanReduce(), hor_flip=False,
                 crop_size=None, crop_method=None):
        super().__init__(img_name_list_path, voc12_root,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        out['label'] = torch.from_numpy(self.label_list[idx])

        return out

class VOC12ClassificationDatasetMSF(VOC12ClassificationDataset):

    def __init__(self, img_name_list_path, voc12_root,
                 img_normal=TorchvisionMeanReduce(),
                 scales=(1.0,)):
        self.scales = scales

        super().__init__(img_name_list_path, voc12_root, img_normal=img_normal)
        self.scales = scales

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = imageio.imread(get_img_path(name_str, self.voc12_root))

        ms_img_list = []
        # 不同scale加入
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            s_img = self.img_normal(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            # cancle 翻转加入
            # ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
            ms_img_list.append(np.stack([s_img], axis=0))
        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]

        out = {"name": name_str, "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
               "label": torch.from_numpy(self.label_list[idx])}
        return out


class VOCDataSetVal(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255,cues_dir=None, cues_name=None):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split(' ') for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.cues_reader = CuesReader(cues_dir, cues_name)
        self.files = []
        for name,ids in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s" % name)
            mask = osp.join(self.root, "SegmentationClassAug/%s.png" % name.split('.')[0])
            self.files.append({
                "img": img_file,
                "id": ids,
                "mask" : mask
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label, cues):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        cues = cv2.resize(cues, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label, cues

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        mask = cv2.imread(datafiles["mask"], cv2.IMREAD_GRAYSCALE)
        image_id = datafiles["id"]
        labels, cues= self.cues_reader.get_forward(image_id)
        size = image.shape

        # construct the markers for cues in image size
        markers_new = np.ones((41, 41))*255
        pos = np.where(cues == 1)
        markers_new[pos[1], pos[2]] = pos[0]
        img_h, img_w, _ = image.shape
        markers = zoom(markers_new, (float(img_h) / markers_new.shape[0], float(img_w) / markers_new.shape[1]), order=0)

        # resize long operation
        if self.scale:
            # image, mask, markers = self.generate_scale_label(image, mask, markers)
            image = imutils.random_resize_long(image, 320, 640)
            # rescale only apply for cam generation
            # image = imutils.random_scale(image, scale_range=self.rescale, order=3)
            markers = zoom(markers,
                           (float(image.shape[0]) / markers.shape[0], float(image.shape[1]) / markers.shape[1]),
                           order=0)
            mask = zoom(mask, (float(image.shape[0]) / mask.shape[0], float(image.shape[1]) / mask.shape[1]), order=0)

        image = Image.fromarray(image)
        # Color Transform
        color_transform = transforms.Compose(
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        )
        image = color_transform.transforms(image)

        # Do the RandAug only apply for img
        random_transform = transforms.Compose(
            RandAugmentOI(n=2, m=10)
        )
        image = random_transform.transforms(image)

        image = np.array(image)

        # Norm op for image
        IRNetNormFlag = False
        if IRNetNormFlag:
            img_norm = TorchvisionNormalize()
            image = img_norm(image)
        else:
            image = np.asarray(image, np.float32)
            image -= self.mean

        img_h, img_w, _ = image.shape

        # do the padding operation before Crop
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(mask, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
            markers_pad = cv2.copyMakeBorder(markers, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad, markers_pad = image, mask, markers
        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        # image = zoom(image, ((321.0) / image.shape[0], (321.0) / image.shape[1]), order=0)

	    # resize to 321 to avoid CUDA OOM
        # image_ = np.ndarray(shape=(321, 321, 3))
        # for i in range(3):
        #     image_[:, :, i] = (zoom(image[:, :, i], ((321.0) / (image.shape[0]), (321.0) / (image.shape[1])), order=0))
        # image = image_
        # label = zoom(label, (float(321.0) / label.shape[0], float(321.0) / label.shape[1]), order=0)

        # reconstruct the 41 x 41 cues
        markers_pad = np.asarray(markers_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        markers_pad = zoom(markers_pad, (41.0 / markers_pad.shape[0], 41.0 / markers_pad.shape[1]), order=0)
        cues_new = np.zeros(cues.shape)
        for class_i in range(cues.shape[0]):
            pos = np.where(markers_pad == class_i)
            if len(pos)==0:
                continue
            cues_new[class_i,pos[0],pos[1]] = 1
        image = image[:,:,[2, 1, 0]]
        image = image.transpose((2, 0, 1))

        # do the mirror operation in Horizontal
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            cues_new = cues_new[:, :, ::flip]
            label = label[:, ::flip]
        return image.copy(),cues_new.copy(), labels

class VOCRetrainDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split(' ') for i_id in open(list_path)]
        if not max_iters==None:
           self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for image,label in self.img_ids:
            img_file = (self.root+image)
            label_file = label
            self.files.append({
                "img": img_file,
                "label": label_file,
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        return image

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        if self.scale:
            image = self.generate_scale_label(image)

        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w, _ = image.shape

        label = zoom(label, (float(img_h) / 41.0, float(img_w) / 41.0),order=0)
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(255.0))
        else:
            img_pad, label_pad = image, label

        img_h, img_w,_ = img_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image[:,:,[2, 1, 0]]
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
        return image.copy(),label.copy()

