from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import os

import torch
from torch.utils import data
from torch.autograd import Variable
import os.path as osp
from config import config
from config import update_config
from datainfo.dataset import VOCDataSet,VOC12ClassificationDatasetMSF
from tool import *
from tool.dist_ops import synchronize
from tool import torchutils,imutils,dgcnutils,pyutils
import importlib
import cv2
import warnings

cudnn.enabled = True

def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id % args.gpu_nums):

        model.cuda()

        for iter, pack in enumerate(data_loader):
            # pack['img'] contains a list for 4 resolution, each one in list is different shape
            # torch.Size([1, 2, 3, 281, 500])
            # torch.Size([1, 2, 3, 140, 250])
            # torch.Size([1, 2, 3, 422, 750])
            # torch.Size([1, 2, 3, 562, 1000])
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size'] # [tensor([281]), tensor([500])]

            # 正反都算
            # output is the list contains 4 results, each one in list is different shape
            # torch.Size([20, 18, 32])
            # torch.Size([20, 9, 16])
            # torch.Size([20, 27, 47])
            # torch.Size([20, 36, 63])
            outputs = [model(img[0].cuda(non_blocking=True))
                       for img in pack['img']]
            # shape torch.Size([21, 366, 500])
            pred = torch.sum(torch.stack(
                [F.interpolate(o, size, mode='bilinear', align_corners=True)[0] for o in outputs]), 0) # torch.Size([20, 71, 125])

            pred = pred.cpu().data.numpy()
            pred_exp = np.exp(pred - np.max(pred, axis=0, keepdims=True))
            probs = pred_exp / np.sum(pred_exp, axis=0, keepdims=True)
            eps = 0.00001
            probs[probs < eps] = eps

            if args.eval_with_crf:
                # crf needs HWC format image
                im = np.ascontiguousarray(np.ascontiguousarray(pack['img'][1][0][0], dtype=np.float32).transpose((1, 2, 0)) \
                                          + imutils.VOC12_MEAN_RGB[None, None, :], dtype=np.uint8)
                probs = imutils.crf_inference(im, probs, scale_factor=1.0, labels=args.num_classes)

            result = np.argmax(probs, axis=0)
            # save cams
            if not os.path.exists(args.evaluation_out_dir_4_this_epoch):
                os.makedirs(args.evaluation_out_dir_4_this_epoch)
            save_path = osp.join(args.evaluation_out_dir_4_this_epoch, img_name + '.png')
            cv2.imwrite(save_path, result)

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    torchutils.setup_seed(args.random_seed)

    update_config(config)
    model = getattr(importlib.import_module(args.model), 'get_seg_model')(config)

    # model = torch.nn.DataParallel(model)
    n_threads = torch.cuda.device_count() * 8
    dataset = VOC12ClassificationDatasetMSF(args.voc12_eval_list,
                                                             voc12_root=args.voc12_data_dir, scales=args.multi_scales)
    for i in range(args.epoch):
        args.evaluation_out_dir_4_this_epoch = args.evaluation_out_dir + "/Epoch_" + str(i+1)

        # load params
        model.load_state_dict(torchutils.load_state_no_module(args.snapshot_out_dir + "/Epoch_" + str(i+1) + '.pth',
                                         map_location=torch.device('cpu')))

        model.eval()

        dataset = torchutils.split_dataset(dataset, n_threads)

        print('[ ', end='')
        multiprocessing.spawn(_work, nprocs=n_threads, args=(model, dataset, args), join=True)
        print(']')

        torch.cuda.empty_cache()
        break