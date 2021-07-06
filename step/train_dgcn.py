import torch
from torch.utils import data
from torch.autograd import Variable
import os.path as osp
from config import config
from config import update_config
from datainfo.dataset import VOCDataSet
from tool import *
from tool.dist_ops import synchronize
from tool import torchutils,imutils,dgcnutils,pyutils
import importlib


def run(args):
    torchutils.setup_seed(args.random_seed)

    # model initial
    update_config(config)
    model = getattr(importlib.import_module(args.model), 'get_seg_model')(config)
    if args.gpu_nums > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.train()

    # optimizer
    optimizer = torchutils.StepOptimizer([
        {'params': model.get_0x_lr_params(), 'lr': 0 * args.learning_rate, 'weight_decay': args.weight_decay},
        {'params': model.get_1x_lr_params(), 'lr': 1 * args.learning_rate, 'weight_decay': args.weight_decay},
        {'params': model.get_10x_lr_params(), 'lr': 10 * args.learning_rate, 'weight_decay': args.weight_decay},
    ], lr=args.learning_rate, weight_decay=args.weight_decay, gamma=args.gamma)
    optimizer.zero_grad()

    # dataset
    train_dataset = VOCDataSet(args.voc12_data_dir, args.voc12_data_list, max_iters=args.num_steps,
                               crop_size=(map(int, args.input_size.split(','))),
                               scale=True, mirror=True, mean_rgb=imutils.VOC12_MEAN_RGB, cues_dir=args.voc12_cues_dir,
                               cues_name=args.voc12_cues_name)
    torch.cuda.set_device(args.local_rank)
    if args.gpu_nums > 1:
        # torch.distributed.init_process_group(backend="nccl", init_method="env://")
        torch.distributed.init_process_group(backend="nccl", init_method='tcp://localhost:23456', rank=args.local_rank, world_size=args.gpu_nums)
        synchronize()
        world_size = torch.distributed.get_world_size()
        args.batch_size_n = int(args.batch_size / world_size)
        model = torch.nn.parallel.DistributedDataParallel(model.to(args.local_rank), device_ids=[args.local_rank],
                                                          output_device=args.local_rank, broadcast_buffers=False)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size_n, shuffle=False, num_workers=4,
                                      pin_memory=True, sampler=train_sampler)
    else:
        args.batch_size_n = args.batch_size
        model = torch.nn.DataParallel(model).cuda()
        trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size_n, shuffle=False, num_workers=4,
                                      pin_memory=True, sampler=None)

    # restore
    if args.restore_from is not None and args.restore_epoch is not None:
        print('load model ...')
        state_dict = torch.load(osp.join(args.snapshot_dir, 'Epoch_' + '2' + '.pth'))
        model.load_state_dict(state_dict)
        del state_dict

    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()

    # train
    min_prob = torch.tensor(0.0000001, device='cuda')
    for ep in range(args.epoch):

        if args.local_rank==0:
            print('Epoch %d/%d' % (ep + 1, args.epoch))

        if (args.restore_epoch is not None) and (ep < args.restore_epoch):
            continue

        if args.gpu_nums > 1:
            train_sampler.set_epoch(ep)

        for step_in_epoch, batch in enumerate(trainloader):
            images, cues, labels, masks = batch
            images = Variable(images).cuda().float()

            # foward
            feature, pred = model(images)
            feature = torch.nn.functional.interpolate(feature, size=[41, 41], mode='bilinear', align_corners=True)
            pred = torch.nn.functional.interpolate(pred, size=[41, 41], mode='bilinear', align_corners=True)

            # get KNN-Matrix
            labels = (cues.sum(dim=3).sum(dim=2) > 0).int()
            knn_matrix = dgcnutils.cacul_knn_matrix_(feature)

            # calcute crf loss
            probs = dgcnutils.softmax(pred, min_prob)
            crf = dgcnutils.crf_operation(images.cpu().detach().numpy(), probs.cpu().detach().numpy())
            crf_constrain_loss = dgcnutils.constrain_loss(probs, crf)

            # calculate peudo mask and seeding loss
            supervision = dgcnutils.generate_supervision(feature.cpu().detach().numpy(), labels, cues, None,
                                               probs.cpu().detach().numpy(), knn_matrix.cpu().detach().numpy())
            seeding_loss = dgcnutils.cal_seeding_loss(probs, supervision)

            # total loss
            loss = crf_constrain_loss + seeding_loss
            avg_meter.add({'boundary_loss': crf_constrain_loss.item(),
                           "seeding_loss": seeding_loss.item(),
                           "total_loss": loss.item(),})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(ep)

            if (optimizer.global_step-1)%100 == 0 and args.local_rank == 0:
                max_step = args.num_steps / args.batch_size * args.epoch + 1
                timer.update_progress(optimizer.global_step / max_step)

                print('step_in_this_epoch:%5d/%5d' % (step_in_epoch, args.num_steps / args.batch_size),
                      'boundary_loss:%.4f' % (avg_meter.pop('boundary_loss')),
                      'seeding_loss:%.4f' % (avg_meter.pop('seeding_loss')),
                      'total_loss:%.4f' % (avg_meter.pop('total_loss')),
                      'imps:%.1f' % ((step_in_epoch + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'lr: %.8f' % (optimizer.param_groups[1]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()),
                      'step_in_total:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      flush=True)

            if step_in_epoch == int(args.num_steps / args.batch_size) and args.local_rank == 0:
                print('save model ...')
                torch.save(model.state_dict(), osp.join(args.snapshot_out_dir, 'Epoch_' + str(ep + 1) + '.pth'))
                break

    torch.cuda.empty_cache()






