import os
from tool import torchutils
import cv2
from tool.dgcnutils import ConfusionMatrix
from multiprocessing import Pool


def eval(args):
    data_list = []

    eval_list = [i.strip() for i in open(args.voc12_eval_list) if not i.strip() == '']
    gt_dir = args.voc12_data_dir + "SegmentationClassAug/"

    for index, img_id in enumerate(eval_list):
        pred_img_path = os.path.abspath(os.path.join(args.evaluation_out_dir_4_this_epoch, img_id + '.png'))
        gt_img_path = os.path.join(gt_dir, img_id + '.png')
        pred = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
        data_list.append([gt.flatten(), pred.flatten()])

    ConfM = ConfusionMatrix(args.num_classes)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    with open(args.evaluation_result_4_this_epoch, 'w') as f:
        f.write('meanIOU: ' + str(aveJ) + '\n')
        f.write(str(j_list) + '\n')
        f.write('\n')
        f.write(str(M) + '\n')

    return str(aveJ)


def run(args):
    torchutils.setup_seed(args.random_seed)

    miou_list = []
    for i in range(args.epoch):
        args.evaluation_out_dir_4_this_epoch = args.evaluation_out_dir + "/Epoch_" + str(i + 1)
        args.evaluation_result_4_this_epoch = args.evaluation_out_dir_4_this_epoch + ".txt"
        miou = eval(args)
        miou_list.append(miou)
        print("Epoch_" + str(i + 1) + " processed")

    for i in range(len(miou_list)):
        print('Epoch_' + str(i + 1) + ' miou: ' + miou_list[i])
