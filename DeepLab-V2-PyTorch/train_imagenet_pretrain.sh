# Training DeepLab-V2 using pseudo segmentation labels

DATASET="voc12_2gpu"
LOG_DIR="Deeplabv2_pseudo_segmentation_labels_2gpu"
GT_DIR="/home/lianghuizhu/Project8_WSSS/hrnet/tools/output/pascal_ctx/seg_hrnet_w48_cls21_480x480_sgd_lr4e-3_wd1e-4_bs_16_epoch20_both_best/NewDL321/evaluation/checkpoint_epoch6/"
#GT_DIR="/home/lianghuizhu/data/VOC2012/VOCdevkit/VOC2012/SegmentationClassAug"
CUDA_VISIBLE_DEVICES=2,3 python main.py train -c configs/${DATASET}.yaml --gt_path=${GT_DIR} --log_dir=${LOG_DIR}
# CUDA_VISIBLE_DEVICES=3 python main.py train -c configs/voc12_2gpu.yaml --gt_path=/home/lianghuizhu/data/VOC2012/VOCdevkit/VOC2012/SegmentationClassAug --log_dir=Deeplabv2_pseudo_segmentation_labels_2gpu
#---------------------------------------------------------------------------------------------------------------------
# train -c configs/voc12_2gpu.yaml --gt_path=/home/lianghuizhu/Project8_WSSS/hrnet/tools/output/pascal_ctx/seg_hrnet_w48_cls21_480x480_sgd_lr4e-3_wd1e-4_bs_16_epoch20_both_best/NewDL321/evaluation/checkpoint_epoch6/ --log_dir=Deeplabv2_pseudo_segmentation_labels_2gpu

# test
#-c
#configs/voc12.yaml
#-m
#data/models/Deeplabv2_pseudo_segmentation_labels_2gpu/deeplabv2_resnet101_msc/train_aug_cls/checkpoint_final.pth
#--log_dir=Deeplabv2_pseudo_segmentation_labels_2gpu_final
# DATASET=voc12_4gpu
# LOG_DIR=Deeplabv2_pseudo_segmentation_labels_4gpu
# GT_DIR=refined_pseudo_segmentation_labels
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py train -c configs/${DATASET}.yaml --gt_path=${GT_DIR} --log_dir=${LOG_DIR}