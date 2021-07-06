import argparse
import os

from tool import pyutils
from tool.torchutils import how_many_gpus


def str2bool(v):
    if v.lower() in ("yes", "true", 't', 'y', '1'):
        return True
    elif v.lower() in ("no", "false", 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # experiment about
    parser.add_argument("--name", type=str, default="dgcn")
    parser.add_argument("--gpu_nums", type=int, default=1)
    parser.add_argument("--num_workers", default=os.cpu_count() // 2, type=int)

    # dataset
    parser.add_argument("--voc12_data_dir", type=str, default="/data/zhulianghui/data/VOC2012/VOCdevkit/VOC2012/",
                        help="Path to the directory containing the PASCAL VOC12 dataset.")
    parser.add_argument("--voc12_data_list", type=str, default="./datainfo/list/input_list.txt",
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--voc12_eval_list", type=str, default="./datainfo/list/val.txt")
    parser.add_argument("--voc12_cues_dir", type=str, default="./datainfo/")
    parser.add_argument("--voc12_cues_name", type=str, default="localization_cues-sal.pickle")
    parser.add_argument("--num_classes", type=int, default=21,
                        help="Number of classes to predict (including background).")

    # training params
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--epoch", type=int, default=1,
                        help="Training epoch.")
    parser.add_argument("--learning_rate", type=float, default=75e-5,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num_steps", type=int, default=10582,
                        help="Number of training steps, here we use VOC12's train-aug imgs' num.")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--gamma", type=float, default=0.3,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random_seed", type=int, default=504,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="process rank on node.")
    parser.add_argument("--lr_scheduler", type=str, default="step",
                        help="decide to use which optimizer.")

    # model
    parser.add_argument("--model", type=str, default="models.seg_hrnet",
                        help="decide which backbone.")
    parser.add_argument("--restore_from", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore_epoch", type=int, default=None,
                        help="Where restore model parameters from.")

    # data process
    parser.add_argument("--ignore_label", type=int, default=255,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input_size", type=str, default="512,512",
                        help="Comma-separated string with height and width of images.")

    # Output Path
    parser.add_argument("--snapshot_out_dir", type=str, default="./results/snapshots/",
                        help="Where to save snapshots of the model.")
    parser.add_argument("--evaluation_out_dir", type=str, default="./results/evaluations/",
                        help="Where to save the evaluation of the model output")
    parser.add_argument("--logger_out_dir", type=str, default="./results/logger/",
                        help="Where to save the log of the logger output")

    # make pred about
    parser.add_argument("--multi_scales", default=(0.75, 1.0, 1.25),
                        help="Multi-scale inferences")
    parser.add_argument("--eval_with_crf", default=True,
                        help="If you want to eval image with crf post-process.")

    # Step
    parser.add_argument("--train_dgcn_pass", default=False, type=str2bool)
    parser.add_argument("--make_pred_pass", default=True, type=str2bool)
    parser.add_argument("--eval_pred_pass", default=True, type=str2bool)

    args = parser.parse_args()

    args.gpu_nums = how_many_gpus()

    # path about
    os.makedirs("results", exist_ok=True)
    os.makedirs(args.snapshot_out_dir, exist_ok=True)
    os.makedirs(args.snapshot_out_dir + args.name, exist_ok=True)
    os.makedirs(args.evaluation_out_dir, exist_ok=True)
    os.makedirs(args.evaluation_out_dir + args.name, exist_ok=True)
    os.makedirs(args.logger_out_dir, exist_ok=True)
    os.makedirs(args.logger_out_dir + args.name, exist_ok=True)

    args.snapshot_out_dir = args.snapshot_out_dir + args.name
    args.evaluation_out_dir = args.evaluation_out_dir + args.name
    args.logger_out_dir = args.logger_out_dir + args.name

    pyutils.Logger(args.logger_out_dir + args.name + ".log")
    print(vars(args))
    print("print args done!")

    if args.train_dgcn_pass is True:
        import step.train_dgcn

        timer = pyutils.Timer("step.train_dgcn:")
        step.train_dgcn.run(args)

    if args.make_pred_pass is True:
        import step.make_pred

        timer = pyutils.Timer("step.make_pred:")
        step.make_pred.run(args)

    if args.eval_pred_pass is True:
        import step.eval_pred

        timer = pyutils.Timer("step.eval_pred:")
        step.eval_pred.run(args)
