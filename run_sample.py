import argparse
import os
from misc import pyutils
import torch
import numpy as np
import random


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_args_parser():
    parser = argparse.ArgumentParser("WSSS", add_help=False)
    parser.add_argument("--seed", default=2022, type=int, help="hbd")
    parser.add_argument("--num_classes", default=21, type=int)

    # GPU NUMBER
    parser.add_argument("--gpu_number", type=int)

    # Experiment Version
    parser.add_argument("--experiment_ver", type=str, help="experiment name", required=True)

    # Environment
    parser.add_argument("--num_workers", default=os.cpu_count() // 2, type=int)
    parser.add_argument("--voc12_root", type=str, help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.", required=True)

    # Output Path
    parser.add_argument("--log_name", default="train_eval_log", type=str)
    parser.add_argument("--root_out_dir", type=str, required=True)
    parser.add_argument("--checkpoint_out_dir", type=str, default="checkpoints")
    parser.add_argument("--where_cam_from", default="transformer", type=str)
    parser.add_argument("--cam1_weights_name", default="cnn_checkpoint.pth", type=str)
    parser.add_argument("--cam2_weights_name", default="tf_checkpoint.pth", type=str)
    parser.add_argument("--irn_weights_name", default="irn_checkpoint.pth", type=str)
    parser.add_argument("--ir_label_out_dir", default="ir_label", type=str)
    parser.add_argument("--sem_seg_out_png_dir", default="sem_seg_out_png", type=str)
    parser.add_argument("--sem_seg_out_npy_dir", default="sem_seg_npy", type=str)

    # Hyperparams
    parser.add_argument("--cam_batch_size", default=32, type=int)
    parser.add_argument("--cam_num_epoches", default=20, type=int)
    parser.add_argument("--top_bot_k", nargs="+", type=int, required=True, help="[topV1, botV1, topV2, botV2]")
    parser.add_argument("--phi_c2t", type=float, required=True)
    parser.add_argument("--phi_t2c", type=float, required=True)
    parser.add_argument("--tau", type=float, default=0.1)

    # Dataset
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str, help="voc12/train_aug.txt to train a fully supervised model, " "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    parser.add_argument("--chainer_eval_set", default="train", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network_branch1", default="net.resnet50_cam", type=str)
    parser.add_argument("--cam_network_branch2", default="net.deit_cam", type=str)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.1, type=float)
    parser.add_argument("--cam_scales", default=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0), help="Multi-scale inferences")
    parser.add_argument("--cam_lambda", default=0.1, type=float)
    
    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.30, type=float)
    parser.add_argument("--conf_bg_thres", default=0.25, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="net.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=32, type=int)
    parser.add_argument("--irn_num_epoches", default=4, type=int) 
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8, help="Hyper-parameter that controls the number of random walk iterations," "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--sem_seg_bg_thres", default=0.25, type=float)

    # Step
    parser.add_argument("--train_cam_pass", default=1, type=int)
    parser.add_argument("--make_cam_pass", default=1, type=int)
    parser.add_argument("--eval_cam_pass", default=1, type=int)
    parser.add_argument("--cam_to_ir_label_pass", default=1, type=int)
    parser.add_argument("--train_irn_pass", default=1, type=int)
    parser.add_argument("--make_sem_seg_npy_pass", default=1, type=int)
    parser.add_argument("--eval_sem_seg_pass_npy", default=1, type=int) 
    parser.add_argument("--make_sem_seg_pass", default=0, type=int)
    parser.add_argument("--eval_sem_seg_pass", default=0, type=int)  
    
    parser.add_argument("--gen_seg_label", help="When use gonna make .png file not directly but from .npy file", default=1, type=int)

    return parser


def main(args):
    # When Using GPU individually
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.makedirs(f"{args.root_out_dir}/{args.experiment_ver}/{args.where_cam_from}", exist_ok=True)
    os.makedirs(f"{args.root_out_dir}/{args.experiment_ver}/{args.ir_label_out_dir}", exist_ok=True)
    os.makedirs(f"{args.root_out_dir}/{args.experiment_ver}/{args.sem_seg_out_npy_dir}", exist_ok=True)
    os.makedirs(f"{args.root_out_dir}/{args.experiment_ver}/{args.sem_seg_out_png_dir}", exist_ok=True)
    os.makedirs(f"{args.root_out_dir}/{args.experiment_ver}/{args.checkpoint_out_dir}", exist_ok=True)

    pyutils.Logger(os.path.join(args.root_out_dir, args.experiment_ver, args.log_name + ".log"))
    print(*vars(args).items(), sep="\n")

    # fixing seed
    seed_everything(args.seed)
    if args.train_cam_pass:
        import step.train_cam
        timer = pyutils.Timer("\n\nstep.train_cam:\n\n")
        step.train_cam.run(args)

    # Step
    if args.make_cam_pass:
        import step.make_cam
        timer = pyutils.Timer("\n\nstep.make_cam:\n\n")
        step.make_cam.run(args)

    if args.eval_cam_pass:
        import step.eval_cam
        timer = pyutils.Timer("\n\nstep.eval_cam:\n\n")
        step.eval_cam.run(args)

    
    # IRNet start
    if args.cam_to_ir_label_pass:
        import step.cam_to_ir_label
        timer = pyutils.Timer("\n\nstep.cam_to_ir_label:\n\n")
        step.cam_to_ir_label.run(args)

    if args.train_irn_pass:
        import step.train_irn

        timer = pyutils.Timer("\n\nstep.train_irn:\n\n")
        step.train_irn.run(args)

    # make mask
    if args.make_sem_seg_npy_pass:
        import step.make_sem_seg_labels_npy

        timer = pyutils.Timer("\n\nstep.make_sem_seg_labels_npy:\n\n")
        step.make_sem_seg_labels_npy.run(args)
    
    # threshold에 맞춰서 npy 값 찾기
    best_thres = None  # Exploring Best Threshold
    if args.eval_sem_seg_pass_npy:
        import step.eval_sem_seg_thres
        timer = pyutils.Timer("\n\nstep.eval_sem_seg_thres:\n\n")
        best_thres = step.eval_sem_seg_thres.run(args)

    if best_thres != None and args.gen_seg_label:
        import step.gen_seg_label
        timer = pyutils.Timer("\n\nstep.eval_sem_seg_thres:\n\n")
        best_thres = step.gen_seg_label.run(best_thres, args)

    if args.make_sem_seg_pass is True:  # Generate directly .png file about sem_seg
        import step.make_sem_seg_labels
        timer = pyutils.Timer("\n\nstep.make_sem_seg_labels:\n\n")
        step.make_sem_seg_labels.run(args)

    if args.eval_sem_seg_pass is True:
        import step.eval_sem_seg
        timer = pyutils.Timer("\n\nstep.eval_sem_seg:\n\n")
        step.eval_sem_seg.run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)