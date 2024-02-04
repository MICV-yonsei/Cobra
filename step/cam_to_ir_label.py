import os
import numpy as np
import imageio

from torch import multiprocessing
from torch.utils.data import DataLoader

import voc12.dataloader
from misc import torchutils, imutils
import torch

def round_to_nearest_05(n):
    return round(n * 20) / 20

def get_bounds(n):
    rounded = round_to_nearest_05(n)
    if n == rounded:
        return (rounded - 0.05, rounded + 0.05)
    elif n < rounded:
        return (rounded - 0.05, rounded)
    else:
        return (rounded, rounded + 0.05)

def find_adaptive_thresholds(cam, num_classes, percentile=80):
    class_map = cam.numpy().astype(np.float32)  # Convert to numpy array of type float32
    return np.percentile(class_map, percentile)

def _work(process_id, infer_dataset, args):

    databin = infer_dataset[process_id]
    infer_data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)

    for iter, pack in enumerate(infer_data_loader):
        img_name = voc12.dataloader.decode_int_filename(pack["name"][0])
        img = pack["img"][0].numpy()
        cam_dict = np.load(os.path.join(args.root_out_dir, args.experiment_ver, args.where_cam_from, f"{img_name}.npy"), allow_pickle=True).item()
        
        cams_cnn = cam_dict["high_res_cnn"]
        cams_tran = cam_dict["high_res_seed"]
        cams = torch.cat((cams_cnn.unsqueeze(0), cams_tran.unsqueeze(0)), 0)
        cams, _ = torch.max(cams, 0)
        cams = (cams + cams_tran)/2

        feat_dict = np.load(os.path.join(args.root_out_dir, args.experiment_ver, args.where_cam_from, f"{id}.npy"),  allow_pickle=True).item()
        feat = feat_dict['high_res_attn']
        thres = find_adaptive_thresholds(feat, len(feat), percentile=93)
        
        keys = np.pad(cam_dict["keys"] + 1, (1, 0), mode="constant")
        
        # 1. find confident fg & bg
        ir_bg_thres, _ = get_bounds(thres)
        ir_fg_thres = ir_bg_thres + 0.5
        
        fg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode="constant", constant_values=ir_fg_thres)
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0])
        fg_conf = keys[pred]

        bg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode="constant", constant_values=ir_bg_thres)
        bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, bg_conf_cam, n_labels=keys.shape[0])
        bg_conf = keys[pred]

        # 2. combine confident fg & bg
        conf = fg_conf.copy()
        conf[fg_conf == 0] = 255
        conf[bg_conf + fg_conf == 0] = 0

        imageio.imwrite(os.path.join(args.root_out_dir, args.experiment_ver, args.ir_label_out_dir, f"{img_name}.png"), conf.astype(np.uint8))
        
        if iter % (len(databin) // 20) == 0:
            print(f"{(5 * iter + 1) // (len(databin) // 20)} ", end="")


def run(args):
    dataset = voc12.dataloader.VOC12ImageDataset(args.train_list, voc12_root=args.voc12_root, img_normal=None, to_torch=False)
    dataset = torchutils.split_dataset(dataset, args.num_workers)

    print("[ ", end="")
    multiprocessing.spawn(_work, nprocs=args.num_workers, args=(dataset, args), join=True)
    print("]")
