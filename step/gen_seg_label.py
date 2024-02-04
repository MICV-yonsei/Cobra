

import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import imageio

import torch
import torch.nn.functional as F

from tqdm import tqdm
import os
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels


def crf_inference_label(img, labels, t=10, n_labels=21, gt_prob=0.7):
    h, w = img.shape[:2]
    d = dcrf.DenseCRF2D(w, h, n_labels)
    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=2, compat=3)
    d.addPairwiseBilateral(sxy=45, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=4)
    q = d.inference(t)
    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)



def run(best_thres, args):
    base_path = os.path.join(args.root_out_dir, args.experiment_ver, args.sem_seg_out_npy_dir)
    save_path = os.path.join(args.root_out_dir, args.experiment_ver, args.sem_seg_out_png_dir)
    for npy in tqdm(os.listdir(base_path)):
        path = os.path.join(base_path, npy)
        cam_dict = np.load(path, allow_pickle=True).item()
        keys = cam_dict["keys"]
        rw_up = cam_dict["rw_up"]
        
        rw_up_max_by_class = np.max(rw_up, axis=(1, 2))
        rw_up_max_by_class_min = np.min(rw_up, axis=(1, 2))
        thres = 0.31 * rw_up_max_by_class + rw_up_max_by_class_min
        rw_up[rw_up < thres[:, None, None]] = 0 
        rw_up = torch.from_numpy(rw_up)
        
        rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=0)
        rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()
        raw_img_path = os.path.join(args.voc12_root, 'JPEGImages')
        raw_img = cv2.imread(f'{raw_img_path}/{npy[:-4]}.jpg', cv2.IMREAD_UNCHANGED)

        rw_pred = crf_inference_label(raw_img, rw_pred, n_labels=keys.shape[0])
        rw_pred = keys[rw_pred]
        cls_labels = rw_pred.astype(np.uint8)
        cls_labels[cls_labels == 255] = 0
        imageio.imsave(os.path.join(save_path, npy[:-4] + ".png"), rw_pred.astype(np.uint8))