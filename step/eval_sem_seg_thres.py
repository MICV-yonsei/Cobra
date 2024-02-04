
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import imageio

import torch
import torch.nn.functional as F
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

def find_adaptive_thresholds(cam, num_classes, percentile=80):
    class_map = cam.numpy().astype(np.float32)  # Convert to numpy array of type float32
    return np.percentile(class_map, percentile)

def run(args):
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
    thres = args.sem_seg_bg_thres
    max_iou = -1
    theta = 0.2
    while True:
        print(f"try: {theta: .3f}")
        preds = []
        for id in dataset.ids:
            cam_dict = np.load(os.path.join(args.root_out_dir, args.experiment_ver, args.sem_seg_out_npy_dir, id + '.npy'), allow_pickle=True).item()
            keys = cam_dict['keys']
            rw_up = cam_dict['rw_up']

            feat_dict = np.load(os.path.join(args.root_out_dir, args.experiment_ver, args.where_cam_from, f"{img_name}.npy"), allow_pickle=True).item()
            feat = feat_dict['high_res_attn']
            thres = find_adaptive_thresholds(feat, len(feat), percentile=theta)
         
            rw_up_bg = F.pad(torch.from_numpy(rw_up), (0, 0, 0, 0, 1, 0), value=thres)     
            rw_pred = torch.argmax(rw_up_bg, dim=0)
            rw_pred = keys[rw_pred]
            cls_labels = rw_pred.astype(np.uint8)

            cls_labels[cls_labels == 255] = 0
            preds.append(cls_labels.copy())

        confusion = calc_semantic_segmentation_confusion(preds, labels)

        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        iou = gtjresj / denominator
        
        if np.nanmean(iou) < np.nanmean(max_iou):
            break
        else:
            print(f'miou: {np.nanmean(iou): .3f}\n')
            max_iou = iou
            theta += 0.01

    print(f'best threshold: {theta: .4f}')
    print({'iou': max_iou, 'miou': np.nanmean(max_iou)})
    
    preds = []
    for id in dataset.ids:
        cam_dict = np.load(os.path.join(args.root_out_dir, args.experiment_ver, args.sem_seg_out_npy_dir, id + '.npy'), allow_pickle=True).item()
        keys = cam_dict['keys']
        rw_up = cam_dict['rw_up']
        rw_up_max_by_class = np.max(rw_up, axis=(1, 2))
        rw_up_max_by_class_min = np.min(rw_up, axis=(1, 2))
        thres = theta * rw_up_max_by_class + rw_up_max_by_class_min
        rw_up[rw_up < thres[:, None, None]] = 0       
        rw_up_bg = F.pad(torch.from_numpy(rw_up), (0, 0, 0, 0, 1, 0), value=0)     
        rw_pred = torch.argmax(rw_up_bg, dim=0)

        raw_img_path = os.path.join(args.voc12_root, 'JPEGImages')
        raw_img = cv2.imread(f'{raw_img_path}/{id}.jpg', cv2.IMREAD_UNCHANGED)
        rw_pred = crf_inference_label(raw_img, rw_pred, n_labels=keys.shape[0])
        rw_pred = keys[rw_pred]
        cls_labels = rw_pred.astype(np.uint8)
        cls_labels[cls_labels == 255] = 0
        preds.append(cls_labels.copy())

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    print(iou)
    print({'with CRF miou': np.nanmean(iou)})
    return thres
