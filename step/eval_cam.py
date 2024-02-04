import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion

def find_adaptive_thresholds(cam, num_classes, percentile=80):
    class_map = cam.numpy().astype(np.float32)  # Convert to numpy array of type float32
    return np.percentile(class_map, percentile)

def run(args):
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

    theta = 70
    max_iou = -1
    while True:
        print(f"try: {theta: .3f}")
        preds = []
        for id in dataset.ids:
            #
            cam_dict = np.load(os.path.join(args.root_out_dir, args.experiment_ver, args.where_cam_from, f"{id}.npy"),  allow_pickle=True).item()
            cams_cnn = np.array(cam_dict['high_res_cnn'])
            cams_tran = np.array(cam_dict['high_res_tran'])
            
            cams = np.concatenate((np.expand_dims(cams_tran, 0), np.expand_dims(cams_cnn, 0)), axis=0)
            cams = np.max(cams, axis=0)
            cams = (cams + cams_tran)/2
            #
            feat_dict = np.load(os.path.join(args.root_out_dir, args.experiment_ver, args.where_cam_from, f"{id}.npy"),  allow_pickle=True).item()
            feat = feat_dict['high_res_attn']
            # 
            thres = find_adaptive_thresholds(feat, len(feat), percentile=theta)
            
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=thres)
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels]
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
            theta += 5
            
    print(f'best threshold: {theta: .3f}')
    print({'iou': max_iou, 'miou': np.nanmean(max_iou)})
