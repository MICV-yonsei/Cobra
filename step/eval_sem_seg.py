
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import imageio

def run(args):
    print("-")
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
    base_path = os.path.join(args.root_out_dir, args.experiment_ver, args.sem_seg_out_npy_dir)
    save_path = os.path.join(args.root_out_dir, args.experiment_ver, args.sem_seg_out_png_dir)
    
    preds = []
    for id in dataset.ids:
        cls_labels = imageio.imread(os.path.join(save_path, id + '.png')).astype(np.uint8)
        cls_labels[cls_labels == 255] = 0
        preds.append(cls_labels.copy())

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator

    print(fp[0], fn[0])
    print(np.mean(fp[1:]), np.mean(fn[1:]))

    print({'iou': iou, 'miou': np.nanmean(iou)})