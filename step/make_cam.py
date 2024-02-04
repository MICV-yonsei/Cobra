import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os

import voc12.dataloader
from misc import torchutils, imutils

cudnn.enabled = True


def _work(process_id, model_cnn, model_tran, dataset, args):
    print(process_id, end="/")
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)
    list_name = os.listdir(os.path.join(args.root_out_dir, args.experiment_ver, args.where_cam_from))
    with torch.no_grad(), cuda.device(process_id):

        model_cnn.cuda()
        model_tran.cuda()
        for iter, pack in enumerate(data_loader):

            img_name = pack["name"][0]
            img_npy = f'{img_name}.npy'
            if img_npy in list_name:
                pass
            else:
                label = pack["label"][0]
                size = pack["size"]  # [281, 500]

                strided_size = imutils.get_strided_size(size, 4)

                outputs_cnn = [model_cnn(img[0].cuda(non_blocking=True))[1:] for img in pack['img']] # CAK Branch
                outputs_tran = [model_tran(img[0].cuda(non_blocking=True), return_cam=True)[0][1:] for img in pack["img"]] # SAK Branch
                outputs_tran_attn = [model_tran(img[0].cuda(non_blocking=True), return_cam=True)[1] for img in pack["img"]] # SAK Branch

                strided_seed = torch.sum(torch.stack([F.interpolate(torch.unsqueeze(x, 0), strided_size, mode="bilinear", align_corners=False)[0] for x in outputs_tran]), 0)
                highres_seed = [F.interpolate(torch.unsqueeze(y, 1), size, mode="bilinear", align_corners=False) for y in outputs_tran]
                highres_cnn = [F.interpolate(torch.unsqueeze(y, 1), size, mode="bilinear", align_corners=False) for y in outputs_cnn]
                highres_seed = torch.sum(torch.stack(highres_seed, 0), 0)[:, 0]         
                highres_cnn = torch.sum(torch.stack(highres_cnn, 0), 0)[:, 0]
                highres_seed_attn_weight = [F.interpolate(torch.unsqueeze(y, 1), size, mode="bilinear", align_corners=False) for y in outputs_tran_attn]
                highres_seed_attn_weight = torch.sum(torch.stack(highres_seed_attn_weight, 0), 0).squeeze(0).squeeze(0)

                valid_cat = torch.nonzero(label)[:, 0]  # [class1, class2, ...]

                strided_seed = strided_seed[valid_cat]
                strided_seed /= F.adaptive_max_pool2d(strided_seed, (1, 1)) + 1e-5

                highres_seed = highres_seed[valid_cat]
                highres_seed /= F.adaptive_max_pool2d(highres_seed, (1, 1)) + 1e-5

                highres_cnn = highres_cnn[valid_cat]
                highres_cnn /= F.adaptive_max_pool2d(highres_cnn, (1, 1)) + 1e-5
                
                # Save cams
                np.save(os.path.join(args.root_out_dir, args.experiment_ver, args.where_cam_from, img_name + ".npy"), {"keys": valid_cat, "cam": strided_seed.cpu(), "high_res_seed": highres_seed.cpu(), "high_res_cnn": highres_cnn.cpu(), 'high_res_attn': highres_seed_attn_weight.cpu()})
                
            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print(f"{((5*iter+1)//(len(databin) // 20)):d}", end="/")


def run(args):
    # CAK Branch
    model_cnn = getattr(importlib.import_module(args.cam_network_branch1), 'CAM')()
    model_cnn.load_state_dict(torch.load(os.path.join(args.root_out_dir, args.experiment_ver, args.checkpoint_out_dir, f"ep19_cnn_checkpoint.pth")), strict=True)
    
    # SAK Branch
    model_tran = getattr(importlib.import_module(args.cam_network_branch2), "Net")(num_classes=args.num_classes)
    model_tran.load_state_dict(torch.load(os.path.join(args.root_out_dir, args.experiment_ver, args.checkpoint_out_dir, f"ep19_tf_checkpoint.pth")), strict=True)
    
    model_cnn.eval()
    model_tran.eval()

    n_gpus = torch.cuda.device_count()
    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list, voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)
    print("[ ", end="")
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model_cnn, model_tran, dataset, args), join=True)
    print("]")
    torch.cuda.empty_cache()