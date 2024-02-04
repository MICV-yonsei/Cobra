
import importlib
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.backends import cudnn
from torch.utils.data import DataLoader
import os
import loss_modules
import voc12.dataloader
from misc import pyutils, torchutils
import sys
cudnn.enabled = True

def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:
            ap_i = average_precision_score(y_true[i], y_pred[i])
            AP.append(ap_i)
    return AP


def validate(model1, model2,  data_loader):
    print('\nValidating', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss_logit_CNN', 'loss_logit_TF')

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']
            label = pack['label'].cuda(non_blocking=True)

            # cam_cnn, _ = model1(img)  # CAK Branch
            pred_cnn, _1, _2 = model1(img)  # CAK Branch
            cam_tf = model2(img)  # SAK Branch

            cam_tf = cam_tf[0]
            # loss_logit_CNN = F.multilabel_soft_margin_loss(torchutils.gap2d(cam_cnn), label)
            loss_logit_CNN = F.multilabel_soft_margin_loss(pred_cnn[:, 1:], label)
            loss_logit_TF = F.multilabel_soft_margin_loss(torchutils.gap2d(cam_tf[:, 1:]), label)

            val_loss_meter.add({'loss_logit_CNN': loss_logit_CNN.item()})
            val_loss_meter.add({'loss_logit_TF': loss_logit_TF.item()})

            mAP_CNN = []
            mAP_TF = []
            
            # output = torch.sigmoid(torchutils.gap2d(cam_cnn))
            output = torch.sigmoid(pred_cnn[:, 1:])
            mAP_list = compute_mAP(label, output)
            mAP_CNN = mAP_CNN + mAP_list
            
            output1 = torch.sigmoid(torchutils.gap2d(cam_tf[:, 1:]))
            mAP_list_ = compute_mAP(label, output1)
            mAP_TF = mAP_TF + mAP_list_

    mAP_1 = np.mean(mAP_CNN)
    mAP_2 = np.mean(mAP_TF)
    print(f"mAP_CNN: {mAP_1:.3f} {mAP_2:.3f}")
    print(f"loss: loss_logit_CNN({val_loss_meter.pop('loss_logit_CNN'):.4f}) loss_logit_TF({val_loss_meter.pop('loss_logit_TF'):.4f})")
    print("******************************")
    return


def run(args):
    model1 = getattr(importlib.import_module(args.cam_network_branch1), 'Net')(avgpool_thres = 0.35) # CAK Branch
    model2 = getattr(importlib.import_module(args.cam_network_branch2), 'Net')(pretrained=True, num_classes=args.num_classes) # SAK Branch

    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root, resize_long=(320, 640), hor_flip=True, crop_size=224, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root, crop_size=224)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups1 = model1.trainable_parameters()
    optimizer1 = torchutils.PolyOptimizer([
        {'params': param_groups1[0], 'lr': args.cam_learning_rate,'weight_decay': args.cam_weight_decay},
        {'params': param_groups1[1], 'lr': 10*args.cam_learning_rate,'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    param_groups2 = model2.trainable_parameters()
    optimizer2 = torchutils.PolyOptimizer([
        {'params': param_groups2[0], 'lr': args.cam_learning_rate,'weight_decay': args.cam_weight_decay},
        {'params': param_groups2[1], 'lr': 10*args.cam_learning_rate,'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model1 = nn.DataParallel(model1).cuda()
    model2 = nn.DataParallel(model2).cuda()
    
    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()

    for ep in range(args.cam_num_epoches):
        model1.train()
        model2.train()
        print("******************************")
        for step, pack in enumerate(train_data_loader):
            img = pack['img']
            label = pack['label']
            
            # Add bg label(dummy)
            bg_score = torch.ones((label.shape[0], 1))
            label_bg = torch.cat((bg_score, label), dim=1).cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)

            # Outputs of models
            # cam_cnn, embed_cnn = model1(img)  # CAK Branch
            cam_cnn, embed_cnn = model1(img)  # CAK Branch
            cam_tf, attn_weights, embed_tf = model2(img)  # SAK Branch
            
            # Summation of attn_weight by layer
            attn_weights = attn_weights.sum(1)
            
            # Normalize embeddings
            embed_tf = F.normalize(embed_tf.flatten(start_dim=-2), dim=-2)
            embed_cnn = F.normalize(embed_cnn.flatten(start_dim=-2), dim=-2)

            # 1. Classification Loss (Logit Loss)
            cam_cnn_cls = cam_cnn[:, 1:]
            cam_tf_cls = cam_tf[:, 1:]
            label = label.cuda(non_blocking=True)
            loss_logit_CNN = F.multilabel_soft_margin_loss(torchutils.gap2d(cam_cnn_cls), label)
            loss_logit_TF = F.multilabel_soft_margin_loss(torchutils.gap2d(cam_tf_cls), label)

            # 2. Inter CAM Loss - L1 Loss
            cam_cnn_224 = F.interpolate(cam_cnn, size=(224, 224), mode='bilinear', align_corners=True)
            cam_tf_224 = F.interpolate(cam_tf, size=(224, 224), mode='bilinear', align_corners=True)
            loss_interCAM1 = torch.mean(torch.abs(cam_cnn_224 - cam_tf_224))
            loss_interCAM2 = loss_modules.loss_CAM(cam_cnn_224, cam_tf_224, label)
            
            # 3. CAP, SAP Loss
            loss_CAP = loss_modules.cll_v1(args, cam_cnn, embed_tf, label_bg).reshape([])
            loss_SAP = loss_modules.cll_v2(args, attn_weights, embed_cnn).reshape([])

            loss = loss_logit_CNN + loss_logit_TF
            if ep >= 5: 
                loss += 0.1 * (loss_interCAM1) # + loss_interCAM2)
                loss += (args.phi_c2t * loss_CAP) + (args.phi_t2c * loss_SAP)

            avg_meter.add({'loss_SAP': loss_SAP.item()})
            avg_meter.add({'loss_logit_CNN': loss_logit_CNN.item()})
            avg_meter.add({'loss_logit_TF': loss_logit_TF.item()})
            avg_meter.add({'loss_interCAM1': loss_interCAM1.item()})
            avg_meter.add({'loss_interCAM2': loss_interCAM2.item()})
            avg_meter.add({'loss_CAP': loss_CAP.item()})
            avg_meter.add({'loss': loss.item()})
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            if (optimizer1.global_step-1) % 100 == 0:
                timer.update_progress(optimizer1.global_step / max_step)
                print(f"# EPOCH : {ep} || exp_ver : {args.experiment_ver}")
                print(f"step:{optimizer1.global_step - 1}/{max_step}", flush=True)
                print(f"\nloss: loss_logit_CNN({avg_meter.pop('loss_logit_CNN'):.3f}), loss_logit_TF({avg_meter.pop('loss_logit_TF'):.3f}), loss_interCAM1({avg_meter.pop('loss_interCAM1'):.3f}), loss_interCAM2({avg_meter.pop('loss_interCAM2'):.3f})", flush=True)
                print(f"loss_cne: loss_CAP({avg_meter.pop('loss_CAP'): .3f}), loss_SAP({avg_meter.pop('loss_SAP'): .3f}), Total loss({avg_meter.pop('loss'): .3f})")
                print(f"imps: {((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()): .3f}", flush=True)
                print(f"lr: {(optimizer1.param_groups[0]['lr']): .3f}", flush=True)
                print(f"etc:{(timer.str_estimated_complete())}",flush=True)
                print("..............................")
        else:   
            model1.eval()
            model2.eval()
            validate(model1, model2, val_data_loader)
            timer.reset_stage()
        
        #save checkpoint per epoch
        torch.save(model1.module.state_dict(), os.path.join(args.root_out_dir, args.experiment_ver, args.checkpoint_out_dir, f"ep{ep}_{args.cam1_weights_name}"))
        torch.save(model2.module.state_dict(), os.path.join(args.root_out_dir, args.experiment_ver, args.checkpoint_out_dir, f"ep{ep}_{args.cam2_weights_name}"))
        
    torch.cuda.empty_cache()
