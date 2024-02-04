#!/bin/bash
#source ~/.bachrc
title="best"
pascal_dataset_path="/media/dataset/VOCdevkit/VOC2012"

python run_sample.py \
--voc12_root ${pascal_dataset_path} \
--root_out_dir ./results \
--experiment_ver ${title} \
--top_bot_k 20 5 5 20 \
--where_cam_from transformer19 \
--phi_c2t 0.1 \
--phi_t2c 0.1
