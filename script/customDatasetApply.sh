#!/bin/bash

CUSTOM_DIR=/home/thomas/Data/Cajun/Evaluation/Methods/superpoint_graph/custom
# python learning/custom_dataset.py --CUSTOM_PATH=$CUSTOM_DIR
# CUDA_VISIBLE_DEVICES=0 python learning/main_custom.py --dataset custom_dataset --CUSTOM_SET_PATH $CUSTOM_DIR --epochs -1 --lr_steps '[275,320]' --test_nth_epoch 50 --model_config 'gru_10_0,f_13' --ptn_nfeat_stn 14 --nworkers 2 --pc_attribs xyzrgbelpsvXYZ --odir "${CUSTOM_DIR}/results/pretrained" --resume RESUME
python learning/main_custom.py --dataset custom_dataset --CUSTOM_SET_PATH $CUSTOM_DIR --epochs -1 --lr_steps '[275,320]' --test_nth_epoch 50 --model_config 'gru_10_0,f_13' --ptn_nfeat_stn 14 --nworkers 2 --pc_attribs xyzrgbelpsvXYZ --odir "${CUSTOM_DIR}/results/pretrained" --resume RESUME
