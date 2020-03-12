#!/bin/bash

CUSTOM_DIR=/home/thomas/Data/Cajun/Evaluation/Methods/superpoint_graph/custom
# previous voxel_width = 0.03
python partition/partition_custom.py --ROOT_PATH $CUSTOM_DIR --voxel_width 0.01 --reg_strength 0.03 --k_nn_geof 99
