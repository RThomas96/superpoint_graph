#!/bin/bash

CUSTOM_DIR=/home/thomas/Data/Cajun/Evaluation/Methods/superpoint_graph/custom
python partition/visualize_result.py --ROOT_PATH $CUSTOM_DIR --file_path $1 --file_res $2
