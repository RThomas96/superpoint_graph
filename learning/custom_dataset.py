#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:16:14 2018

@author: landrieuloic
""""""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
    Template file for processing custome datasets
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import os
import functools
import torch
import torchnet as tnt
import h5py
import spg
from sklearn.linear_model import RANSACRegressor

import sys
sys.path.append("./utils")
from colorLabelManager import ColorLabelManager


def get_datasets(args, test_seed_offset=0):
    """build training and testing set"""
    
    #for a simple train/test organization
    trainset = ['train/' + f for f in os.listdir(args.CUSTOM_SET_PATH + '/superpoint_graphs/train') if not os.path.isdir(f)]
    testset  = ['test/' + f for f in os.listdir(args.CUSTOM_SET_PATH + '/superpoint_graphs/test') if not os.path.isdir(args.CUSTOM_SET_PATH + '/superpoint_graphs/test/' + f)]
    
    # Load superpoints graphs
    testlist, trainlist, validlist = [], [], []
    for n in trainset:
        trainlist.append(spg.spg_reader(args, args.CUSTOM_SET_PATH + '/superpoint_graphs/' + n, True))
        validlist.append(spg.spg_reader(args, args.CUSTOM_SET_PATH + '/superpoint_graphs/' + n, True))
    for n in testset:
        testlist.append(spg.spg_reader(args, args.CUSTOM_SET_PATH + '/superpoint_graphs/' + n, True))

    # Normalize edge features
    if args.spg_attribs01:
       trainlist, testlist, validlist, scaler = spg.scaler01(trainlist, testlist, validlist=validlist)

    " functools.partial(arg1, arg2, ...) = return the function : arg1(arg2, arg3, ...) "
    " ListDataset(arg1, arg2) = dataset wich load data listed on arg1, and loaded with the function arg2. " 
    "                           In our case arg2 is a functools.partial with spg.loader "
    "                           And this function arg2 with load data given by arg1, in our case arg1 is data returned by spg_to_igraph"
    return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(spg.loader, train=True, args=args, db_path=args.CUSTOM_SET_PATH)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.CUSTOM_SET_PATH, test_seed_offset=test_seed_offset)) ,\
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in validlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.CUSTOM_SET_PATH, test_seed_offset=test_seed_offset)) ,\
            scaler

def get_info(args):
    colors = ColorLabelManager()

    edge_feats = 0
    try:
        for attrib in args.edge_attribs.split(','):
            a = attrib.split('/')[0]
            if a in ['delta_avg', 'delta_std', 'xyz']:
                edge_feats += 3
            else:
                edge_feats += 1
    # If it is used for supervized_partition, there is no edge_feats attribute
    except AttributeError:
        edge_feats = 0

    loss = 'none'
    try:
        loss = args.loss_weights
    except AttributeError:
        loss = args.loss_weight

    if loss == 'none':
        weights = np.ones((colors.nbColor,),dtype='f4')
    else:
        print("Loss weights not implemented yet !")
        weights = np.ones((colors.nbColor,),dtype='f4')
    weights = torch.from_numpy(weights).cuda() if args.cuda else torch.from_numpy(weights)
    return {
        'node_feats': 11 if args.pc_attribs=='' else len(args.pc_attribs),
        'edge_feats': edge_feats,
        'class_weights': weights,
        'classes': colors.nbColor,
        'inv_class_map': colors.nameDict
    }