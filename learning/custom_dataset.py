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


def get_datasets(args, pathManager, i, test_seed_offset=0):
    """build training and testing set"""
    
    dataset = pathManager.allDataDataset[i]

    # Load superpoints graphs
    testlist, trainlist, validlist = [], [], []
    for n in dataset['train']:
        trainlist.append(spg.spg_reader(args, pathManager.getFilesFromDataset(n)[5]))
    for n in dataset['test']:
        testlist.append(spg.spg_reader(args, pathManager.getFilesFromDataset(n)[5]))
    for n in dataset['validation']:
        validlist.append(spg.spg_reader(args, pathManager.getFilesFromDataset(n)[5]))
    if len(validlist) == 0:
        validlist = trainlist

    # Normalize edge features
    if args.spg_attribs01:
       trainlist, testlist, validlist, scaler = spg.scaler01(trainlist, testlist, validlist=validlist)

    " functools.partial(arg1, arg2, ...) = return the function : arg1(arg2, arg3, ...) "
    " ListDataset(arg1, arg2) = dataset wich load data listed on arg1, and loaded with the function arg2. " 
    "                           In our case arg2 is a functools.partial with spg.loader "
    "                           And this function arg2 with load data given by arg1, in our case arg1 is data returned by spg_to_igraph"
    return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(spg.loader, train=True, args=args, pathManager=pathManager)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(spg.loader, train=False, args=args, pathManager=pathManager, test_seed_offset=test_seed_offset)) ,\
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in validlist],
                                    functools.partial(spg.loader, train=False, args=args, pathManager=pathManager, test_seed_offset=test_seed_offset)) ,\
            scaler

def get_info(args):
    colors = ColorLabelManager(args.colorCode)

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

    if loss == 'equal':
        weights = np.ones((colors.nbColor,),dtype='f4')
        weights = torch.from_numpy(weights).cuda() if args.cuda else torch.from_numpy(weights)
    else:
        print("Hard coded loss weights !")
        #weights = np.array([0.69422756, 0.84267807, 0.56590259, 0.99259166, 0.99582052, 0.99747999, 0.99924599, 0.99910761, 0.98240414, 0.9561125,  0.98127003], dtype='f4') # Weights on points
        weights = np.array([0.74550129, 0.81159779, 0.61670704, 0.98157257, 0.98767797, 0.99452492, 0.99897419, 0.99616868, 0.97169765, 0.9522197,  0.97037522], dtype='f4')  # Weight on spp for regStrength 0.05
        #weights = np.array([0.74550129, 0.81159779, 0.61670704, 1, 1], dtype='f4')  # Weight on spp for regStrength 0.05

        #print("Loss weights not implemented yet !")
        #weights = np.ones((colors.nbColor,),dtype='f4')
        weights = torch.from_numpy(weights).cuda() if args.cuda else torch.from_numpy(weights)
    return {
        'node_feats': 11 if args.pc_attribs=='' else len(args.pc_attribs),
        'edge_feats': edge_feats,
        'class_weights': weights,
        'classes': colors.nbColor,
        #'inv_class_map': colors.nameDict[1:]
        'inv_class_map': {x: colors.label2Name[x] for x in colors.label2Name if x not in [0]}
    }
