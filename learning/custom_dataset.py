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


def get_datasets(args, test_seed_offset=0):
    """build training and testing set"""
    
    #for a simple train/test organization
    trainset = ['train/' + f for f in os.listdir(args.ROOT_PATH + '/superpoint_graphs/train') if not os.path.isdir(f)]
    testset  = ['test/' + f for f in os.listdir(args.ROOT_PATH + '/superpoint_graphs/test') if not os.path.isdir(args.ROOT_PATH + '/superpoint_graphs/test/' + f)]
    
    # Load superpoints graphs
    testlist, trainlist, validlist = [], [], []
    for n in trainset:
        trainlist.append(spg.spg_reader(args, args.ROOT_PATH + '/superpoint_graphs/' + n, True))
        validlist.append(spg.spg_reader(args, args.ROOT_PATH + '/superpoint_graphs/' + n, True))
    for n in testset:
        testlist.append(spg.spg_reader(args, args.ROOT_PATH + '/superpoint_graphs/' + n, True))

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
    edge_feats = 0
    for attrib in args.edge_attribs.split(','):
        a = attrib.split('/')[0]
        if a in ['delta_avg', 'delta_std', 'xyz']:
            edge_feats += 3
        else:
            edge_feats += 1

    return {
        'node_feats': 11 if args.pc_attribs=='' else len(args.pc_attribs),
        'edge_feats': edge_feats,
        'classes': 10, #CHANGE TO YOUR NUMBER OF CLASS
        'inv_class_map': {0:'class_A', 1:'class_B'}, #etc...
    }

def preprocess_pointclouds(SEMA3D_PATH):
    """ Preprocesses data by splitting them by components and normalizing."""

    for n in ['train', 'test']:
        pathParsed = '{}/parsed/{}/'.format(SEMA3D_PATH, n)
        pathFeature = '{}/features/{}/'.format(SEMA3D_PATH, n)
        pathGraph = '{}/superpoint_graphs/{}/'.format(SEMA3D_PATH, n)
        if not os.path.exists(pathParsed):
            os.makedirs(pathParsed)
        random.seed(0)

        for fileName in os.listdir(pathGraph):
            print(fileName)
            if fileName.endswith(".h5"):
                ####################
                # Computation of all the features usefull for local descriptors computation made by PointNET
                ####################
                # This file is geometric features computed to SuperPoint construction
                # There are still usefull for local descriptors computation 
                geometricFeatureFile = h5py.File(pathFeature + fileName, 'r')
                xyz = geometricFeatureFile['xyz'][:]
                rgb = geometricFeatureFile['rgb'][:].astype(np.float)
                rgb = rgb/255.0 - 0.5
                # elpsv = np.stack([ featureFile['xyz'][:,2][:], featureFile['linearity'][:], featureFile['planarity'][:], featureFile['scattering'][:], featureFile['verticality'][:] ], axis=1)
                lpsv = geometricFeatureFile['geof'][:] 
                lpsv -= 0.5 #normalize

                # Compute elevation with simple Ransac from low points
                if n == "train":
                    e = xyz[:,2] / 4 - 0.5 # (4m rough guess)
                else :
                    low_points = ((xyz[:,2]-xyz[:,2].min() < 0.5)).nonzero()[0]
                    reg = RANSACRegressor(random_state=0).fit(xyz[low_points,:2], xyz[low_points,2])
                    e = xyz[:,2]-reg.predict(xyz[:,:2])
                    e /= np.max(np.abs(e),axis=0)
                    e *= 0.5

                # rescale to [-0.5,0.5]; keep xyz
                #warning - to use the trained model, make sure the elevation is comparable
                #to the set they were trained on
                #i.e. ~0 for roads and ~0.2-0.3 for builings for sema3d
                # and -0.5 for floor and 0.5 for ceiling for s3dis
                
                # elpsv[:,0] /= 100 # (rough guess) #adapt 
                # elpsv[:,1:] -= 0.5
                # rgb = rgb/255.0 - 0.5

                # Add some new features, why not ?
                room_center = xyz[:,[0,1]].mean(0) #compute distance to room center, useful to detect walls and doors
                distance_to_center = np.sqrt(((xyz[:,[0,1]]-room_center)**2).sum(1))
                distance_to_center = (distance_to_center - distance_to_center.mean())/distance_to_center.std()
                ma, mi = np.max(xyz,axis=0,keepdims=True), np.min(xyz,axis=0,keepdims=True)
                xyzn = (xyz - mi) / (ma - mi + 1e-8)   # as in PointNet ("normalized location as to the room (from 0 to 1)")

                parsedData = np.concatenate([xyz, rgb, e[:,np.newaxis], lpsv, xyzn, distance_to_center[:,None]], axis=1)

                # Old features
                # parsedData = np.concatenate([xyz, rgb, elpsv], axis=1)

                graphFile = h5py.File(pathGraph + fileName, 'r')
                nbComponents = len(graphFile['components'].keys())

                with h5py.File(pathParsed + fileName, 'w') as parsedFile:
                    for components in range(nbComponents):
                        idx = graphFile['components/{:d}'.format(components)][:].flatten()
                        if idx.size > 10000: # trim extra large segments, just for speed-up of loading time
                            ii = random.sample(range(idx.size), k=10000)
                            idx = idx[ii]
                        parsedFile.create_dataset(name='{:d}'.format(components), data=parsedData[idx,...])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
    parser.add_argument('--CUSTOM_PATH', default='datasets/custom_set')
    args = parser.parse_args()
    preprocess_pointclouds(args.CUSTOM_PATH)


