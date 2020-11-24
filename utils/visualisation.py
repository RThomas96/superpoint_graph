import os
import sys
import random
import numpy as np

import cloudIO as io
from colorLabelManager import ColorLabelManager

#from numpy import genfromtxt
#import laspy

#import glob
#import pandas as pd
#from sklearn.neighbors import NearestNeighbors
#from partition.ply_c import libply_c
#from sklearn.decomposition import PCA
#import colorsys

#------------------------------------------------------------------------------
def writeTransition(filename, xyz, edge_source, is_transition, dataType = "laz"):
    """write a ply with random colors for each components for only a specific label"""

    red=np.array([255, 0, 0])
    blue=np.array([0, 0, 255])

    transitions = np.array([is_transition==1]).nonzero()[1]
    transitionsIdx = edge_source[transitions] 

    color = np.zeros(xyz.shape)
    color[:] = blue
    color[transitionsIdx] = red
    io.write_file(filename, xyz, color)
#------------------------------------------------------------------------------
def writePartitionFilter(filename, xyz, components, labels, filterLabel, dataType="laz"):
    """write a ply with random colors for each components for only a specific label"""

    labelOfEachSpp = labels.argmax(1)
    idxOfFilteredSpp = np.argwhere(labelOfEachSpp==int(filterLabel)).flatten()
    components=components[idxOfFilteredSpp]

    random_color = lambda: random.randint(0, 255)
    color = np.zeros(xyz.shape)
    for i_com in range(0, len(components)):
        color[components[i_com], :] = [random_color(), random_color()
        , random_color()]

    io.write_file(filename, xyz, color, dataType)
#------------------------------------------------------------------------------
def writePartition(filename, xyz, components, dataType="laz"):
    """write a ply with random colors for each components"""
    random_color = lambda: random.randint(0, 255)
    color = np.zeros(xyz.shape)
    for i_com in range(0, len(components)):
        color[components[i_com], :] = [random_color(), random_color(), random_color()]

    io.write_file(filename, xyz, color, dataType)
#------------------------------------------------------------------------------
def writeGeof(filename, xyz, geof, dataType="laz"):
    color = np.array(255 * geof[:, [0, 1, 3]], dtype='uint8')
    io.write_file(filename, xyz, color, dataType)
#------------------------------------------------------------------------------
def writeGeofstd(filename, xyz, geof, components, in_component, dataType="laz"):
    geofpt = np.copy(components)   
    geofpt = [geof[x] for x in components]
    std1 = [np.std(x[:, 0]) for x in geofpt] 
    std2 = [np.std(x[:, 1]) for x in geofpt] 
    std3 = [np.std(x[:, 2]) for x in geofpt] 
    values = np.array([std1, std2, std3]).T
    componentsColor = np.array(255 * values, dtype='uint8')

    color = componentsColor[np.array(in_component)]

    io.write_file(filename, xyz, color, dataType)
#------------------------------------------------------------------------------
def writePrediction(filename, xyz, prediction, dataType="laz"):
    """write a ply with colors for each class"""
    colorLabelManager = ColorLabelManager()
    n_label = colorLabelManager.nbColor
    if len(prediction.shape) > 1 and prediction.shape[1] > 1:
        prediction = np.argmax(prediction, axis = 1)
    color = np.zeros(xyz.shape)
    for i_label in range(0, n_label):
        # There is a +1 here cause the unknow label 0 isn't pass to the network so the first label become the label 1
        color[np.where(prediction == i_label), :] = colorLabelManager.label2Color[i_label+1]

    io.write_file(filename, xyz, color, dataType)
#------------------------------------------------------------------------------
def reduced_labels2full(labels_red, components, n_ver):
    """distribute the labels of superpoints to their repsective points"""
    labels_full = np.zeros((n_ver, ), dtype='uint8')
    for i_com in range(0, len(components)):
        labels_full[components[i_com]] = labels_red[i_com]
    return labels_full
#------------------------------------------------------------------------------
def writeRawPrediction(filename, xyz, prediction, components, dataType="laz"):
    """write a ply with colors for each class"""
    import pudb; pudb.set_trace()
    sorted = np.sort(prediction, axis=1)
    sppColor = [1-x[-2]/x[-1] for x in sorted]
    sppColor = np.array(sppColor) * 355
    sppColor = [[0, x, 0] for x in sppColor]

    ptColor = np.zeros([len(xyz), 3])
    for i_com in range(0, len(components)):
        ptColor[components[i_com]] = sppColor[i_com]

    
    io.write_file(filename, xyz, ptColor, dataType)
