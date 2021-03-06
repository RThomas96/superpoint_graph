"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    Original verion: 2017 Loic Landrieu, Martin Simonovsky
    Current version: 2020 Richard Thomas
    Script for partioning into simples shapes and prepare data
"""
import random
import os.path
import shutil
import sys
import numpy as np
import argparse
from pathlib import Path
from timeit import default_timer as timer
sys.path.append("./partition/cut-pursuit/build/src")
sys.path.append("./partition/ply_c")
sys.path.append("./partition")
sys.path.append("./utils")
import libply_c
from datetime import datetime
import time
from glob import glob
import h5py
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors

from colorLabelManager import ColorLabelManager
from pathManager import PathManager

import provider

# from graphs import *
# from provider import *

def interpolate_labels(xyz_up, xyz, labels):
    """interpolate the labels of the pruned cloud to the full cloud"""
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis = 1)
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(xyz)
    distances, neighbor = nn.kneighbors(xyz_up)
    return labels[neighbor].flatten()

parser = argparse.ArgumentParser(description='Apply labels from a labelised file into another file')
parser.add_argument('ROOT_PATH', help='name of the folder containing the data directory')
parser.add_argument('fileName', help='name of the folder containing the data directory')
#parser.add_argument('labelisedFile', help='name of the folder containing the data directory')
#parser.add_argument('originalFile', help='name of the folder containing the data directory')
args = parser.parse_args()

colorManager = ColorLabelManager()

times = [0.,0.,0.,0.] # Time for computing: features / partition / spg

scriptPath = os.path.dirname(os.path.abspath(__file__)) + "/../projects/" + args.ROOT_PATH
print("scriptPath = " + scriptPath)
voxelisedPath = scriptPath + "/data/voxelised"
voxelisedPath = voxelisedPath + "/test" if os.path.isdir(voxelisedPath + "/test/" + args.fileName) else voxelisedPath + "/train"
voxelisedPath = voxelisedPath + "/" + args.fileName 
print("voxelisedPath = " + voxelisedPath)

originalFile = glob(voxelisedPath + "/*-noLabel.ply")[0]
print("originalFile = " + originalFile)
labelisedFile = glob(voxelisedPath + "/*-labelised.ply")[0]
print("labelisedFile = " + labelisedFile)

outFileName = originalFile.split('/')[-1].replace('-noLabel', '')

outFile = voxelisedPath + "/" + outFileName 
print("outFile = " + outFile)

print("Reading labelised file")
xyz, rgb, labels, objects = provider.read_ply(labelisedFile)

if len(labels) > 0:
    raise NameError('{} file already contain labels'.format(originalFile[-1]))

labels = np.zeros(len(rgb))
rgbToLabel = colorManager.colorDict 
for i, color in enumerate(rgb):
    key = str(color[0])+str(color[1])+str(color[2])
    try:
        labels[i] = (rgbToLabel[key])
    except KeyError:
        labels[i] = -1 

print("Reading original file")
xyz_up, rgb_up, labels_up, objects_up = provider.read_ply(originalFile)

print("Interpolate labels")
labels_up = interpolate_labels(xyz_up, xyz, labels)

print("Writing the upsampled prediction file")
provider.write_ply_labels(outFile, xyz_up, rgb_up, labels_up)
