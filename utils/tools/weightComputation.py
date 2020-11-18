"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
    
this functions outputs varied ply file to visualize the different steps
"""
import os.path
import numpy as np
import argparse
import sys
sys.path.append("./partition/")
sys.path.append("./utils")
from plyfile import PlyData, PlyElement
from pathlib import Path
import provider
import os
from os import listdir
from os.path import isfile, join
from colorLabelManager import ColorLabelManager

sys.path.append("./supervized_partition/")
import graph_processing as graph

def countNbPtPerLabel(labels, nbLabels):
    res = np.zeros(nbLabels+1)
    for label in labels:
        res[label] += 1
    return res

parser = argparse.ArgumentParser(description='Generate ply file from prediction file')
parser.add_argument('ROOT_PATH', help='Folder name which contains data')
parser.add_argument('--supervized', action='store_true', help='Wether to read existing files or overwrite them')
args = parser.parse_args()

#---path to data---------------------------------------------------------------
root = os.path.dirname(os.path.realpath(__file__)) + '/../projects/' + args.ROOT_PATH
fea_path = root + "/features"
supervized_fea_path = root + "/features_supervized"

colorManager = ColorLabelManager()
nbLabel = colorManager.nbColor

if args.supervized:
    allFiles = [supervized_fea_path + "/test/" + f for f in listdir(supervized_fea_path + "/test") if isfile(join(supervized_fea_path + "/test", f))]
    allFiles += [supervized_fea_path + "/train/" + f for f in listdir(supervized_fea_path + "/train") if isfile(join(supervized_fea_path + "/train", f))]
else:
    allFiles = [fea_path + "/test/" + f for f in listdir(fea_path + "/test") if isfile(join(fea_path + "/test", f))]
    allFiles += [fea_path + "/train/" + f for f in listdir(fea_path + "/train") if isfile(join(fea_path + "/train", f))]

finalStat = np.zeros(nbLabel+1)

if args.supervized:
    for filename in allFiles:
        xyz, rgb, edg_source, edg_target, is_transition, local_geometry, labels, objects, elevation, xyn = graph.read_structure(filename, False)
        finalStat += countNbPtPerLabel(labels, nbLabel)
else:
    for filename in allFiles:
        geof, xyz, rgb, graph_nn, labels = provider.read_features(fea_file)
        finalStat += countNbPtPerLabel(labels, nbLabel)

totalNumber = finalStat.sum()
perc = (finalStat / totalNumber)*100
weight = 1-(finalStat / totalNumber)

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
print("Pure values")
print(finalStat)
print(totalNumber)
print("All percentage")
print(perc)
print("Weights")
print(weight)
