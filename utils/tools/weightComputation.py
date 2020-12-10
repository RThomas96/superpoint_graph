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
import cloudIO
import os
from os import listdir
from os.path import isfile, join
from colorLabelManager import ColorLabelManager

sys.path.append("./supervized_partition/")

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
root = os.path.dirname(os.path.realpath(__file__)) + '/../../projects/' + args.ROOT_PATH
spg_path = root + "/superpoint_graphs"
fea_path   = root + "/features"

colorManager = ColorLabelManager()
nbLabel = colorManager.nbColor

allFeaFiles = [fea_path + "/"  + f for f in listdir(fea_path) if isfile(join(fea_path, f))]
allSpgFiles = [spg_path + "/" + f for f in listdir(spg_path) if isfile(join(spg_path, f))]

finalStat = np.zeros(nbLabel+1)
for i, filename in enumerate(allFeaFiles):
    name, extention = os.path.splitext(os.path.basename(filename))
    if extention != ".h5":
        continue
    print("Start counting " + name)

    geof, xyz, rgb, graph_nn, labels = cloudIO.read_features(filename)
    try:
        graph_spg, components, in_component = cloudIO.read_spg(allSpgFiles[i])
    except IndexError:
        print("Bad opening, search")
        for idx, spgName in enumerate(allSpgFiles):
            spgname, spgextention = os.path.splitext(os.path.basename(spgName))
            if spgname == name:
                graph_spg, components, in_component = cloudIO.read_spg(allSpgFiles[idx])
                break

    labelEachSpp = np.array([np.bincount(labels[x]).argmax() for x in components])
    for i, x in enumerate(np.bincount(labelEachSpp)):
        finalStat[i] += x 


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
