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
import cloudIO as provider
import os
from os import listdir
from os.path import isfile, join
from colorLabelManager import ColorLabelManager

sys.path.append("./supervized_partition/")

parser = argparse.ArgumentParser(description='Generate ply file from prediction file')
parser.add_argument('ROOT_PATH', help='Folder name which contains data')
parser.add_argument('--supervized', action='store_true', help='Wether to read existing files or overwrite them')
parser.add_argument('--nbPt', default=10, help='If a superpoint has less pt than this value, it is deleted')
parser.add_argument('--name', default='', help='If a superpoint has less pt than this value, it is deleted')
args = parser.parse_args()

#---path to data---------------------------------------------------------------
root = os.path.dirname(os.path.realpath(__file__)) + '/../../projects/' + args.ROOT_PATH
spg_path = root + "/superpoint_graphs"
fea_path   = root + "/features"

colorManager = ColorLabelManager()
nbLabel = colorManager.nbColor

allFeaFiles = [fea_path + "/"  + f for f in listdir(fea_path) if isfile(join(fea_path, f))]

allSpgFiles = [spg_path + "/" + f for f in listdir(spg_path) if isfile(join(spg_path, f))]

filterByName = False
if args.name != '':
    filterByName = True

for i, filename in enumerate(allFeaFiles):
    name, extention = os.path.splitext(os.path.basename(filename))
    if extention != ".h5":
        continue
    if filterByName and (name != args.name and filename != args.name):
        continue
    print("Start cleaning " + name)

    geof, xyz, rgb, graph_nn, labels = provider.read_features(filename)
    try:
        graph_spg, components, in_component = provider.read_spg(allSpgFiles[i])
    except IndexError:
        print("Bad opening, search")
        for idx, spgName in enumerate(allSpgFiles):
            spgname, spgextention = os.path.splitext(os.path.basename(spgName))
            if spgname == name:
                graph_spg, components, in_component = provider.read_spg(allSpgFiles[idx])
                break

    sppDeleted = len([comp for comp in components if len(comp) < int(args.nbPt)])
    clean_comp = [idx for comp in components if len(comp) > int(args.nbPt) for idx in comp]  
    cloudXyz = np.array([xyz[pt] for pt in clean_comp])
    cloudRgb = np.array([rgb[pt] for pt in clean_comp])
    cloudLabels = np.array([labels[pt] for pt in clean_comp])

    path = os.path.split(filename)[0]
    provider.write_file(path + "/" + name + "-Clean.laz", cloudXyz, cloudRgb, cloudLabels, "laz")
    print(str((1-(len(clean_comp)/len(xyz)))*100) + "% of points deleted")
    print(str(sppDeleted) + "/" + str(len(components)) + " spp deleted i.e " + str((sppDeleted / len(components))*100) + "%")
