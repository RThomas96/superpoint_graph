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
from reportManager import ReportManagerSupervized
from fractions import Fraction

sys.path.append("./supervized_partition/")
import graph_processing as graph

parser = argparse.ArgumentParser(description='Generate ply file from prediction file')
parser.add_argument('ROOT_PATH', help='Folder name which contains data')
parser.add_argument('--supervized', action='store_true', help='Wether to read existing files or overwrite them')
parser.add_argument('--name', default='', help='If a superpoint has less pt than this value, it is deleted')
args = parser.parse_args()

#---path to data---------------------------------------------------------------
root = os.path.dirname(os.path.realpath(__file__)) + '/../projects/' + args.ROOT_PATH
spg_path = root + "/superpoint_graphs"
fea_path   = root + "/features"

colorManager = ColorLabelManager()
nbLabel = colorManager.nbColor

allFeaFilesTest = [fea_path + "/test/" + f for f in listdir(fea_path + "/test") if isfile(join(fea_path + "/test", f))]
allFeaFilesTrain = [fea_path + "/train/" + f for f in listdir(fea_path + "/train") if isfile(join(fea_path + "/train", f))]

allSpgFilesTest = [spg_path + "/test/" + f for f in listdir(spg_path + "/test") if isfile(join(spg_path + "/test", f))]
allSpgFilesTrain = [spg_path + "/train/" + f for f in listdir(spg_path + "/train") if isfile(join(spg_path + "/train", f))]

filterByName = False
if args.name != '':
    filterByName = True

for isTest in range(2):
    allFeaFiles = allFeaFilesTest if isTest == 0 else allFeaFilesTrain
    allSpgFiles = allSpgFilesTest if isTest == 0 else allSpgFilesTrain
    reportManager = ReportManagerSupervized(root, nbLabel)
    
    for i, filename in enumerate(allFeaFiles):
        name, extention = os.path.splitext(os.path.basename(filename))
        if extention != ".h5":
            continue
        if filterByName and (name != args.name and filename != args.name):
            continue
        #print("Start stat computing on " + name)
    
        geof, xyz, rgb, graph_nn, labels = provider.read_features(filename)
        try:
            graph_spg, components, in_component = provider.read_spg(allSpgFiles[i])
        except IndexError:
            for idx, spgName in enumerate(allSpgFiles):
                spgname, spgextention = os.path.splitext(os.path.basename(spgName))
                if spgname == name:
                    graph_spg, components, in_component = provider.read_spg(allSpgFiles[idx])
                    break
    
        reportManager.computeStatsOnSpp(0, components, graph_spg["sp_labels"], True)
    
    name = "test" if isTest == 0 else "train"
    print("Stats of " + name + " dataset")
    reportManager.averageComputations()
    print(reportManager.getStat()[0])
    print(reportManager.getStat()[1])

    for i, name in enumerate(reportManager.getStat()[0]):
        print(name + " : " + str(reportManager.getStat()[1][i]))

    if isTest == 1:
        for i, name in enumerate(reportManager.getStat()[0]):
            print(name + " : " + str(reportManager.getStat()[1][i] / testValue[i] if testValue[i] > 0 else 1 ))
    else:
        testValue = reportManager.getStat()[1]

    # Just in case
    #sppDeleted = len([comp for comp in components if len(comp) < int(args.nbPt)])
    #clean_comp = [idx for comp in components if len(comp) > int(args.nbPt) for idx in comp]  
    #cloudXyz = np.array([xyz[pt] for pt in clean_comp])
    #cloudRgb = np.array([rgb[pt] for pt in clean_comp])
    #cloudLabels = np.array([labels[pt] for pt in clean_comp])

    #path = os.path.split(filename)[0]
    #provider.write_file(path + "/" + name + "-Clean.laz", cloudXyz, cloudRgb, cloudLabels, "laz")
    #print(str((1-(len(clean_comp)/len(xyz)))*100) + "% of points deleted")
    #print(str(sppDeleted) + "/" + str(len(components)) + " spp deleted i.e " + str((sppDeleted / len(components))*100) + "%")
