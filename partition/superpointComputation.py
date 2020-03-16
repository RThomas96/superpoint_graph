"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
    Script for partioning into simples shapes
"""
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
import libply_c
from datetime import datetime
import time
from glob import glob

import libcp
import graphs
import provider
# from graphs import *
# from provider import *

def readFile(file, type):
    if type == "s3dis":
        xyz, rgb, labels, objects = provider.read_s3dis_format(file)
    else :
        xyz, rgb = provider.read_ply(file)
        labels=[]
        objects=[]
    return xyz, rgb, labels, objects

def reduceDensity(xyz, voxel_width, rgb, labels, n_labels):
    if n_labels > 0:
        xyz, rgb, labels, dump = libply_c.prune(xyz.astype('f4'), args.voxel_width, rgb.astype('uint8'), labels.astype('uint8'), np.zeros(1, dtype='uint8'), n_labels, 0)
    else:
        xyz, rgb, labels, dump = libply_c.prune(xyz.astype('f4'), args.voxel_width, rgb.astype('uint8'), np.zeros(1, dtype='uint8'), np.zeros(1, dtype='uint8'), 0, 0)
        labels = []
    return xyz, rgb, labels

def storePreviousFile(fileFullName):
    if(os.path.isfile(fileFullName)):
        fileName   = os.path.splitext(os.path.basename(fileFullName))[0]
        rootPath = os.path.dirname(fileFullName)
        logPath = rootPath + "/log"
        if not os.path.isdir(logPath):
            os.mkdir(logPath)
        timeStamp = datetime.now().strftime("-%d-%m-%Y-%H:%M:%S")
        newName = logPath + "/{}{}.log.h5".format(fileName, timeStamp)
        shutil.move(fileFullName, newName)
        print("Store previous file at {}...".format(newName))

def addRGBToFeature(features, rgb):
    return np.hstack((features, rgb/255.)).astype('float32')#add rgb as a feature for partitioning

class PathManager : 
    def __init__(self, args, dataType="ply"):
        self.rootPath = os.path.dirname(os.path.realpath(__file__)) + '/../' + args.ROOT_PATH
        self.folders = ["test", "train"]
        self.subfolders = ["features", "superpoint_graph", "data"]

        self.allDataFileName = {}
        for folder in self.folders:
            dataPath = self.rootPath + "/data/" + folder
            try:
                allDataFiles = glob(dataPath + "/*."+ dataType)
            except OSError:
                print("{} do not exist ! It is needed and contain input point clouds.".format(dataPath))
            self.allDataFileName[folder] = []
            for dataFile in allDataFiles:
                dataName = os.path.splitext(os.path.basename(dataFile))[0]
                self.allDataFileName[folder].append(dataName)
            if len(self.allDataFileName[folder]) <= 0:
                print("Warning: {} folder is empty or do not contain {} format file".format(folder, dataType))
                #raise FileNotFoundError("Data folder is empty or do not contain {} format files".format(dataType))

parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
parser.add_argument('ROOT_PATH', help='name of the folder containing the data directory')
parser.add_argument('--knn_geofeatures', default=45, type=int, help='number of neighbors for the geometric features')
parser.add_argument('--knn_adj', default=10, type=int, help='adjacency structure for the minimal partition')
parser.add_argument('--lambda_edge_weight', default=1., type=float, help='parameter determine the edge weight for minimal part.')
parser.add_argument('--reg_strength', default=0.03, type=float, help='regularization strength for the minimal partition')
parser.add_argument('--d_se_max', default=0, type=float, help='max length of super edges')
parser.add_argument('--voxel_width', default=0.03, type=float, help='voxel size when subsampling (in m)')
parser.add_argument('--ver_batch', default=0, type=int, help='Batch size for reading large files, 0 do disable batch loading')
parser.add_argument('-ow', '--overwrite', action='store_true', help='Wether to read existing files or overwrite them')
args = parser.parse_args()

if(args.overwrite):
    print("Warning: files will be overwritten !!")

pathManager = PathManager(args)

times = [0.,0.,0.,0.] # Time for computing: features / partition / spg

for folder in pathManager.folders:
    print("=================\n   "+folder+"\n=================")
    if folder == "train" :
        folderTrain = True
        n_labels = 13 # Number of classes
    else:
        folderTrain = False
        n_labels = 0

    for i, fileName in enumerate(pathManager.allDataFileName[folder]):

        dataFolder = pathManager.rootPath + "/data/" + folder + "/" 
        if folderTrain:
            dataFile = dataFolder + fileName + '/' + fileName  + '.txt'
        else:
            dataFile = dataFolder + fileName + '.ply' #or .las

        featureFile  = pathManager.rootPath + "/features/" + folder + "/" + fileName + ".h5" 
        spgFile  = pathManager.rootPath + "/superpoint_graphs/" + folder + "/" + fileName + ".h5" 
        
        print(str(i + 1) + " / " + str(len(pathManager.allDataFileName[folder])) + "---> "+fileName)
        tab="   "

        #--- build the geometric feature file h5 file ---
        if os.path.isfile(featureFile) and not args.overwrite :
            print(tab + "Reading the existing feature file...")
            geof, xyz, rgb, graph_nn, labels = provider.read_features(featureFile)
        else :
            storePreviousFile(featureFile)

            start = time.perf_counter()

            print(tab + "Read {}".format(fileName))
            readType = "s3dis" if folderTrain else "custom"
            xyz, rgb, labels, objects = readFile(dataFile, readType)

            end = time.perf_counter()
            times[0] = times[0] + end - start
            #---Voxelise to reduce density-------
            print(tab + "Reduce point density")
            if args.voxel_width > 0:
                xyz, rgb, labels = reduceDensity(xyz, args.voxel_width, rgb, labels, n_labels)

            start = time.perf_counter()

            #---compute 10 nn graph-------
            print(tab + "Compute both {}_nn and {}_nn graphs...".format(args.knn_adj, args.knn_geofeatures))
            graph_nn, target_fea = graphs.compute_graph_nn_2(xyz, args.knn_adj, args.knn_geofeatures)

            #---compute geometric features-------
            print(tab + "Compute geometric features...")
            geof = libply_c.compute_geof(xyz, target_fea, args.knn_geofeatures).astype('float32')
            end = time.perf_counter()
            times[1] = times[1] + end - start
            del target_fea

            provider.write_features(featureFile, geof, xyz, rgb, graph_nn, labels)

        #--compute the partition------
        sys.stdout.flush()
        if os.path.isfile(spgFile) and not args.overwrite :
            print(tab + "Reading the existing superpoint graph file...")
            graph_sp, components, in_component = provider.read_spg(spgFile)
        else:
            storePreviousFile(spgFile)
            #--- build the spg h5 file --
            start = time.perf_counter()

            # Add rgb to geometric feature to influence superpoint computation
            features = addRGBToFeature(geof, rgb) 

            # heuristic used by s3dis
            # increase the importance of verticality (heuristic)
            # geof[:,3] = 2. * geof[:, 3]                 

            # Add an edge weight proportionnal to distance
            # Weight= 1/dist+meanDist
            graph_nn["edge_weight"] = np.array(1. / ( args.lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = 'float32')

            # Compute optimisation solution with cut pursuit
            print(tab + "Resolve optimisation problem...")
            components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"], graph_nn["edge_weight"], args.reg_strength)
            components = np.array(components, dtype = 'object')

            end = time.perf_counter()
            times[2] = times[2] + end - start

            print(tab + "Computation of the SPG...")
            start = time.perf_counter()
            graph_sp = graphs.compute_sp_graph(xyz, args.d_se_max, in_component, components, labels, n_labels)

            end = time.perf_counter()
            times[3] = times[3] + end - start

            provider.write_spg(spgFile, graph_sp, components, in_component)
        
        # print("Timer : {:0.4f} s / {:0.4f} s / {:0.4f} s ".format(times[0], times[1], times[2]))
        print(f"Timer : {times[0]:0.4f} s loading files / {times[1]:0.4f} s features / {times[2]:0.4f} s superpoints / {times[3]:0.4f} s graph")
        times=[0., 0., 0., 0.]