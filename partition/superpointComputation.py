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
import h5py
from sklearn.linear_model import RANSACRegressor

from colorLabelManager import ColorLabelManager
from pathManager import PathManager
from reportManager import ReportManager

import libcp
import graphs
import provider
# from graphs import *
# from provider import *

def write_ply(file, xyz, rgb, labels):
    if len(labels) > 0:
        provider.write_ply_labels(file, xyz, rgb, labels)
    else :
        provider.write_ply(file, xyz, rgb)

def parseCloudForPointNET(featureFile, graphFile, parseFile):
    """ Preprocesses data by splitting them by components and normalizing."""

    ####################
    # Computation of all the features usefull for local descriptors computation made by PointNET
    ####################
    # This file is geometric features computed to SuperPoint construction
    # There are still usefull for local descriptors computation 
    geometricFeatureFile = h5py.File(featureFile, 'r')
    xyz = geometricFeatureFile['xyz'][:]
    rgb = geometricFeatureFile['rgb'][:].astype(np.float)
    rgb = rgb/255.0 - 0.5
    # elpsv = np.stack([ featureFile['xyz'][:,2][:], featureFile['linearity'][:], featureFile['planarity'][:], featureFile['scattering'][:], featureFile['verticality'][:] ], axis=1)
    lpsv = geometricFeatureFile['geof'][:] 
    lpsv -= 0.5 #normalize

    # Compute elevation with simple Ransac from low points
    #if isTrainFolder:
    #    e = xyz[:,2] / 4 - 0.5 # (4m rough guess)
    #else :
    low_points = ((xyz[:,2]-xyz[:,2].min() < 0.5)).nonzero()[0]
    try:
        reg = RANSACRegressor(random_state=0).fit(xyz[low_points,:2], xyz[low_points,2])
        e = xyz[:,2]-reg.predict(xyz[:,:2])
        e /= np.max(np.abs(e),axis=0)
        e *= 0.5
    except ValueError as error:
        print ("ERROR ransac regressor: " + error) 
        e = xyz[:,2] / 4 - 0.5 # (4m rough guess)

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

    # Concatenante data so that each line have this format
    parsedData = np.concatenate([xyz, rgb, e[:,np.newaxis], lpsv, xyzn, distance_to_center[:,None]], axis=1)

    # Old features
    # parsedData = np.concatenate([xyz, rgb, elpsv], axis=1)

    graphFile = h5py.File(graphFile, 'r')
    nbComponents = len(graphFile['components'].keys())

    with h5py.File(parseFile, 'w') as parsedFile:
        for components in range(nbComponents):
            idx = graphFile['components/{:d}'.format(components)][:].flatten()
            if idx.size > 10000: # trim extra large segments, just for speed-up of loading time
                ii = random.sample(range(idx.size), k=10000)
                idx = idx[ii]
            # For all points in the superpoint ( the set of index "idx"), get all correspondant parsed data and add it to the file
            parsedFile.create_dataset(name='{:d}'.format(components), data=parsedData[idx,...])

def reduceDensity(xyz, voxel_width, rgb, labels, n_labels):
    asNoLabel = False
    if len(labels) == 0:
        asNoLabel = True
        labels = np.array([]) 
        n_labels = 0
    xyz, rgb, labels, dump = libply_c.prune(xyz, args.voxel_width, rgb, labels, np.zeros(1, dtype='uint8'), n_labels, 0)
    if asNoLabel:
        labels = np.array([]) 
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

def mkdirIfNotExist(dir):
    if not os.path.isdir(dir): os.mkdir(dir)

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
parser.add_argument('--save', action='store_true', help='Wether to read existing files or overwrite them')
args = parser.parse_args()

if(args.overwrite):
    print("Warning: files will be overwritten !!")

colors = ColorLabelManager()
n_labels = colors.nbColor
pathManager = PathManager(args)
reportManager = ReportManager(pathManager.rootPath, args)

times = [0.,0.,0.,0.] # Time for computing: features / partition / spg

for folder in pathManager.folders:
    print("=================\n   "+folder+"\n=================")
    reportManager.train = not reportManager.train

    for i, fileName in enumerate(pathManager.allDataFileName[folder]):

        n_labels = colors.nbColor
        dataFolder = pathManager.rootPath + "/data/raw/" + folder + "/" 
        dataFile = dataFolder + fileName + '.ply' #or .las

        featureFile  = pathManager.rootPath + "/features/" + folder + "/" + fileName + ".h5" 
        spgFile  = pathManager.rootPath + "/superpoint_graphs/" + folder + "/" + fileName + ".h5" 
        parseFile  = pathManager.rootPath + "/parsed/" + folder + "/" + fileName + ".h5"

        voxelisedFile  = pathManager.rootPath + "/data/voxelised/" + folder + "/" + fileName + "/" + fileName + "-prunned" + str(args.voxel_width).replace(".", "-") + ".ply"

        for sub in ["/features", "/superpoint_graphs", "/parsed"] : 
            mkdirIfNotExist(pathManager.rootPath + sub)
            for subsub in ["/test", "/train"] : 
                mkdirIfNotExist(pathManager.rootPath + sub + subsub)
        mkdirIfNotExist(pathManager.rootPath + "/reports")

        #if not os.path.isdir(pathManager.rootPath + "/voxelised/" + folder + "/" + fileName): os.mkdir(pathManager.rootPath + "/voxelised/" + folder + "/" + fileName)
        
        print(str(i + 1) + " / " + str(len(pathManager.allDataFileName[folder])) + " ---> "+fileName)
        tab="   "

        #--- build the geometric feature file h5 file ---
        if os.path.isfile(featureFile) and not args.overwrite :
            print(tab + "Reading the existing feature file...")
            geof, xyz, rgb, graph_nn, labels = provider.read_features(featureFile)
        else :
            if args.save:
                storePreviousFile(featureFile)

            start = time.perf_counter()
            if os.path.isfile(voxelisedFile):
                print("Voxelised file found, voxelisation step skipped")
                print("Read voxelised file")
                xyz, rgb, labels, objects = provider.read_ply(voxelisedFile)
                if len(labels) == 0:
                    print("No labels found")
                    n_labels = 0
                else :
                    print("Labels found")
            else :
                print(tab + "Read {}".format(fileName))
                xyz, rgb, labels, objects = provider.read_ply(dataFile)
                if len(labels) == 0:
                    print("No labels found")
                    n_labels = 0
                else :
                    print("Labels found")

                end = time.perf_counter()
                times[0] = times[0] + end - start
                #---Voxelise to reduce density-------
                print(tab + "Reduce point density")
                if args.voxel_width > 0:
                    xyz, rgb, labels = reduceDensity(xyz, args.voxel_width, rgb, labels, n_labels)

                print(tab + "Save reduced density")
                write_ply(voxelisedFile, xyz, rgb, labels.flatten())

            if colors.aggregation:
                colors.aggregateLabels(labels)
            
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
            if args.save:
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

            reportManager.computeStatsOnSpp(components, graph_sp)

            end = time.perf_counter()
            times[3] = times[3] + end - start

            provider.write_spg(spgFile, graph_sp, components, in_component)
        
        if os.path.isfile(parseFile) and not args.overwrite :
            print(tab + "Reading the existing parsed file...")
        else:
            parseCloudForPointNET(featureFile, spgFile, parseFile)

        # print("Timer : {:0.4f} s / {:0.4f} s / {:0.4f} s ".format(times[0], times[1], times[2]))
        print(f"Timer : {times[0]:0.4f} s loading files / {times[1]:0.4f} s features / {times[2]:0.4f} s superpoints / {times[3]:0.4f} s graph")
        times=[0., 0., 0., 0.]

reportManager.saveReport()