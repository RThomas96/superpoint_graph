"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    Original verion: 2017 Loic Landrieu, Martin Simonovsky
    Current version: 2020 Richard Thomas
    Script for partioning into simples shapes and prepare data
"""

import pandas
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

# Label check is different cause laz format always output labels
def has_labels(labels, dataType):
    if dataType == "laz":
        return labels.sum() > 0
    else:
        return len(labels) > 0

# Record a set of interval
class Timer:
    def __init__(self, step):
        self.times = np.zeros(step)

    def start(self, step):
        self.times[step] = time.perf_counter()

    def stop(self, step):
        self.times[step] =  time.perf_counter() - self.times[step]

    def printTimes(self, names):
        strTime = ""
        for i, x in enumerate(self.times):
            strTime += names[i]
            strTime += ": " 
            if x > 60:
                strTime += str(round(x / 60., 2)) 
                strTime += "min / "
            if x > 3600:
                strTime += str(round(x / 60. / 60., 2)) 
                strTime += "h / "
            else:
                strTime += str(round(x, 2)) 
                strTime += "s / "
        print(strTime)

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

def storePreviousFile(fileFullName, timeStamp):
    if(os.path.isfile(fileFullName)):
        fileName   = os.path.splitext(os.path.basename(fileFullName))[0]
        rootPath = os.path.dirname(fileFullName)
        logPath = rootPath + "/log"
        if not os.path.isdir(logPath):
            os.mkdir(logPath)
        newName = logPath + "/{}{}.log.h5".format(fileName, timeStamp)
        shutil.move(fileFullName, newName)
        print("Store previous file at {}...".format(newName))

def addRGBToFeature(features, rgb):
    return np.hstack((features, rgb/255.)).astype('float32')#add rgb as a feature for partitioning

def mkdirIfNotExist(dir):
    if not os.path.isdir(dir): os.mkdir(dir)

def main(args):
    parser = argparse.ArgumentParser(description='Superpoint computation programm')
    parser.add_argument('ROOT_PATH', help='name of the folder containing the data directory')
    parser.add_argument('--knn_geofeatures', default=100, type=int, help='number of neighbors for the geometric features')
    parser.add_argument('--knn_adj', default=10, type=int, help='adjacency structure for the minimal partition')
    parser.add_argument('--lambda_edge_weight', default=1., type=float, help='parameter determine the edge weight for minimal part.')
    parser.add_argument('--reg_strength', default=0.03, type=float, help='regularization strength for the minimal partition')
    parser.add_argument('--d_se_max', default=0, type=float, help='max length of super edges')
    parser.add_argument('--voxel_width', default=0.03, type=float, help='voxel size when subsampling (in m)')
    parser.add_argument('--ver_batch', default=0, type=int, help='Batch size for reading large files, 0 do disable batch loading')

    parser.add_argument('-ow', '--overwrite', action='store_true', help='Wether to read existing files or overwrite them')
    parser.add_argument('--save', action='store_true', help='Wether to save old files before overwrite them')
    parser.add_argument('--timestamp', action='store_true', help='Create a time stamp with time rather than parameters values')
    parser.add_argument('--keep_features', action='store_true', help='If set, do not recompute feature file')
    parser.add_argument('--voxelize', action='store_true', help='Choose to perform voxelization step or not')
    
    args = parser.parse_args(args)
    
    if(args.overwrite):
        print("Warning: files will be overwritten !!")
    
    timer = Timer(4)
    timer.start(0)

    colors = ColorLabelManager()
    n_labels = colors.nbColor
    pathManager = PathManager(args.ROOT_PATH)
    pathManager.createDirForSppComputation()
    
    reportManager = ReportManager(pathManager.rootPath, args)
    
    # Init timestamp
    if args.save:
        if args.timestamp:
            timeStamp = datetime.now().strftime("-%d-%m-%Y-%H:%M:%S")
        else:
            data = np.array(pandas.read_csv(reportManager.getCsvPath(), sep=';', header=None))
            timeStamp="-".join(map(str, data[-1][0:5]))
    
    for dataset in pathManager.dataset:
        print("=================\n   "+dataset+"\n=================")
        reportManager.train = not reportManager.train
    
        # Refresh timestamp
        if args.save and args.timestamp:
            timeStamp = datetime.now().strftime("-%d-%m-%Y-%H:%M:%S")

        for i in range(pathManager.getNbFiles(dataset)):
    
            fileName, dataFile, dataType, voxelisedFile, featureFile, spgFile, parseFile = pathManager.getFilesFromDataset(dataset, i)
            #TODO: cause n_labels is reset
            n_labels = colors.nbColor
    
    
            print(str(i + 1) + " / " + str(len(pathManager.allDataFileName[dataset])) + " ---> "+fileName)
            tab="   "
    
            # Step 1: Features file computation
            if (os.path.isfile(featureFile) and not args.overwrite) or args.keep_features :
                print(tab + "Reading the existing feature file...")
                geof, xyz, rgb, graph_nn, labels = provider.read_features(featureFile)
            else :
                # FIX: voxelisation
                if args.save:
                    storePreviousFile(featureFile, timeStamp)
    
                # Step 1.1: Voxelize the data file 
                timer.start(1)
                start = time.perf_counter()
                if args.voxelize:
                    print("Begin voxelisation step")
                    if os.path.isfile(voxelisedFile): 
                        print("Voxelised file found, voxelisation step skipped")
                        print("Read voxelised file")
                        xyz, rgb, labels, objects = provider.read_file(voxelisedFile, dataType)
                    else:
                        print(tab + "Read {}".format(fileName))
                        xyz, rgb, labels, objects = provider.read_file(dataFile, dataType)
    
                        #---Voxelise to reduce density-------
                        print(tab + "Reduce point density")
                        if args.voxel_width > 0:
                            xyz, rgb, labels = reduceDensity(xyz, args.voxel_width, rgb, labels, n_labels)
    
                        print(tab + "Save reduced density")
                        # BUG HERE !!
                        # Because labels returned by prune algorithme is a 2D vector
                        # With for each voxel the number of point of each label
                        # So label.flatten() return to much information you need to determine the majoritary label
                        # So use label.argmax(1) that return the index of the max value of each sub array

                        mkdirIfNotExist(pathManager.rootPath + "/data/" + dataset + "-voxelised/")
                        provider.write_file(voxelisedFile, xyz, rgb, labels.argmax(1), dataType)
                else:
                    print("Voxelisation step skipped")
                    print("Read data file")
                    xyz, rgb, labels, objects = provider.read_file(dataFile, dataType)
    
                if has_labels(labels, dataType):
                    print("Labels found")
                else :
                    print("No labels found")
                    n_labels = 0
    
                # FIX: color aggregation
                if colors.aggregation:
                    colors.aggregateLabels(labels)
                # Not needed anymore cause the label 0 is now the unknown label
                #else:
                #    labels = np.array([label+1 for label in labels])
                
                start = time.perf_counter()
    
                timer.stop(1)
                # Step 1.2: Compute nn graph
                timer.start(2)
                print(tab + "Compute both {}_nn and {}_nn graphs...".format(args.knn_adj, args.knn_geofeatures))
                graph_nn, target_fea = graphs.compute_graph_nn_2(xyz, args.knn_adj, args.knn_geofeatures)
    
                # Step 1.3: Compute geometric features
                print(tab + "Compute geometric features...")
                geof = libply_c.compute_geof(xyz, target_fea, args.knn_geofeatures).astype('float32')
                del target_fea
    
                # Step 1.4: Compute geometric features
                provider.write_features(featureFile, geof, xyz, rgb, graph_nn, labels)
                timer.stop(2)
    
            # Step 2: Compute superpoint graph
            timer.start(3)
            sys.stdout.flush()
            if os.path.isfile(spgFile) and not args.overwrite :
                print(tab + "Reading the existing superpoint graph file...")
                graph_sp, components, in_component = provider.read_spg(spgFile)
            else:
                if args.save:
                    storePreviousFile(spgFile, timeStamp)
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
                if np.array(in_component).sum() == 0:
                    print("Error: cutpursuit not working, probably due to many duplicate points")
                components = np.array(components, dtype = 'object')
    
    
                print(tab + "Computation of the SPG...")
                start = time.perf_counter()
                graph_sp = graphs.compute_sp_graph(xyz, args.d_se_max, in_component, components, labels, n_labels)
    
                # Structure graph_sp
                # "sp_labels" = nb of points per label
                # Ex: [ 0, 0, 10, 2] --> 10 pt of label 2 and 2 pt of label 3
    
                reportManager.computeStatsOnSpp(components, graph_sp["sp_labels"])
    
                provider.write_spg(spgFile, graph_sp, components, in_component)
            
            if os.path.isfile(parseFile) and not args.overwrite :
                print(tab + "Reading the existing parsed file...")
            else:
                parseCloudForPointNET(featureFile, spgFile, parseFile)
    
            # print("Timer : {:0.4f} s / {:0.4f} s / {:0.4f} s ".format(times[0], times[1], times[2]))
            timer.stop(3)
            timer.stop(0)
            timer.printTimes(["Total", "Voxelisation", "Features", "Superpoint graph"])
    
    reportManager.saveReport()

if __name__ == "__main__":
    main(sys.argv[1:])
