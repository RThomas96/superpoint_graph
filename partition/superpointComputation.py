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
from reportManager import SPPComputationReportManager as ReportManager
from timer import Timer

import libcp
import graphs
import cloudIO as io

import multiprocessing

# Label check is different cause laz format always output labels
def has_labels(labels, dataType):
    if dataType == "laz":
        return labels.sum() > 0
    else:
        return len(labels) > 0

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

    # Pick all points above 0.5m up than the lower point
    val = 0.5
    low_points = ((xyz[:,2]-xyz[:,2].min() < val)).nonzero()[0]
    while len(low_points) < 200:
        val += 0.1
        low_points = ((xyz[:,2]-xyz[:,2].min() < val)).nonzero()[0]
        if val > 10000:
            break
    try:
        reg = RANSACRegressor(random_state=0).fit(xyz[low_points,:2], xyz[low_points,2])
        e = xyz[:,2]-reg.predict(xyz[:,:2])
        e /= np.max(np.abs(e),axis=0)
        e *= 0.5
    except ValueError as error:
        print ("ERROR ransac regressor" + error) 
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

def fullPipeline(i, args, pathManager):
    colors = ColorLabelManager(args.colorCode)
    n_labels = colors.nbColor

    timer = Timer(4)
    timer.start(0)
    fileName, dataFile, dataType, voxelisedFile, featureFile, spgFile, parseFile = pathManager.getFilesFromDataset(i)
    #TODO: cause n_labels is reset
    n_labels = colors.nbColor

    print(str(i + 1) + " / " + str(len(pathManager.allDataFileName)) + " ---> "+fileName)
    tab="   "

    if not args.overwrite and os.path.isfile(featureFile) and os.path.isfile(spgFile) and os.path.isfile(parseFile):
        print("Nothing to compute")
        return

    # Step 1: Features file computation
    if (os.path.isfile(featureFile) and not args.overwrite) or args.keep_features :
        print(tab + "Reading the existing feature file...")
        try:
            geof, xyz, rgb, graph_nn, labels = io.read_features(featureFile)
        except:
            print("Error: cannot load file " + featureFile)
            return
    else:
        # Step 1.1: Voxelize the data file 
        timer.start(1)
        if args.voxelize:
            print("Begin voxelisation step")
            if os.path.isfile(voxelisedFile) and not args.overwrite: 
                print("Voxelised file found, voxelisation step skipped")
                print("Read voxelised file")
            else:
                print(tab + "Reduce point density")
                if args.voxel_width > 0:
                    io.reduceDensity(dataFile, voxelisedFile, args.voxel_width, False if args.keep_density else True)

                print(tab + "Save reduced density")
                if args.only_voxelize:
                    return
            dataFile = voxelisedFile
            dataType = args.format
        else:
            print("Voxelisation step skipped")
            print("Read data file")

        try:
            xyz, rgb, labels, objects = io.read_file(dataFile, dataType)
        except:
            print("Can't read file: " + dataFile)
        if has_labels(labels, dataType):
            print("Labels found")
        else :
            print("No labels found")
            n_labels = 0
            labels = np.array([])

        if colors.needAggregation:
            print("Label aggregation activated !")
            labels = np.array(colors.aggregateLabels(labels))
        
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
        io.write_features(featureFile, geof, xyz, rgb, graph_nn, labels)
        timer.stop(2)

    # Step 2: Compute superpoint graph
    timer.start(3)
    timerOptionnal = Timer(2)
    sys.stdout.flush()
    if os.path.isfile(spgFile) and not args.overwrite :
        print(tab + "Reading the existing superpoint graph file...")
        graph_sp, components, in_component = io.read_spg(spgFile)
    else:
        if args.save:
            storePreviousFile(spgFile, timeStamp)
        #--- build the spg h5 file --

        # Add rgb to geometric feature to influence superpoint computation
        features = addRGBToFeature(geof, rgb) 

        # heuristic used by s3dis
        # increase the importance of verticality (heuristic)
        # geof[:,3] = 2. * geof[:, 3]                 

        # Add an edge weight proportionnal to distance
        # Weight= 1/dist+meanDist
        graph_nn["edge_weight"] = np.array(1. / ( args.lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = 'float32')

        timerOptionnal.start(0)
        # Compute optimisation solution with cut pursuit
        print(tab + "Resolve optimisation problem...")
        components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"], graph_nn["edge_weight"], args.reg_strength)
        if np.array(in_component).sum() == 0:
            print("Error: cutpursuit not working, probably due to many duplicate points")
        components = np.array(components, dtype = 'object')
        timerOptionnal.stop(0)


        timerOptionnal.start(1)
        print(tab + "Computation of the SPG...")
        graph_sp = graphs.compute_sp_graph(xyz, args.d_se_max, in_component, components, labels, n_labels)
        timerOptionnal.stop(1)

        # Structure graph_sp
        # "sp_labels" = nb of points per label
        # Ex: [ 0, 0, 10, 2] --> 10 pt of label 2 and 2 pt of label 3

        io.write_spg(spgFile, graph_sp, components, in_component)
    
    if os.path.isfile(parseFile) and not args.overwrite :
        print(tab + "Reading the existing parsed file...")
    else:
        parseCloudForPointNET(featureFile, spgFile, parseFile)

    timer.stop(3)
    timer.stop(0)
    formattedTimer = timer.getFormattedTimer(["Total", "Voxelisation", "Features", "Superpoint graph"])
    print(formattedTimer)
    timer.reset()

    formattedTimer = timerOptionnal.getFormattedTimer(["Resolve optimisation", "SPG computation"])
    print(formattedTimer)

def duplicateTestFilesIntoValidation(pathManager):
    for i in range(pathManager.getNbFiles("test")):
        allValidationFiles = pathManager.getFilesFromDataset("validation", i)
        allTestFiles = pathManager.getFilesFromDataset("test", i)
        for x, file in enumerate(allTestFiles):
            if os.path.isfile(file) and not os.path.isfile(allValidationFiles[x]):
                os.symlink(allTestFiles[x], allValidationFiles[x])

def duplicateTestReportIntoValidation(pathManager):
    csvTestPath = pathManager.getSppCompCsvReport("test")
    csvValPath = pathManager.getSppCompCsvReport("validation")
    if os.path.isfile(csvTestPath):
        if not os.path.isfile(csvValPath):
            os.symlink(csvTestPath, csvValPath)
    else:
        print("File not found: " + csvTestPath)    
        print("Can't copy validation csv file")    

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
    parser.add_argument('--keep_features', action='store_true', help='If set, do not recompute feature file')
    parser.add_argument('--voxelize', action='store_true', help='Choose to perform voxelization step or not')
    parser.add_argument('--only_voxelize', action='store_true', help='Choose to perform voxelization step or not')
    parser.add_argument('--keep_density', action='store_true', help='Voxelize cloud and keep original density')

    parser.add_argument('--format', default="laz", type=str, help='Format in which all clouds will be saved')
    parser.add_argument('--validationIsTest', action='store_true', help='Choose if validation dataset is the same than test dataset')
    parser.add_argument('--parallel', action='store_true', help='Choose if validation dataset is the same than test dataset')

    parser.add_argument('--colorCode', default='colorCode', help='Dataset name: sema3d|s3dis')
    
    args = parser.parse_args(args)
    
    if(args.overwrite):
        print("Warning: files will be overwritten !!")
    
    pathManager = PathManager(args.ROOT_PATH, format=args.format)
    pathManager.voxelWidth = str(args.voxel_width)
    pathManager.createDirForSppComputation()

    colors = ColorLabelManager(args.colorCode)
    n_labels = colors.nbColor
    
    if args.parallel:
        print("Paralell computation activated")
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:    
            proc = []
            for i in range(pathManager.getNbFiles()):
                proc.append(pool.apply_async(fullPipeline, (i, args,pathManager,)))
            for pr in proc:
                pr.get()
    else:
        print("Sequential computation")
        for i in range(pathManager.getNbFiles()):
            fullPipeline(i, args,pathManager)

    # Stat computation are afterward to avoid parallel process problem
    if not args.only_voxelize:
        for i in range(pathManager.getNbFiles()):
            fileName = pathManager.getFilesFromDataset(i)[0]
            spgFile = pathManager.getFilesFromDataset(i)[5]
            graph_sp, components, in_component = io.read_spg(spgFile)
            reportManager = ReportManager(args, n_labels+1)
            reportManager.computeStatOnSpp(graph_sp["sp_labels"])
            csvPath = pathManager.getSppCompCsvReport(fileName)
            csvReport = reportManager.getCsvReport()
            io.writeCsv(csvPath, csvReport[0], csvReport[1])

if __name__ == "__main__":
    main(sys.argv[1:])
