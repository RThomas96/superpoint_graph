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
sys.path.append("./utils/")
from plyfile import PlyData, PlyElement
from pathlib import Path
import visualisation as visu
import cloudIO as io
import h5py

sys.path.append("./supervized_partition/")
#import graph_processing as graph
from pathManager import PathManager 

def openPredictions(res_file, h5FolderPath, components, xyz):
    try:
        h5FileFolders = list(h5py.File(res_file, 'r').keys())
        if not h5FolderPath in h5FileFolders:
            print("%s does not exist in %s" % (h5FolderPath, res_file))
            raise ValueError("%s does not exist in %s" % (h5FolderPath, res_file))
        pred_red  = np.array(h5py.File(res_file, 'r').get(h5FolderPath))        
        if (len(pred_red) != len(components)):
            print(len(pred_red))
            print(len(components))
            raise ValueError("It looks like the spg is not adapted to the result file") 
        return visu.reduced_labels2full(pred_red, components, len(xyz))
    except OSError:
        raise ValueError("%s does not exist in %s" % (h5FolderPath, res_file))

def openRawPredictions(res_file, h5FolderPath):
    try:
        h5FileFolders = list(h5py.File(res_file, 'r').keys())
        if not h5FolderPath in h5FileFolders:
            print("%s does not exist in %s" % (h5FolderPath, res_file))
            raise ValueError("%s does not exist in %s" % (h5FolderPath, res_file))
        return np.array(h5py.File(res_file, 'r').get(h5FolderPath))        
    except OSError:
        raise ValueError("%s does not exist in %s" % (h5FolderPath, res_file))

def openParsedFeatures(parsed_file):
    e = []
    file = h5py.File(parsed_file, 'r')
    for key in list(file.keys()):
        e.append(np.array(file.get(key)))
    return e 

def main(args):
    parser = argparse.ArgumentParser(description='Generate ply file from prediction file')
    parser.add_argument('ROOT_PATH', help='Folder name which contains data')
    parser.add_argument('fileName', help='Full path of file to display, from data folder, must be "test/X"')
    parser.add_argument('-ow', '--overwrite', action='store_true', help='Wether to read existing files or overwrite them')
    parser.add_argument('--supervized', action='store_true', help='Wether to read existing files or overwrite them')
    parser.add_argument('--outType', default='p', help='which cloud to output: s = superpoints, p = predictions, t = transitions (only for supervized partitions), g = geof, d = geof std, c = confidence, e = elevation')
    parser.add_argument('--filter_label', help='Output only SPP with a specific label')
    parser.add_argument('--log', help='Files are read from log directory, you can set some REG_STRENGTH value to choose which file to choose')
    parser.add_argument('--format', default="laz", type=str, help='Format in which all clouds will be saved')
    parser.add_argument('--specify_run', default=-1, type=str, help='Format in which all clouds will be saved')

    args = parser.parse_args(args)
    
    outSuperpoints = 's' in args.outType
    outPredictions = 'p' in args.outType
    outRawPredictions = 'c' in args.outType
    outTransitions = 't' in args.outType
    outGeof = 'g' in args.outType
    outStd = 'd' in args.outType
    outElevation = 'e' in args.outType

    runIndex = 0
    if args.specify_run > -1:
        runIndex = args.specify_run
    
    pathManager = PathManager(args.ROOT_PATH, args.format)
    
    #if args.log is not None:
    #    folder = os.path.split(args.file_path)[0] + '/log/'
    #    file_name = os.path.split(args.file_path)[1] + args.log +"-1.0-45-10-1000000.log"
    #else:
    #    folder = os.path.split(args.file_path)[0] + '/'
    #    file_name = os.path.split(args.file_path)[1]
    
    fileName, dataFile, dataType, voxelisedFile, featureFile, spgFile, parseFile = pathManager.getFilesFromDataset(args.fileName)
    
    sppFile, predictionFile, transFile, geofFile, stdFile, confPredictionFile = pathManager.getVisualisationFilesFromDataset(args.fileName, runIndex)
    
    #if args.supervized:
    #    xyz, rgb, edg_source, edg_target, is_transition, local_geometry, labels, objects, elevation, xyn = graph.read_structure(supervized_fea_file, False)
    #else:
    geof, xyz, rgb, graph_nn, labels = io.read_features(featureFile)
    
    graph_spg, components, in_component = io.read_spg(spgFile)

    if outStd:
        visu.writeGeofstd(stdFile, xyz, geof, components, in_component)
    
    if outGeof:
        visu.writeGeof(geofFile, xyz, geof)
    
    if args.supervized and outTransitions:
        visu.writeTransition(transFile, xyz, edg_source, is_transition)
    
    if outPredictions:
        try:
            pred_full = openPredictions(pathManager.getPredictionFile(runIndex), fileName, components, xyz)
            visu.writePrediction(predictionFile, xyz, pred_full)
        except ValueError:
            print("Can't visualize predictions")

    # Confidence
    if outRawPredictions:
        pred_raw = openRawPredictions(pathManager.getRawPredictionFile(runIndex), fileName)
        visu.writeRawPrediction(confPredictionFile, xyz, pred_raw, components)

    if outElevation:
        parsedFeatures = openParsedFeatures(parseFile)
        visu.writeElevation(confPredictionFile.replace("_conf", "_elevation"), parsedFeatures)
    
    if outSuperpoints and args.filter_label is not None:
        print("Filter activated")
        visu.writePartitionFilter(sppFile, xyz, components, graph_spg["sp_labels"], args.filter_label)
    
    if outSuperpoints and args.filter_label is None:
        visu.writePartition(sppFile, xyz, components)

if __name__ == "__main__":
    main(sys.argv[1:])
