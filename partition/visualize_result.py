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
from confusionMatrix import ConfusionMatrix
from colorLabelManager import ColorLabelManager

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
    parser.add_argument('ROOT_PATH', help='Folder name which contains results data')
    parser.add_argument('fileName', help='Full path of file to display, from data folder, must be "test/X"')
    parser.add_argument('--inPath', default='', type=str, help='Optionnal path in which all inputs files will be used')
    parser.add_argument('-ow', '--overwrite', action='store_true', help='Wether to read existing files or overwrite them')
    parser.add_argument('--supervized', action='store_true', help='Wether to read existing files or overwrite them')
    parser.add_argument('--outType', default='p', help='which cloud to output: s = superpoints, p = predictions, P = predictions with true colors, t = transitions (only for supervized partitions), g = geof, d = geof std, c = confidence, e = elevation, m = confusion matrix')
    parser.add_argument('--filter_label', help='Output only SPP with a specific label')
    parser.add_argument('--format', default="laz", type=str, help='Format in which all clouds will be saved')
    parser.add_argument('--specify_run', default=-1, type=int, help='Format in which all clouds will be saved')
    parser.add_argument('--colorCode', default="colorCode", type=str, help='Format in which all clouds will be saved')
    parser.add_argument('--all_run', default=-1, type=int, help='Format in which all clouds will be saved')

    parser.add_argument('--best', action='store_true', help='Wether to read existing files or overwrite them')

    args = parser.parse_args(args)
    
    outSuperpoints = 's' in args.outType
    outPredictions = 'p' in args.outType or 'P' in args.outType 
    outPredictionsTrueColor = 'P' in args.outType
    outRawPredictions = 'c' in args.outType
    outTransitions = 't' in args.outType
    outGeof = 'g' in args.outType
    outStd = 'd' in args.outType
    outElevation = 'e' in args.outType
    outConfusionMatrix = 'm' in args.outType

    runIndex = 0
    if args.specify_run > -1:
        runIndex = args.specify_run
    if args.all_run > -1:
        runIndex = args.all_run

    #pathManager = PathManager(args.ROOT_PATH, args.format)
    pathManager = PathManager(args.ROOT_PATH, sppCompRootPath=args.inPath, format=args.format)
    
    files = [args.fileName]
    if args.all_run > -1:
        files = [os.path.splitext(x)[0] for x in pathManager.allDataDataset[runIndex]["test"]] 
        if args.outType == 'g': 
            files += [os.path.splitext(x)[0] for x in pathManager.allDataDataset[runIndex]["train"]] 

    for chooseFile in files:
    
        fileName, dataFile, dataType, voxelisedFile, featureFile, spgFile, parseFile = pathManager.getFilesFromDataset(chooseFile)
        
        sppFile, predictionFile, transFile, geofFile, stdFile, confPredictionFile, elevationFile, confusionMatrixFile = pathManager.getVisualisationFilesFromDataset(chooseFile, runIndex)

        geof, xyz, rgb, graph_nn, labels = io.read_features(featureFile)
        #import pudb; pudb.set_trace()
        graph_spg, components, in_component = io.read_spg(spgFile)

        if outStd:
            visu.writeGeofstd(stdFile, xyz, geof, components, in_component)
        
        if outGeof:
            visu.writeGeof(geofFile, xyz, geof)
        
        if args.supervized and outTransitions:
            visu.writeTransition(transFile, xyz, edg_source, is_transition)
        
        if outPredictions:
            try:
                predFile = pathManager.getPredictionFile(runIndex)
                if args.best:
                    predFile = pathManager.getBestPredictionFile(runIndex)

                outFile = predictionFile
                if args.best:
                    outFile = outFile.replace("_pred.", "_predBest.") 

                pred_full = openPredictions(predFile, fileName, components, xyz)
                if outPredictionsTrueColor:
                    io.write_laz_labels(outFile, xyz, rgb, pred_full)
                else:
                    visu.writePrediction(outFile, xyz, pred_full, colorFile=args.colorCode)
            except ValueError:
                print("Can't visualize predictions")

        # Confidence
        if outRawPredictions:
            pred_raw = openRawPredictions(pathManager.getRawPredictionFile(runIndex), fileName)
            visu.writeRawPrediction(confPredictionFile, xyz, pred_raw, components)

        if outElevation:
            parsedFeatures = openParsedFeatures(parseFile)
            visu.writeElevation(elevationFile, parsedFeatures)
        
        if outSuperpoints and args.filter_label is not None:
            print("Filter activated")
            visu.writePartitionFilter(sppFile, xyz, components, graph_spg["sp_labels"], args.filter_label)
        
        if outSuperpoints and args.filter_label is None:
            visu.writePartition(sppFile, xyz, components)

        if outConfusionMatrix:
            # Convert prediction labels into aggregate labels

            gtCloud = voxelisedFile
            predCloud = predictionFile

            gtxyz, gtrgb, labels, [] = io.read_laz(gtCloud)
            predxyz, predrgb, pred_full, [] = io.read_laz(predictionFile)

            A = []
            convert = []
            colorLabelManager = ColorLabelManager(args.colorCode)
            for i in range(len(colorLabelManager.file)):
                val = colorLabelManager.file[i][0]
                if val not in A:
                    A.append(val)
                    convert.append(i)

            pred_full = pred_full[labels!=0] 
            labels = labels[labels!=0]
            labels = labels-1

            pred_full = np.array(convert)[np.array(pred_full)]

            confusionMatrix = ConfusionMatrix(11)
            confusionMatrix.addPrediction(pred_full, labels)
            matrix = confusionMatrix.confusionMatrix
            os.makedirs(os.path.dirname(confusionMatrixFile), exist_ok=True)
            with open(confusionMatrixFile,'wb') as f:
                for line in matrix:
                    np.savetxt(f, line, fmt='%i')


if __name__ == "__main__":
    main(sys.argv[1:])
