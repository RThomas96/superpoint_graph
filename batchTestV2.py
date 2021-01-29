import sys
sys.path.append("./partition/")
sys.path.append("./learning/")
sys.path.append("./utils/")

import os
import argparse
from timer import Timer
import json
import time
import pandas as pd
import os
import numpy as np

from superpointComputation import main as superpointComputation 
from visualize_result import main as visualize
from train import main as train 
from shutil import copy
from shutil import SameFileError
from pathManager import PathManager
from collections import defaultdict

# Small function to create an empty file
def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

def getID(args, originalArgs, step):
    ID = ""
    if args["data"] != originalArgs["data"]:
        path = args["data"]
        datasetName = os.path.basename(path)
        ID += datasetName + "-"
    for key, values in args[step].items():
        if key != "--epochs" and values != originalArgs[step][key]:
            if key == "--colorCode":
                ID += str(values) + "-"
            else:
                ID += str(key)[2:] + ":"  + str(values) + "-"
    return ID[:-1]

def toList(args, projectPath):
    sppArgList = [projectPath]
    for arg in args:
        sppArgList.append(arg)
        val = str(args[arg])
        if val != "none": 
            sppArgList.append(val)
    return sppArgList

def meanAllCsvTrainingReports(pathManager):
    nbRun = pathManager.getNbRun()
    allCsvFiles = defaultdict(list) 
    means = {}
    for dataset in ["train", "test", "validation"]:
        meansDiv = {}
        for i in range(nbRun):
            file = pathManager.getTrainingCsvReport(dataset, i)
            if os.path.isfile(file):
                allCsvFiles[dataset].append(pd.read_csv(file, index_col=False, na_values='nan', sep=';'))

                if i == 0:
                    for key in allCsvFiles[dataset][-1].keys():
                        meansDiv[key] = 0 

                for key in meansDiv.keys(): meansDiv[key] += 1
                allNanColumn = allCsvFiles[dataset][-1].columns[allCsvFiles[dataset][-1].isna().any()].tolist()
                for col in allNanColumn:
                    meansDiv[col] -= 1

                allCsvFiles[dataset][-1] = allCsvFiles[dataset][-1].fillna(0)

        if bool(allCsvFiles):
            means[dataset] = allCsvFiles[dataset][0]
            for data in allCsvFiles[dataset][1:]:
                means[dataset] += data

            for key in means[dataset].keys():
                means[dataset][key] /= float(meansDiv[key]) 

            print("Write mean report into " + pathManager.getMeanTrainingCsvReport(dataset))
            means[dataset].to_csv(pathManager.getMeanTrainingCsvReport(dataset), index=False, na_rep='nan', sep=';')

def computeGeneralReport(pathManager, name, args):
    dataset = "test"
    csvReport = pd.read_csv(pathManager.getMeanTrainingCsvReport(dataset), index_col=False, na_values='nan', sep=';')
    epoch = csvReport.iloc[-1:].iloc[0, 0] 
    mean_avg_iou = csvReport[-10:].mean()[4]
    avg_iou = csvReport.iloc[-1:].iloc[0, 4] 
    max_iou = csvReport.iloc[:, 4].max()
    max_iou_epoch = csvReport.iloc[csvReport.iloc[:, 4].idxmax(), 0]
    run0 = pd.read_csv(pathManager.getTrainingCsvReport(dataset, 0), index_col=False, na_values='nan', sep=';').iloc[-1:].iloc[0, 4]
    run1 = pd.read_csv(pathManager.getTrainingCsvReport(dataset, 1), index_col=False, na_values='nan', sep=';').iloc[-1:].iloc[0, 4]
    run2 = pd.read_csv(pathManager.getTrainingCsvReport(dataset, 2), index_col=False, na_values='nan', sep=';').iloc[-1:].iloc[0, 4]
    run3 = pd.read_csv(pathManager.getTrainingCsvReport(dataset, 3), index_col=False, na_values='nan', sep=';').iloc[-1:].iloc[0, 4]

    data = {"name" : os.path.basename(name), "dataset" : os.path.basename(args["data"]), "epoch" : epoch, "avg_iou" : avg_iou, "mean_avg_iou" : mean_avg_iou, "max_iou" : max_iou, "max_iou_epoch" : max_iou_epoch, "run0" : run0, "run1" : run1, "run2" : run2, "run3" : run3}
    df = pd.DataFrame(data=data, index=[0])

    generalPath = "/gpfswork/rech/hnb/umt46nt/superpoint_graph/CSV/finalReport.csv"
    df.to_csv(generalPath, mode='a', header=not os.path.exists(generalPath), index=False, na_rep='nan', sep=';')


def main(args):

    print("Pipeline start at: " + time.asctime())
    timer = Timer(1)
    timer.start(0)

    parser = argparse.ArgumentParser(description='Batch pipeline')
    parser.add_argument('arg_json_file', help='Project name')
    parser.add_argument('--arg_original', default="args_original.json", type=str, help='Project name')
    parser.add_argument('--general_path', default="BETA4", type=str, help='Project name')
    parser.add_argument('--visu_outType', default="spge", type=str, help='Project name')
    parser.add_argument('--relative_colorCode', default="", type=str, help='Project name')
    parser.add_argument('--fileNameVisu', default="", type=str, help='Project name')

    parser.add_argument('--step', default='ptvsV', type=str, help='Only perform preprocessing step')
    args = parser.parse_args(args)

    with open(args.arg_json_file) as f:
        pipelineArgs = json.load(f)

    with open(args.arg_original) as f:
        originalArgs = json.load(f)

    #Copy colorcode in both arguments, it is equal whatever
    pipelineArgs["training"]["--colorCode"] = pipelineArgs["--colorCode"]
    pipelineArgs["sppComp"]["--colorCode"] = pipelineArgs["--colorCode"]

    originalArgs["training"]["--colorCode"] = originalArgs["--colorCode"]
    originalArgs["sppComp"]["--colorCode"] = originalArgs["--colorCode"]

    sppCompID = getID(pipelineArgs, originalArgs, 'sppComp')
    if sppCompID == "":
        sppCompID = "default"
    trainingID = getID(pipelineArgs, originalArgs, 'training')
    if trainingID == "":
        trainingID = "default"
    trainingID = sppCompID + "+" + trainingID

    sppCompPath = args.general_path + "/preprocess/" + sppCompID
    trainingPath = args.general_path + "/training/" + trainingID

    print("SPPCOMP PATH: " + sppCompID)
    print("TRAINING PATH: " + trainingID)

    os.makedirs("projects/" + sppCompPath, exist_ok=True)
    os.makedirs("projects/" + trainingPath, exist_ok=True)

    try:
        copy(args.arg_json_file,"projects/" + trainingPath)
    except SameFileError:
        pass # Can happen if the json file used is the one copied by previous training step
    try:
        os.symlink(pipelineArgs["data"], "projects/" + sppCompPath + "/data")
    except FileExistsError:
        print("Project already exist, scanning to complete")

    preprocessStep = "p" in args.step
    trainingStep = "t" in args.step
    visuStep = "v" in args.step or "V" in args.step
    allVisuStep = "V" in args.step
    statsStep = "s" in args.step

    if preprocessStep:
        print("Preprocess step")
        superpointComputation(toList(pipelineArgs['sppComp'], sppCompPath))

    if trainingStep:
        print("Training step")
        train(toList(pipelineArgs['training'], trainingPath) + ["--inPath", "projects/" + sppCompPath])
        touch("projects/" + trainingPath + "/trainingFinished")

    if visuStep:
        print("Visualisation step")
        if not allVisuStep:
            visualize([trainingPath, args.fileNameVisu, "--inPath", "projects/" + sppCompPath, "--outType", args.visu_outType])
        else:
            colorCode = pipelineArgs["--colorCode"]
            if args.relative_colorCode != "":
                colorCode = args.relative_colorCode 
            visualize([trainingPath, "None", "--inPath", "projects/" + sppCompPath, "--outType", args.visu_outType, "--colorCode", colorCode, "--all_run", "0"])
            visualize([trainingPath, "None", "--inPath", "projects/" + sppCompPath, "--outType", args.visu_outType, "--colorCode", colorCode, "--all_run", "0", "--best"])
            visualize([trainingPath, "None", "--inPath", "projects/" + sppCompPath, "--outType", args.visu_outType, "--colorCode", colorCode, "--all_run", "1"])
            visualize([trainingPath, "None", "--inPath", "projects/" + sppCompPath, "--outType", args.visu_outType, "--colorCode", colorCode, "--all_run", "1", "--best"])
            visualize([trainingPath, "None", "--inPath", "projects/" + sppCompPath, "--outType", args.visu_outType, "--colorCode", colorCode, "--all_run", "2"])
            visualize([trainingPath, "None", "--inPath", "projects/" + sppCompPath, "--outType", args.visu_outType, "--colorCode", colorCode, "--all_run", "2", "--best"])
            visualize([trainingPath, "None", "--inPath", "projects/" + sppCompPath, "--outType", args.visu_outType, "--colorCode", colorCode, "--all_run", "3"])
            visualize([trainingPath, "None", "--inPath", "projects/" + sppCompPath, "--outType", args.visu_outType, "--colorCode", colorCode, "--all_run", "3", "--best"])

    if statsStep:
        pathManager = PathManager(trainingPath, sppCompRootPath="projects/" + sppCompPath)
        meanAllCsvTrainingReports(pathManager)
        computeGeneralReport(pathManager, args.arg_json_file, pipelineArgs)

    timer.stop(0)
    print(timer.getFormattedTimer(["Total time: "]))

if __name__ == "__main__":
    main(sys.argv[1:])
