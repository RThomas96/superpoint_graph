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

from superpointComputation import main as superpointComputation 
from visualize_result import main as visualize
from train import main as train 
from shutil import copy
from pathManager import PathManager
from collections import defaultdict

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
        for i in range(nbRun):
            file = pathManager.getTrainingCsvReport(dataset, i)
            if os.path.isfile(file):
                allCsvFiles[dataset].append(pd.read_csv(file, index_col=False, na_values='nan', sep=';'))
        means[dataset] = allCsvFiles[dataset][0]
        for data in allCsvFiles[dataset][1:]:
            means[dataset] += data
        means[dataset] /= float(pathManager.getNbRun()) 
        print("Write mean report into " + pathManager.getMeanTrainingCsvReport(dataset))
        means[dataset].to_csv(pathManager.getMeanTrainingCsvReport(dataset), index=False, na_rep='nan', sep=';')

def main(args):

    print("Pipeline start at: " + time.asctime())
    timer = Timer(1)
    timer.start(0)

    parser = argparse.ArgumentParser(description='Batch pipeline')
    parser.add_argument('arg_json_file', help='Project name')
    parser.add_argument('--arg_original', default="args_original.json", type=str, help='Project name')
    parser.add_argument('--general_path', default="BETA", type=str, help='Project name')

    parser.add_argument('--step', default='ptvs', type=str, help='Only perform preprocessing step')
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

    copy(args.arg_json_file,"projects/" + trainingPath)
    try:
        os.symlink(pipelineArgs["data"], "projects/" + sppCompPath + "/data")
    except FileExistsError:
        print("Project already exist, scanning to complete")

    preprocessStep = "p" in args.step
    trainingStep = "t" in args.step
    visuStep = "v" in args.step
    statsStep = "s" in args.step

    if preprocessStep:
        superpointComputation(toList(pipelineArgs['sppComp'], sppCompPath))

    if trainingStep:
        train(toList(pipelineArgs['training'], trainingPath) + ["--inPath", "projects/" + sppCompPath])

    if visuStep:
        visualize([trainingPath, "LPA3-1", "--inPath", "projects/" + sppCompPath, "--outType", "spge"])

    if statsStep:
        pathManager = PathManager(trainingPath, sppCompRootPath="projects/" + sppCompPath)
        meanAllCsvTrainingReports(pathManager)

    timer.stop(0)
    print(timer.getFormattedTimer(["Total time: "]))


        #visualize([sppArgs.getProjectPath(), "LPA3-1", "--outType", "spctgde", "--format", "laz"])
    #else:
        #visualize([sppArgs.getProjectPath(), "LPA3-1", "--outType", "sgde", "--format", "laz"])

    #    #B7NoEnt 
    #    #train([sppArgs.getProjectPath(), "--epoch", "300", "--resume", "--batch_size", "6", "--parallel", "--loss_weights", "none"])
    #    #B7
    #    #train([sppArgs.getProjectPath(), "--epoch", "300", "--resume", "--batch_size", "6", "--parallel"])
    #    #B7CleanNN200
    #    train([sppArgs.getProjectPath(), "--epoch", "300", "--resume", "--batch_size", "6", "--parallel", "--spg_augm_nneigh", "200"])
    #    visualize([sppArgs.getProjectPath(), "LPA3-1", "--outType", "spctgde", "--format", "laz"])
    #else:
    #    visualize([sppArgs.getProjectPath(), "LPA3-1", "--outType", "sgde", "--format", "laz"])

if __name__ == "__main__":
    main(sys.argv[1:])
