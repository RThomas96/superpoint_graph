import sys
sys.path.append("./partition/")
sys.path.append("./learning/")
sys.path.append("./utils/")

import os
import argparse
from timer import Timer
import json
import time

from superpointComputation import main as superpointComputation 
from visualize_result import main as visualize
from train import main as train 

def getID(args, originalArgs, step):
    ID = ""
    for key, values in args[step].items():
        if values != originalArgs[step][key]:
            ID += str(key)[2:] + str(values) + "-"
    return ID[:-1]

def toList(args, projectPath):
    sppArgList = [projectPath]
    for arg in args:
        sppArgList.append(arg)
        val = str(args[arg])
        if val != "none": 
            sppArgList.append(val)
    return sppArgList

def main(args):

    print("Pipeline start at: " + time.asctime())
    timer = Timer(1)
    timer.start(0)

    parser = argparse.ArgumentParser(description='Batch pipeline')
    parser.add_argument('arg_json_file', help='Project name')
    parser.add_argument('--arg_original', default="args_original.json", type=str, help='Project name')
    parser.add_argument('--general_path', default="BETA", type=str, help='Project name')
    parser.add_argument('--preproc_only', action='store_true', help='Only perform preprocessing step')
    args = parser.parse_args(args)

    with open(args.arg_json_file) as f:
        pipelineArgs = json.load(f)

    with open(args.arg_original) as f:
        originalArgs = json.load(f)

    sppCompID = getID(pipelineArgs, originalArgs, 'sppComp')
    if sppCompID == "":
        sppCompID = "default"
    trainingID = getID(pipelineArgs, originalArgs, 'training')
    if trainingID == "":
        trainingID = "default"

    sppCompPath = args.general_path + "/preprocess/" + sppCompID
    trainingPath = args.general_path + "/training/" + trainingID

    os.makedirs("projects/" + sppCompPath, exist_ok=True)
    os.makedirs("projects/" + trainingPath, exist_ok=True)

    try:
        os.symlink(pipelineArgs["data"], "projects/" + sppCompPath + "/data")
    except FileExistsError:
        print("Project already exist, scanning to complete")

    superpointComputation(toList(pipelineArgs['sppComp'], sppCompPath))
    if not args.preproc_only:
        train(toList(pipelineArgs['training'], trainingPath) + ["--inPath", "projects/" + sppCompPath])

        #visualize([sppArgs.getProjectPath(), "LPA3-1", "--outType", "spctgde", "--format", "laz"])
    #else:
        #visualize([sppArgs.getProjectPath(), "LPA3-1", "--outType", "sgde", "--format", "laz"])

    ####################

    #rawSppArgs = {
    #        "--knn_geofeatures" : 100,
    #        "--knn_adj" : 10,
    #        "--lambda_edge_weight" : float(1.),
    #        "--reg_strength" : float(0.01),
    #        "--d_se_max" : 0,
    #        "--voxel_width" : float(0.01),
    #        "--validationIsTest" :"none",
    #        "--voxelize" :"none",
    #        "--parallel" : "none"
    #}

    #allProjectsPath = []
    #jobs = []

    #it = rank
    #rawSppArgs["--reg_strength"] += float(it) * float(args.it) + float(args.start) * float(args.it)
    #rawSppArgs["--reg_strength"] = round(rawSppArgs["--reg_strength"], 2)# Avoid decimal error
    #sppArgs = SppArgs(args.project_path, rawSppArgs)
    #allProjectsPath.append("projects/" + sppArgs.getProjectPath())
    #try:
    #    os.makedirs("projects/" + sppArgs.getProjectPath())
    #    os.symlink(args.data_path, "projects/" + sppArgs.getProjectPath() + "/data")
    #except FileExistsError:
    #    print("Project already exist, scan to complete")

    #superpointComputation(sppArgs.toList() + ["--format", "laz"])
    #if not args.preproc_only:
    #    #B7NoEnt 
    #    #train([sppArgs.getProjectPath(), "--epoch", "300", "--resume", "--batch_size", "6", "--parallel", "--loss_weights", "none"])
    #    #B7
    #    #train([sppArgs.getProjectPath(), "--epoch", "300", "--resume", "--batch_size", "6", "--parallel"])
    #    #B7CleanNN200
    #    train([sppArgs.getProjectPath(), "--epoch", "300", "--resume", "--batch_size", "6", "--parallel", "--spg_augm_nneigh", "200"])
    #    visualize([sppArgs.getProjectPath(), "LPA3-1", "--outType", "spctgde", "--format", "laz"])
    #else:
    #    visualize([sppArgs.getProjectPath(), "LPA3-1", "--outType", "sgde", "--format", "laz"])

    #timer.stop(0)
    #print(timer.getFormattedTimer(["Total time: "]))

if __name__ == "__main__":
    main(sys.argv[1:])
