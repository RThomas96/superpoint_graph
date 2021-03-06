import argparse
import sys
import os
import shutil
sys.path.append("./partition/")
sys.path.append("./learning/")
from superpointComputation import main as superpointComputation 
from visualize_result import main as visualize
from train import main as train 
import pandas as pd
import csv
import torch.multiprocessing as multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from mpi4py import MPI
import time
from timer import Timer

class SppArgs:
    def __init__(self, project_path, argDict):
        self.dict = argDict
        self.project_path = project_path + "/"

    def getId(self):
        project_id = "" 
        for i in self.dict.values():
            if str(i) != "none":
                project_id += str(i) + "-"
        return project_id[:-1]
        
    def toList(self):
        sppArgList = [self.getProjectPath()]
        for arg in self.dict:
            sppArgList.append(arg)
            val = str(self.dict[arg])
            if val != "none": 
                sppArgList.append(val)
        return sppArgList

    def getProjectPath(self):
        return self.project_path + self.getId()

# Insert all columns from f2 into f1 
def mergeCSV(fileName1, fileName2):
    f1 = pd.read_csv(fileName1, sep=';', index_col=False)
    f2 = pd.read_csv(fileName2, sep=';', index_col=False).tail(1)

    size = len(f1.columns)
    for i in range(0, len(f2.columns)):
        f1.insert(size + i, f2.columns[i], f2.to_numpy().flatten()[i])
    return f1

def mergeCsvPerPair(files1, files2):
    combi=[]
    for i in range(0, len(files1)):
        try:
            combi.append(mergeCSV(files1[i], files2[i]))
        except FileNotFoundError:
            print("Error: file " + files1[i] + " or " + files2[i] + " doesn't exist !!")
    return combi
        

def main(args):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print("Pipeline start at: " + time.asctime())
    timer = Timer(1)
    timer.start(0)

    parser = argparse.ArgumentParser(description='Batch pipeline')
    parser.add_argument('project_path', help='Project name')
    parser.add_argument('data_path', help='Directory with the data files')
    parser.add_argument('--preproc_only', action='store_true', help='Only perform preprocessing step')
    parser.add_argument('--start', default=0, type=int, help='Only perform preprocessing step')
    parser.add_argument('--it', default=0.01, type=float, help='Only perform preprocessing step')
    args = parser.parse_args(args)

    rawSppArgs = {
            "--knn_geofeatures" : 100,
            "--knn_adj" : 10,
            "--lambda_edge_weight" : float(1.),
            "--reg_strength" : float(0.01),
            "--d_se_max" : 0,
            "--voxel_width" : float(0.01),
            "--validationIsTest" :"none",
            "--voxelize" :"none",
            "--parallel" : "none"
    }

    allProjectsPath = []
    jobs = []

    it = rank
    rawSppArgs["--reg_strength"] += float(it) * float(args.it) + float(args.start) * float(args.it)
    rawSppArgs["--reg_strength"] = round(rawSppArgs["--reg_strength"], 2)# Avoid decimal error
    sppArgs = SppArgs(args.project_path, rawSppArgs)
    allProjectsPath.append("projects/" + sppArgs.getProjectPath())
    try:
        os.makedirs("projects/" + sppArgs.getProjectPath())
        os.symlink(args.data_path, "projects/" + sppArgs.getProjectPath() + "/data")
    except FileExistsError:
        print("Project already exist, scan to complete")

    superpointComputation(sppArgs.toList() + ["--format", "laz"])
    if not args.preproc_only:
        #B7NoEnt 
        #train([sppArgs.getProjectPath(), "--epoch", "300", "--resume", "--batch_size", "6", "--parallel", "--loss_weights", "none"])
        #B7
        #train([sppArgs.getProjectPath(), "--epoch", "300", "--resume", "--batch_size", "6", "--parallel"])
        #B7CleanNN200
        #train([sppArgs.getProjectPath(), "--epoch", "300", "--resume", "--batch_size", "6", "--parallel", "--spg_augm_nneigh", "200"])
        #B7NN5e4
        train([sppArgs.getProjectPath(), "--epoch", "300", "--resume", "--batch_size", "6", "--parallel", "--spg_augm_nneigh", "50000"])
        visualize([sppArgs.getProjectPath(), "LPA3-1", "--outType", "spctgde", "--format", "laz"])
    else:
        visualize([sppArgs.getProjectPath(), "LPA3-1", "--outType", "sgde", "--format", "laz"])

    timer.stop(0)
    print(timer.getFormattedTimer(["Total time: "]))

    #import pudb; pudb.set_trace()

    #allProjectsPath = ["projects/" + args.project_path + "/" + x for x in os.listdir("projects/" + args.project_path) if os.path.isdir("projects/" + args.project_path + "/" + x)]
    ## Concat all csv files
    #sppStatTest = [f +"/reports/sppComputation/testStats.csv" for f in allProjectsPath]
    #trainStatTest = [f +"/reports/training/testStats.csv" for f in allProjectsPath]

    #sppStatValidation = [f +"/reports/sppComputation/validationStats.csv" for f in allProjectsPath]
    #trainStatValidation = [f +"/reports/training/validationStats.csv" for f in allProjectsPath]

    #fullReportTest = mergeCsvPerPair(sppStatTest, trainStatTest)
    #fullReportTest = pd.concat(fullReportTest)
    #fullReportTest.to_csv("projects/" + sppArgs.getProjectPath() + "/../full_spp_testDataset.csv", index=False, na_rep='nan', sep=';')

    #fullReportValidation = mergeCsvPerPair(sppStatValidation, trainStatValidation)
    #fullReportValidation = pd.concat(fullReportValidation)
    #fullReportValidation.to_csv("projects/" + sppArgs.getProjectPath() + "/../full_spp_validationDataset.csv", index=False, na_rep='nan', sep=';')

if __name__ == "__main__":
    main(sys.argv[1:])
