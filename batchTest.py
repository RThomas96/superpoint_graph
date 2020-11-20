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

def main(args):
    parser = argparse.ArgumentParser(description='Batch pipeline')
    parser.add_argument('project_path', help='Project name')
    parser.add_argument('data_path', help='Directory with the data files')
    args = parser.parse_args(args)

    rawSppArgs = {
            "--knn_geofeatures" : 100,
            "--knn_adj" : 10,
            "--lambda_edge_weight" : float(1.),
            "--reg_strength" : float(0.03),
            "--d_se_max" : 0,
            "--voxel_width" : float(0.03),
            "--voxelize" : "none"
    }

    allProjectsPath = []
    for i in range(1, 10):
        rawSppArgs["--reg_strength"] += float(0.01)
        rawSppArgs["--reg_strength"] = round(rawSppArgs["--reg_strength"], 2)# Avoid decimal error
        sppArgs = SppArgs(args.project_path, rawSppArgs)
        allProjectsPath.append("projects/" + sppArgs.getProjectPath())
        try:
            os.makedirs("projects/" + sppArgs.getProjectPath())
            os.symlink(args.data_path, "projects/" + sppArgs.getProjectPath() + "/data")

            superpointComputation(sppArgs.toList())
            train([sppArgs.getProjectPath(), "--epoch", "10"])
            visualize([sppArgs.getProjectPath(), "test", "LPA3-1", "--outType", "sptgd"])

        except FileExistsError:
            print("This project already exist, pass")

    # Concat all csv files
    sppStatTest = [f +"/reports/sppComputation/statsTest.csv" for f in allProjectsPath]
    trainStatTest = [f +"/reports/training/statsTest.csv" for f in allProjectsPath]

    #import pudb; pudb.set_trace()

    combi=[]
    for x in range(0, len(sppStatTest)):

        f1 = pd.read_csv(sppStatTest[x], sep=';', index_col=False)
        f2 = pd.read_csv(trainStatTest[x], sep=';', index_col=False).tail(1)

        size = len(f1.columns)
        for i in range(0, len(f2.columns)):
            f1.insert(size + i, f2.columns[i], f2.to_numpy().flatten()[i])
        combi.append(f1)
        
    final = pd.concat(combi)
    final.to_csv("projects/" + sppArgs.getProjectPath() + "/../full_spp_testDataset.csv", index=False, na_rep='nan', sep=';')

    #combined_csv = pd.concat([pd.read_csv(f +"/reports/sppComputation/statsTraining.csv") for f in allProjectsPath])
    #combined_csv.to_csv("projects/" + sppArgs.getProjectPath() + "/../full_spp_trainDataset.csv", index=False)
    
if __name__ == "__main__":
    main(sys.argv[1:])
