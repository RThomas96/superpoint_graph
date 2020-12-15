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
import time
from timer import Timer

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

    parser = argparse.ArgumentParser(description='Batch pipeline')
    parser.add_argument('project_path', help='Project name')
    args = parser.parse_args(args)

    print("Merging csv from " + args.project_path + " projects")

    allProjectsPath = ["projects/" + args.project_path + "/" + x for x in os.listdir("projects/" + args.project_path) if os.path.isdir("projects/" + args.project_path + "/" + x)]
    # Concat all csv files
    sppStatTest = [f +"/reports/sppComputation/testStats.csv" for f in allProjectsPath]
    trainStatTest = [f +"/reports/training/testStats.csv" for f in allProjectsPath]

    sppStatValidation = [f +"/reports/sppComputation/validationStats.csv" for f in allProjectsPath]
    trainStatValidation = [f +"/reports/training/validationStats.csv" for f in allProjectsPath]

    fullReportTest = mergeCsvPerPair(sppStatTest, trainStatTest)
    fullReportTest = pd.concat(fullReportTest).drop_duplicates()
    fullReportTest.to_csv("full_spp_testDataset.csv", index=False, na_rep='nan', sep=';')

    fullReportValidation = mergeCsvPerPair(sppStatValidation, trainStatValidation)
    fullReportValidation = pd.concat(fullReportValidation).drop_duplicates()
    fullReportValidation.to_csv("full_spp_validationDataset.csv", index=False, na_rep='nan', sep=';')

if __name__ == "__main__":
    main(sys.argv[1:])
