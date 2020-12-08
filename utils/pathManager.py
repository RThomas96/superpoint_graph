import os
from glob import glob
from datetime import datetime
import json
from collections import defaultdict

def mkdirIfNotExist(dir):
    if not os.path.isdir(dir): 
        print("Create empty dir: " + dir)
        os.mkdir(dir)

class PathManager : 
    def __init__(self, projectName, format = "laz"):
        self.rootPath = os.path.dirname(os.path.realpath(__file__)) + '/../projects/' + projectName 
        if not os.path.isdir(self.rootPath):
            raise NameError('The root subfolder you indicate doesn\'t exist')

        self.outFormat = format

        # Report hierarchy
        self.reportPath = self.rootPath + "/reports"
        self.sppCompReportPath = self.reportPath + "/sppComputation"
        self.trainingReportPath = self.reportPath + "/training"

        self.localReportPath = self.sppCompReportPath + "/" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.generalReport = self.localReportPath + "/generalReport.report"
        self.timeReport= self.localReportPath + "/sppComputationBenchmark.report"

        # Usefull to choose the right voxel file 
        self.voxelWidth = "0.03"

        self.allDataFileName = {}
        self.allDataFileType = {}

        dataPath = self.rootPath + "/data/"
        self.allDataFileName = []
        self.allDataFileType = []
        try:
            allDataLazFiles = glob(dataPath + "/*.laz")
            allDataPlyFiles = glob(dataPath + "/*.ply")
        except OSError:
            print("{} do not exist ! It is needed and contain input point clouds.".format(dataPath))

        for dataFile in allDataLazFiles:
            dataName = os.path.splitext(os.path.basename(dataFile))[0]
            self.allDataFileName.append(dataName)
            self.allDataFileType.append("laz")

        for dataFile in allDataPlyFiles:
            dataName = os.path.splitext(os.path.basename(dataFile))[0]
            self.allDataFileName.append(dataName)
            self.allDataFileType.append("ply")

        self.allDataDataset = []
        with open(dataPath + "datasetConfig") as json_file:
            data = json.load(json_file)
            for runId, run in data.items():
                self.allDataDataset.append(defaultdict(list))
                for dataset, files in run.items():
                    for file in files:
                        self.allDataDataset[-1][dataset].append(file)

    def getModelFile(self, i):
        return self.rootPath + "/results/" + str(i) + "/model.pth.h5"

    def getPredictionFile(self, i):
        return self.rootPath + "/results/" + str(i) + "/predictions.h5"

    def getRawPredictionFile(self, i):
        return self.rootPath + "/results/" + str(i) + "/rawPredictions.h5"

    def getNbRun(self):
        return len(self.allDataDataset)

    def getFilesFromDataset(self, i):
        # If i is a file name instead of an index
        if isinstance(i, str):
            name = os.path.splitext(i)[0]
            i = self.allDataFileName.index(name)
        fileName = self.allDataFileName[i]
        dataType = self.allDataFileType[i]
        dataFile = self.rootPath + "/data/" + fileName + '.' + dataType
    
        featureFile  = self.rootPath + "/features/" + fileName + ".h5" 
        spgFile  = self.rootPath + "/superpoint_graphs/" + fileName + ".h5" 
        parseFile  = self.rootPath + "/parsed/" + fileName + ".h5"
    
        voxelisedFile  = self.rootPath + "/data/voxelised/" + fileName + '-' + self.voxelWidth + '.' + self.outFormat
        #voxelisedFile  = self.rootPath + "/data/voxelised/" + dataset + "/" + fileName + "/" + fileName + "-prunned" + str(args.voxel_width).replace(".", "-") + "." + dataType
        return fileName, dataFile, dataType, voxelisedFile, featureFile, spgFile, parseFile

    def getVisualisationFilesFromDataset(self, i, runIndex):
        # If i is a file name instead of an index
        if isinstance(i, str):
            i = self.allDataFileName.index(i)
        fileName = self.allDataFileName[i]
        #dataType = self.allDataFileType[dataset][i]
    
        sppFile   = self.rootPath + "/visualisation/superpoints/" + fileName + "_spp." + self.outFormat
        predictionFile   = self.rootPath + "/visualisation/predictions/" + fileName + "_" + str(runIndex) + "_pred." + self.outFormat
        transFile   = self.rootPath + "/visualisation/features/" + fileName + "_trans." + self.outFormat
        geofFile   = self.rootPath + "/visualisation/features/" + fileName + "_geof." + self.outFormat
        stdFile   = self.rootPath + "/visualisation/features/" + fileName  + "_std." + self.outFormat
        confidencePredictionFile   = self.rootPath + "/visualisation/features/" + fileName + "_" + str(runIndex) + "_conf." + self.outFormat

        return sppFile, predictionFile, transFile, geofFile, stdFile, confidencePredictionFile

    def getNbFiles(self):
        return len(self.allDataFileName)

    def createDirForSppComputation(self):
        for sub in ["/features", "/superpoint_graphs", "/parsed"] : 
            mkdirIfNotExist(self.rootPath + sub)
        mkdirIfNotExist(self.reportPath)
        mkdirIfNotExist(self.rootPath + "/data/voxelised")
        mkdirIfNotExist(self.sppCompReportPath)
        mkdirIfNotExist(self.trainingReportPath)
        mkdirIfNotExist(self.localReportPath)
        mkdirIfNotExist(self.rootPath + "/results")
        for i in range(self.getNbRun()):
            mkdirIfNotExist(self.rootPath + "/results/" + str(i))
            mkdirIfNotExist(self.trainingReportPath + "/" + str(i))

    def saveGeneralReport(self, formattedReport):
        print("Save report")
        file = open(self.generalReport, "w")
        file.write(formattedReport)
        file.close()

    def getSppCompCsvReport(self, file):
        return self.sppCompReportPath + "/" + file + "Stats.csv"

    def getTrainingCsvReport(self, dataset, i):
        return self.trainingReportPath + "/" + str(i) + "/" + dataset  + "Stats.csv"

    def getMeanTrainingCsvReport(self, dataset):
        return self.trainingReportPath + "/" + dataset + "Stats.csv"
