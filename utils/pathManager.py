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
    def __init__(self, projectName, sppCompRootPath = "", format = "laz"):
        self.rootPath = os.path.dirname(os.path.realpath(__file__)) + '/../projects/' + projectName 
        if not os.path.isdir(self.rootPath):
            raise NameError('The root subfolder you indicate doesn\'t exist')

        if sppCompRootPath == "":
            self.sppCompRootPath = self.rootPath
            self.trainingRootPath = self.rootPath
        else:
            self.sppCompRootPath = sppCompRootPath
            self.trainingRootPath = self.rootPath

        self.outFormat = format

        #SppComp
        self.sppCompReportPath = self.sppCompRootPath + "/reports/sppComputation"
        self.voxelWidth = "0.03"
        dataPath = self.sppCompRootPath + "/data/"
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

        #Training
        self.trainingReportPath = self.trainingRootPath + "/reports/training"

        self.allDataDataset = []
        with open(dataPath + "datasetConfig") as json_file:
            data = json.load(json_file)
            for runId, run in data.items():
                self.allDataDataset.append(defaultdict(list))
                for dataset, files in run.items():
                    for file in files:
                        self.allDataDataset[-1][dataset].append(file)

    def getModelFile(self, i):
        return self.trainingRootPath + "/results/" + str(i) + "/model.pth.h5"

    def getPredictionFile(self, i):
        return self.trainingRootPath + "/results/" + str(i) + "/predictions.h5"

    def getRawPredictionFile(self, i):
        return self.trainingRootPath + "/results/" + str(i) + "/rawPredictions.h5"

    def getNbRun(self):
        return len(self.allDataDataset)

    def getFilesFromDataset(self, i):
        # If i is a file name instead of an index
        if isinstance(i, str):
            name = os.path.splitext(i)[0]
            i = self.allDataFileName.index(name)
        fileName = self.allDataFileName[i]
        dataType = self.allDataFileType[i]
        dataFile = self.sppCompRootPath + "/data/" + fileName + '.' + dataType
    
        featureFile  = self.sppCompRootPath + "/features/" + fileName + ".h5" 
        spgFile  = self.sppCompRootPath + "/superpoint_graphs/" + fileName + ".h5" 
        parseFile  = self.sppCompRootPath + "/parsed/" + fileName + ".h5"
    
        voxelisedFile  = self.sppCompRootPath + "/data/voxelised/" + fileName + '-' + self.voxelWidth + '.' + self.outFormat
        #voxelisedFile  = self.rootPath + "/data/voxelised/" + dataset + "/" + fileName + "/" + fileName + "-prunned" + str(args.voxel_width).replace(".", "-") + "." + dataType
        return fileName, dataFile, dataType, voxelisedFile, featureFile, spgFile, parseFile

    def getVisualisationFilesFromDataset(self, i, runIndex):
        # If i is a file name instead of an index
        if isinstance(i, str):
            i = self.allDataFileName.index(i)
        fileName = self.allDataFileName[i]
        #dataType = self.allDataFileType[dataset][i]
    
        sppFile   = self.sppCompRootPath + "/visualisation/superpoints/" + fileName + "_spp." + self.outFormat
        predictionFile   = self.trainingRootPath + "/visualisation/predictions/" + fileName + "_" + str(runIndex) + "_pred." + self.outFormat
        transFile   = self.sppCompRootPath + "/visualisation/features/" + fileName + "_trans." + self.outFormat
        geofFile   = self.sppCompRootPath + "/visualisation/features/" + fileName + "_geof." + self.outFormat
        stdFile   = self.sppCompRootPath + "/visualisation/features/" + fileName  + "_std." + self.outFormat
        confidencePredictionFile   = self.trainingRootPath + "/visualisation/features/" + fileName + "_" + str(runIndex) + "_conf." + self.outFormat
        elevationFile   = self.sppCompRootPath + "/visualisation/features/" + fileName  + "_elevation." + self.outFormat

        return sppFile, predictionFile, transFile, geofFile, stdFile, confidencePredictionFile, elevationFile

    def getNbFiles(self):
        return len(self.allDataFileName)

    def createDirForSppComputation(self):
        for sub in ["/features", "/superpoint_graphs", "/parsed"] : 
            mkdirIfNotExist(self.sppCompRootPath + sub)
        mkdirIfNotExist(self.sppCompRootPath + "/reports")
        mkdirIfNotExist(self.sppCompReportPath)
        mkdirIfNotExist(self.sppCompRootPath + "/data/voxelised")

    def createDirForTraining(self):
        mkdirIfNotExist(self.trainingRootPath + "/reports")
        mkdirIfNotExist(self.trainingReportPath)
        mkdirIfNotExist(self.trainingRootPath + "/results")
        for i in range(self.getNbRun()):
            mkdirIfNotExist(self.trainingRootPath + "/results/" + str(i))
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
