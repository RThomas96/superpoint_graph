import os
from glob import glob
from datetime import datetime

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

        # Result hierarchy
        self.predictionFile = self.rootPath + "/results/predictions.h5"
        self.rawPredictionFile = self.rootPath + "/results/rawPredictions.h5"
        self.modelFile = self.rootPath + "/results/model.pth.h5"

        # Datasets hierarchy
        self.dataset = ["test", "train", "validation"]

        # Usefull to choose the right voxel file 
        self.voxelWidth = "0.03"

        self.allDataFileName = {}
        self.allDataFileType = {}
        for folder in self.dataset:
            dataPath = self.rootPath + "/data/" + folder
            self.allDataFileName[folder] = []
            self.allDataFileType[folder] = []
            try:
                allDataLazFiles = glob(dataPath + "/*.laz")
                allDataPlyFiles = glob(dataPath + "/*.ply")
            except OSError:
                print("{} do not exist ! It is needed and contain input point clouds.".format(dataPath))

            for dataFile in allDataLazFiles:
                dataName = os.path.splitext(os.path.basename(dataFile))[0]
                self.allDataFileName[folder].append(dataName)
                self.allDataFileType[folder].append("laz")

            for dataFile in allDataPlyFiles:
                dataName = os.path.splitext(os.path.basename(dataFile))[0]
                self.allDataFileName[folder].append(dataName)
                self.allDataFileType[folder].append("ply")

            if len(self.allDataFileName[folder]) <= 0:
                print("Warning: {} folder is empty or do not contain laz or ply format file".format(folder))
                #raise FileNotFoundError("Data folder is empty or do not contain {} format files".format(dataType))

    def getFilesFromDataset(self, dataset, i):
        # If i is a file name instead of an index
        if isinstance(i, str):
            i = self.allDataFileName[dataset].index(i)
        fileName = self.allDataFileName[dataset][i]
        dataType = self.allDataFileType[dataset][i]
        dataFile = self.rootPath + "/data/" + dataset + "/" + fileName + '.' + dataType
    
        featureFile  = self.rootPath + "/features/" + dataset + "/" + fileName + ".h5" 
        spgFile  = self.rootPath + "/superpoint_graphs/" + dataset + "/" + fileName + ".h5" 
        parseFile  = self.rootPath + "/parsed/" + dataset + "/" + fileName + ".h5"
    
        voxelisedFile  = self.rootPath + "/data/" + dataset + "-voxelised/" + fileName + '-' + self.voxelWidth + '.' + self.outFormat
        #voxelisedFile  = self.rootPath + "/data/voxelised/" + dataset + "/" + fileName + "/" + fileName + "-prunned" + str(args.voxel_width).replace(".", "-") + "." + dataType
        return fileName, dataFile, dataType, voxelisedFile, featureFile, spgFile, parseFile

    def getVisualisationFilesFromDataset(self, dataset, i):
        # If i is a file name instead of an index
        if isinstance(i, str):
            i = self.allDataFileName[dataset].index(i)
        fileName = self.allDataFileName[dataset][i]
        #dataType = self.allDataFileType[dataset][i]
    
        sppFile   = self.rootPath + "/visualisation/superpoints/" + fileName + "_spp." + self.outFormat
        predictionFile   = self.rootPath + "/visualisation/predictions/" + fileName + "_pred." + self.outFormat
        transFile   = self.rootPath + "/visualisation/features/" + fileName + "_trans." + self.outFormat
        geofFile   = self.rootPath + "/visualisation/features/" + fileName + "_geof." + self.outFormat
        stdFile   = self.rootPath + "/visualisation/features/" + fileName  + "_std." + self.outFormat
        confidencePredictionFile   = self.rootPath + "/visualisation/predictions/" + fileName  + "_conf." + self.outFormat

        return sppFile, predictionFile, transFile, geofFile, stdFile, confidencePredictionFile

    def getNbFiles(self, dataset):
        return len(self.allDataFileName[dataset])

    def createDirForSppComputation(self):
        for sub in ["/features", "/superpoint_graphs", "/parsed"] : 
            mkdirIfNotExist(self.rootPath + sub)
            for subsub in self.dataset : 
                mkdirIfNotExist(self.rootPath + sub + "/" + subsub)
        mkdirIfNotExist(self.reportPath)
        mkdirIfNotExist(self.sppCompReportPath)
        mkdirIfNotExist(self.trainingReportPath)
        mkdirIfNotExist(self.localReportPath)
        mkdirIfNotExist(self.rootPath + "/results")

    def saveGeneralReport(self, formattedReport):
        print("Save report")
        file = open(self.generalReport, "w")
        file.write(formattedReport)
        file.close()

    def getSppCompCsvReport(self, dataset):
        return self.sppCompReportPath + "/" + dataset + "Stats.csv"

    def getTrainingCsvReport(self, dataset):
        return self.trainingReportPath + "/" + dataset + "Stats.csv"
