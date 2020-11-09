import os
import csv
from glob import glob
from datetime import datetime

def mkdirIfNotExist(dir):
    if not os.path.isdir(dir): 
        print("Create empty dir: " + dir)
        os.mkdir(dir)

class PathManager : 
    def __init__(self, projectName):
        self.rootPath = os.path.dirname(os.path.realpath(__file__)) + '/../projects/' + projectName 
        if not os.path.isdir(self.rootPath):
            raise NameError('The root subfolder you indicate doesn\'t exist')

        # Report hierarchy
        self.reportPath = self.rootPath + "/reports"
        self.sppCompReportPath = self.reportPath + "/sppComputation"
        self.trainingReportPath = self.reportPath + "/training"

        self.localReportPath = self.sppCompReportPath + "/" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.generalReport= self.localReportPath + "/generalReport.report"
        self.timeReport= self.localReportPath + "/sppComputationBenchmark.report"

        self.sppCompTrainingCsv = self.sppCompReportPath + "/statsTraining.csv"
        self.sppCompTestingCsv = self.sppCompReportPath + "/statsTesting.csv"


        self.dataset = ["test", "train"]

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
        fileName = self.allDataFileName[dataset][i]
        dataType = self.allDataFileType[dataset][i]
        dataFile = self.rootPath + "/data/" + dataset + "/" + fileName + '.' + dataType
    
        featureFile  = self.rootPath + "/features/" + dataset + "/" + fileName + ".h5" 
        spgFile  = self.rootPath + "/superpoint_graphs/" + dataset + "/" + fileName + ".h5" 
        parseFile  = self.rootPath + "/parsed/" + dataset + "/" + fileName + ".h5"
    
        voxelisedFile  = self.rootPath + "/data/" + dataset + "-voxelised/" + fileName + '.' + dataType
        #voxelisedFile  = self.rootPath + "/data/voxelised/" + dataset + "/" + fileName + "/" + fileName + "-prunned" + str(args.voxel_width).replace(".", "-") + "." + dataType
        return fileName, dataFile, dataType, voxelisedFile, featureFile, spgFile, parseFile

    def getNbFiles(self, dataset):
        return len(self.allDataFileName[dataset])

    def createDirForSppComputation(self):
        for sub in ["/features", "/superpoint_graphs", "/parsed"] : 
            mkdirIfNotExist(self.rootPath + sub)
            for subsub in ["/test", "/train"] : 
                mkdirIfNotExist(self.rootPath + sub + subsub)
        mkdirIfNotExist(self.reportPath)
        mkdirIfNotExist(self.sppCompReportPath)
        mkdirIfNotExist(self.trainingReportPath)
        mkdirIfNotExist(self.localReportPath)

    def saveGeneralReport(self, formattedReport):
        print("Save report")
        file = open(self.generalReport, "w")
        file.write(formattedReport)
        file.close()

    def saveCsvReport(self, report, isTraining):
        if isTraining:
            file = self.sppCompTrainingCsv
        else:
            file = self.sppCompTestingCsv

        isFile = os.path.isfile(file)
        with open(file, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if not isFile:
                spamwriter.writerow(report[0])# report[0] is header
            spamwriter.writerow(report[1])# report[1] is data
