import os
from glob import glob

class PathManager : 
    def __init__(self, projectName):
        self.rootPath = os.path.dirname(os.path.realpath(__file__)) + '/../projects/' + projectName 
        if not os.path.isdir(self.rootPath):
            raise NameError('The root subfolder you indicate doesn\'t exist')

        dataFolder = self.rootPath + "/data/raw/"
        if not os.path.isdir(self.rootPath):
            raise NameError('The data folder doesn\'t exist or is empty')

        self.folders = ["test", "train"]

        self.allDataFileName = {}
        self.allDataFileType = {}
        for folder in self.folders:
            dataPath = dataFolder + folder
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
