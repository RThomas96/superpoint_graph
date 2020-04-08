import os
from glob import glob

class PathManager : 
    def __init__(self, args, dataType="ply"):
        self.rootPath = os.path.dirname(os.path.realpath(__file__)) + '/../' + args.ROOT_PATH
        if not os.path.isdir(self.rootPath):
            raise NameError('The root subfolder you indicate doesn\'t exist')

        dataFolder = self.rootPath + "/data/raw/"
        if not os.path.isdir(self.rootPath):
            raise NameError('The data folder doesn\'t exist or is empty')

        self.folders = ["test", "train"]

        self.allDataFileName = {}
        for folder in self.folders:
            dataPath = dataFolder + folder
            self.allDataFileName[folder] = []
            try:
                allDataFiles = glob(dataPath + "/*."+ dataType)
            except OSError:
                print("{} do not exist ! It is needed and contain input point clouds.".format(dataPath))
            for dataFile in allDataFiles:
                dataName = os.path.splitext(os.path.basename(dataFile))[0]
                self.allDataFileName[folder].append(dataName)
            if len(self.allDataFileName[folder]) <= 0:
                print("Warning: {} folder is empty or do not contain {} format file".format(folder, dataType))
                #raise FileNotFoundError("Data folder is empty or do not contain {} format files".format(dataType))