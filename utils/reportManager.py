import numpy as np
from pathlib import Path
from datetime import datetime
from operator import truediv
from collections import Counter
from colorLabelManager import ColorLabelManager

class ReportManager:
    def __init__(self, rootPath, args):
        self.filePath = rootPath + "/reports/" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ".report"
        Path(rootPath + "/reports").mkdir(parents=True, exist_ok=True)
        " Indicate if next values added to the class will be from from training data "
        self.train = True

        " Parameters "
        self.regStrength = args.reg_strength 
        self.knnGeo = args.knn_geofeatures 
        self.knnAdj = args.knn_adj

        " First value are from data training, second one from data test "
        self.nbSuperpoints = [0, 0]
        self.nbOfPoint = [0, 0]
        self.avgNbOfPointPerSpp = [0, 0]
        self.nbOfSppPerClass = [{}, {}] 

    " Add a value at the right index whether or not it's from training data "
    def addValue(self, val, toAdd):
        val[0 if self.train else 1] += toAdd

    def assignValue(self, val, toAdd):
        val[0 if self.train else 1] = toAdd

    def computeStatsOnSpp(self, components, graph):
        self.addValue(self.nbSuperpoints, len(components))
        for spp in components: self.addValue(self.nbOfPoint, len(spp))
        labelOfEachPointPerSpp = np.array(graph["sp_labels"])
        labelOfEachSpp = labelOfEachPointPerSpp.argmax(1)
        self.assignValue(self.nbOfSppPerClass, Counter(labelOfEachSpp))

    def averageComputations(self):
        self.avgNbOfPointPerSpp = list(map(truediv, self.nbOfPoint, self.nbSuperpoints))

    def renameDict(self):
        colorLabelManager = ColorLabelManager()
        nameDict = colorLabelManager.nameDict
        for i in [0, 1]:
            for key in nameDict.keys():
                try:
                    self.nbOfSppPerClass[i][nameDict[key]] = self.nbOfSppPerClass[i].pop(key)
                # If a label hasn't any point, there is no entry in nbOfSppPerClass because of Counter()
                # We don't need to rename it
                except KeyError:
                    pass

    def saveReport(self):
        self.averageComputations()
        self.renameDict()
        report = ""
        report += "# Superpoint computation report\n"
        report += datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
        report += "\n"
        report += "First value is training data, second one is testing\n"
        report += "\n"
        report += "## Parameters\n"
        report += "\n"
        report += "Regularization strength: {}\n".format(self.regStrength)
        report += "Knn geometric features: {}\n".format(self.knnGeo)
        report += "Knn adjacency graph: {}\n".format(self.knnAdj)
        report += "\n"
        report += "## General analysis\n"
        report += "\n"
        report += "Number of points: {} \n".format(self.nbOfPoint)
        report += "\n"
        report += "## Superpoints analysis\n"
        report += "\n"
        report += "Number of superpoints: {} \n".format(self.nbSuperpoints)
        report += "Average number of points per superpoints: {} \n".format(self.avgNbOfPointPerSpp)
        report += "\n"
        report += "Number of superpoint per class:\n"
        report += "Training: {} \n".format(self.nbOfSppPerClass[0])
        report += "Testing: {} \n".format(self.nbOfSppPerClass[1])

        print("Save report")
        file = open(self.filePath, "w")
        file.write(report)
        file.close()