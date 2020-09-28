import pdb
import csv
import os.path
import numpy as np
from pathlib import Path
from datetime import datetime
from operator import truediv
from collections import Counter
from colorLabelManager import ColorLabelManager

class ReportManager:
    def __init__(self, rootPath, args):
        self.filePath = rootPath + "/reports/" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ".report"
        self.csvPathTraining = rootPath + "/reports/statsTraining.csv"
        self.csvPathTesting = rootPath + "/reports/statsTesting.csv"
        Path(rootPath + "/reports").mkdir(parents=True, exist_ok=True)
        " Indicate if next values added to the class will be from from training data "
        self.train = True

        " Parameters "
        self.regStrength = args.reg_strength 
        self.knnGeo = args.knn_geofeatures 
        self.knnAdj = args.knn_adj
        self.lambdaWeight = args.lambda_edge_weight

        " First value are from data training, second one from data test "
        self.nbSuperpoints = [0, 0]
        self.nbOfPoint = [0, 0]
        self.avgNbOfPointPerSpp = [0, 0]
        self.nbOfSppPerClass = [{}, {}] 
        self.accuracy = [0, 0]
        self.accuracyPerClass = [{}, {}] 

    " Add a value at the right index whether or not it's from training data "
    def addValue(self, val, toAdd):
        val[0 if self.train else 1] += toAdd

    def assignValue(self, val, toAdd):
        val[0 if self.train else 1] = toAdd

    def getValue(self, val):
        return val[0 if self.train else 1]

    def getCsvPath(self):
        return self.csvPathTraining if self.train else self.csvPathTesting

    def computeStatsOnSpp(self, components, graph):
        self.addValue(self.nbSuperpoints, len(components))
        for spp in components: self.addValue(self.nbOfPoint, len(spp))
        nbPtPerLabelForEachSpp = np.array(graph["sp_labels"])
        # Search index of the maximum value for each spp i.e the label in majority 
        labelOfEachSpp = nbPtPerLabelForEachSpp.argmax(1)

        self.assignValue(self.nbOfSppPerClass, Counter(labelOfEachSpp))
    
        minorityLabels=np.copy(nbPtPerLabelForEachSpp)
        for i, idx in enumerate(labelOfEachSpp):
            minorityLabels[i][idx] = 0 

        wrongPt = np.sum(minorityLabels)
        wrongPtPerClass = np.sum(minorityLabels, axis=0)

        nbWrongPt = ((self.getValue(self.nbOfPoint) - wrongPt)/self.getValue(self.nbOfPoint))*100.
        self.assignValue(self.accuracy, nbWrongPt)

        nbPtPerLabel = np.sum(nbPtPerLabelForEachSpp, axis=0)

        nbPtPerLabel[nbPtPerLabel == 0] = 1
        accPerLabel = (nbPtPerLabel - wrongPtPerClass)/nbPtPerLabel

        dictAccPerLabel = {}
        for i, val in enumerate(accPerLabel):
            dictAccPerLabel[i] = val*100.
            #elif nbPtPerLabel[i] > 0:
            #    pdb.set_trace()
            #    dictAccPerLabel[i] = 0

        self.assignValue(self.accuracyPerClass, dictAccPerLabel)


    def averageComputations(self):
        self.avgNbOfPointPerSpp = list(map(truediv, self.nbOfPoint, self.nbSuperpoints))

    def renameDict(self):
        colorLabelManager = ColorLabelManager()
        nameDict = colorLabelManager.nameDict
        for i in [0, 1]:
            for key in nameDict.keys():
                try:
                    self.nbOfSppPerClass[i][nameDict[key]] = self.nbOfSppPerClass[i].pop(key)
                except KeyError:
                    # If a label hasn't any point, there is no entry in nbOfSppPerClass because of Counter()
                    # We don't need to rename it
                    # But we want to keep the value
                    self.nbOfSppPerClass[i][nameDict[key]] = 0
                    pass
                try:
                    self.accuracyPerClass[i][nameDict[key]] = self.accuracyPerClass[i].pop(key)
                except KeyError:
                    print("Error: miss value in accuracy per class")

    # Pre-requisite: dictionary renamed
    def getStat(self):
        header = list()
        header.append("Regularization strength")
        header.append("Lambda edge weight")
        header.append("Knn geometric features")
        header.append("Knn adjacency graph")
        header.append("Total number of points")
        header.append("Total accuracy")
        for name in list(self.getValue(self.accuracyPerClass).keys()):
            header.append(name)
        header.append("Number of superpoints")
        header.append("Avg number of points per superpoint")
        for name in list(self.getValue(self.nbOfSppPerClass).keys()):
            header.append(name)

        stat = list()
        stat.append(self.regStrength)
        stat.append(self.lambdaWeight)
        stat.append(self.knnGeo)
        stat.append(self.knnAdj)
        stat.append(self.getValue(self.nbOfPoint))
        stat.append(self.getValue(self.accuracy))
        stat = stat + list(self.getValue(self.accuracyPerClass).values())
        stat.append(self.getValue(self.nbSuperpoints))
        stat.append(self.getValue(self.avgNbOfPointPerSpp))
        stat = stat + list(self.getValue(self.nbOfSppPerClass).values())

        return [header, stat]


    def saveStat(self):
        self.train = True
        csvFile1 = self.getStat()
        self.train = False
        csvFile2 = self.getStat()

        isFile = os.path.isfile(self.csvPathTraining)
        with open(self.csvPathTraining, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if not isFile:
                spamwriter.writerow(csvFile1[0])
            spamwriter.writerow(csvFile1[1])

        isFile = os.path.isfile(self.csvPathTesting)
        with open(self.csvPathTesting, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if not isFile:
                spamwriter.writerow(csvFile2[0])
            spamwriter.writerow(csvFile2[1])

    def saveReport(self):
        self.averageComputations()
        self.renameDict()
        self.saveStat()
        report = ""
        report += "# Superpoint computation report\n"
        report += datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
        report += "\n"
        report += "First value is training data, second one is testing\n"
        report += "\n"
        report += "## Parameters\n"
        report += "\n"
        report += "Regularization strength: {}\n".format(self.regStrength)
        report += "Lambda edge weight: {}\n".format(self.lambdaWeight)
        report += "Knn geometric features: {}\n".format(self.knnGeo)
        report += "Knn adjacency graph: {}\n".format(self.knnAdj)
        report += "\n"
        report += "## General analysis\n"
        report += "\n"
        report += "Number of points: {} \n".format(self.nbOfPoint)
        report += "Total accuracy : {} \n".format(self.accuracy)
        report += "Accuracy per class:\n"
        report += "Training: {} \n".format(self.accuracyPerClass[0])
        report += "Testing: {} \n".format(self.accuracyPerClass[1])
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
