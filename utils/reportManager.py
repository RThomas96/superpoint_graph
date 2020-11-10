import os.path
import numpy as np
from pathlib import Path
from operator import truediv
from collections import Counter
from colorLabelManager import ColorLabelManager
from datetime import datetime

class ReportManager:
    def __init__(self, rootPath, args, nbLabels):
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
        self.nbOfSppPerClass = np.zeros([2,nbLabels])
        self.wrongPt = [0, 0]
        self.wrongPtPerClass = np.zeros([2,nbLabels]) 
        self.nbPtPerClass = np.zeros([2,nbLabels]) 

        " These values need to be updated "
        self.accuracy = [0, 0]
        self.accuracyPerClass =np.zeros([2,nbLabels]) 
 
    " Add a value at the right index whether or not it's from training data "
    def addValue(self, val, toAdd):
        val[0 if self.train else 1] += toAdd

    def assignValue(self, val, toAdd):
        val[0 if self.train else 1] = toAdd

    def getValue(self, val):
        return val[0 if self.train else 1]

    def computeStatsOnSpp(self, components, nbPtPerLabelForEachSpp):
        tr = 0 if self.train else 1

        components = np.array(components)
        self.addValue(self.nbSuperpoints, len(components))
        for spp in components: self.addValue(self.nbOfPoint, len(spp))
        
        # Search index of the maximum value for each spp i.e the label in majority 
        labelOfEachSpp = nbPtPerLabelForEachSpp.argmax(1)

        #self.assignValue(self.nbOfSppPerClass, Counter(labelOfEachSpp))
        for label in labelOfEachSpp:
            self.nbOfSppPerClass[tr][label] += 1
    
        minorityLabels=np.copy(nbPtPerLabelForEachSpp)
        for i, idx in enumerate(labelOfEachSpp):
            minorityLabels[i][idx] = 0 

        self.wrongPt[tr] += np.sum(minorityLabels)
        self.wrongPtPerClass[tr] += np.sum(minorityLabels, axis=0)

        self.nbPtPerClass[tr] += np.sum(nbPtPerLabelForEachSpp, axis=0) 

        # Update accuracies
        self.accuracy[tr] = ((self.nbOfPoint[tr] - self.wrongPt[tr]) / self.nbOfPoint[tr]) * 100.
        self.accuracyPerClass[tr] = ((self.nbPtPerClass[tr] - self.wrongPtPerClass[tr]) / self.nbPtPerClass[tr])*100.
        for i, val in enumerate(self.nbOfSppPerClass[tr]):
            if val == 0:
                self.accuracyPerClass[tr][i] = 0


    def averageComputations(self):
        self.assignValue(self.avgNbOfPointPerSpp, self.getValue(self.nbOfPoint) / self.getValue(self.nbSuperpoints))

    def renameDict(self):
        colorLabelManager = ColorLabelManager()
        nameDict = colorLabelManager.nameDict
        i = 0 if self.train else 1
        renamedDict = [{}] 
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

    def getNamedDict(self, values, i = -1):
        colorLabelManager = ColorLabelManager()
        nameDict = colorLabelManager.nameDict
        if i == -1:
            i = 0 if self.train else 1
        renamedDict = {} 
        for key in nameDict.keys():
            if values[i][int(key)] > 0:
                renamedDict[nameDict[key]] = values[i][key]
        return renamedDict

    def getCsvReport(self, getTraining):
        if getTraining:
            self.train = True
        else:
            self.train = False
        self.averageComputations()
        #self.renameDict()

        header = list()
        header.append("Regularization strength")
        header.append("Lambda edge weight")
        header.append("Knn geometric features")
        header.append("Knn adjacency graph")
        header.append("Total number of points")
        header.append("Total accuracy")
        for name in list(self.getNamedDict(self.accuracyPerClass).keys()):
            header.append(name)
        header.append("Number of superpoints")
        header.append("Avg number of points per superpoint")
        for name in list(self.getNamedDict(self.nbOfSppPerClass).keys()):
            header.append(name)

        stat = list()
        stat.append(self.regStrength)
        stat.append(self.lambdaWeight)
        stat.append(self.knnGeo)
        stat.append(self.knnAdj)
        stat.append(self.getValue(self.nbOfPoint))
        stat.append(self.getValue(self.accuracy))
        stat = stat + list(self.getValue(self.accuracyPerClass))
        stat.append(self.getValue(self.nbSuperpoints))
        stat.append(self.getValue(self.avgNbOfPointPerSpp))
        stat = stat + list(self.getValue(self.nbOfSppPerClass))

        return [header, stat]


    def getFormattedReport(self):
        self.averageComputations()
        #self.renameDict()
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
        report += "Training: {} \n".format(self.getNamedDict(self.accuracyPerClass, 0))
        report += "Testing: {} \n".format(self.getNamedDict(self.accuracyPerClass, 1))
        report += "\n"
        report += "## Superpoints analysis\n"
        report += "\n"
        report += "Number of superpoints: {} \n".format(self.nbSuperpoints)
        report += "Average number of points per superpoints: {} \n".format(self.avgNbOfPointPerSpp)
        report += "\n"
        report += "Number of superpoint per class:\n"
        report += "Training: {} \n".format(self.getNamedDict(self.nbOfSppPerClass, 0))
        report += "Testing: {} \n".format(self.getNamedDict(self.nbOfSppPerClass, 1))

        return report

    def getPredictionFile(fileName):
        outPredFileName = fileName + "_pred.ply"
        outPredFile   = root + "/visualisation/predictions/" + fileName 
        Path(root + "/visualisation/predictions/" + args.predFileName).mkdir(parents=True, exist_ok=True)


