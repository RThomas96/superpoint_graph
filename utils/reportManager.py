import os.path
import numpy as np
from pathlib import Path
from operator import truediv
from collections import Counter
from colorLabelManager import ColorLabelManager
from datetime import datetime
from sklearn.metrics import confusion_matrix
import math

class ConfusionMatrix:
    def __init__(self, number_of_labels = 2):
        self.number_of_labels = number_of_labels
        self.confusionMatrix = np.matrix(np.zeros(shape=(self.number_of_labels,self.number_of_labels)))

    def addPrediction(self, prediction, groundTruth):
        CM = np.matrix(confusion_matrix(predictions, groundTruth))
        self.confusionMatrix += CM

    # Add "nbPrediction" times the value "prediction" into the confusion matrix, if the groundTruth is "ground truth"
    def addBatchPrediction(self, prediction, nbPrediction, groundTruth):
        self.confusionMatrix[groundTruth, prediction] += nbPrediction

    def getTruePositivPerClass(self):
        return np.diagonal(self.confusionMatrix)

    def getTruePositiv(self, label):
        return self.getTruePositivPerClass()[label]

    def getFalsePositivPerClass(self):
        res = np.copy(self.confusionMatrix)
        res[np.diag_indices_from(res)] = 0.
        return res.sum(axis=0)

    def getFalsePositiv(self, label):
        return self.getFalsePositivPerClass()[label]

    def getTotalAccuracy(self): 
        return (self.getTruePositivPerClass().sum() / self.getNbValues()) * 100

    def getAccuracyPerClass(self): 
        return (self.getTruePositivPerClass() / (self.getTruePositivPerClass() + self.getFalsePositivPerClass())) * 100 

    def getNbValues(self):
        return self.confusionMatrix.sum()

    def getNbValuesPerClass(self):
        return self.confusionMatrix.sum(axis=1)


class StatManagerOnSPP:
    def __init__(self, nbLabels):
        self.nbSuperpoints = 0
        self.nbSppPerClass = np.zeros(nbLabels)
        self.CM = ConfusionMatrix(nbLabels)

    def addBatchPrediction(self, predictedSpp, groundTruth):
        self.nbSuperpoints += 1
        self.nbSppPerClass[groundTruth] += 1
        for label, nb in enumerate(predictedSpp):
            self.CM.addBatchPrediction(label, nb, groundTruth)

    def getAvgNbOfPtPerSpp(self):
        return self.getNbPt() / self.getNbSpp()

    def getNbSpp(self):
        return self.nbSuperpoints

    def getNbPt(self):
        return self.CM.getNbValues()

    def getTotalAccuracy(self): 
        return self.CM.getTotalAccuracy()

    def getAccuracyPerClass(self): 
        return self.CM.getAccuracyPerClass()

class SPPComputationReportManager:
    def __init__(self, rootPath, args, nbLabels):
        " Indicate if next values added to the class will be from from training data "
        self.train = True

        " Parameters "
        self.regStrength = args.reg_strength 
        self.knnGeo = args.knn_geofeatures 
        self.knnAdj = args.knn_adj
        self.lambdaWeight = args.lambda_edge_weight

        self.stats = [StatManagerOnSPP(nbLabels), StatManagerOnSPP(nbLabels)]

    def computeStatOnSpp(self, nbPtPerLabelForEachSpp):
        for predicted in nbPtPerLabelForEachSpp:
            groundTruth = predicted.argmax()
            self.stats[0 if self.train else 1].addBatchPrediction(predicted, groundTruth)

    def getNamedDict(self, values):
        colorLabelManager = ColorLabelManager()
        label2Name = colorLabelManager.label2Name
        renamedDict = {} 
        for i, val in enumerate(values):
            if not math.isnan(val):
                renamedDict[label2Name[i]] = val
        return renamedDict

    def getCsvReport(self, getTraining):
        if getTraining:
            self.train = True
        else:
            self.train = False

        stat = self.stats[0 if self.train else 1]

        header = list()
        header.append("Regularization strength")
        header.append("Lambda edge weight")
        header.append("Knn geometric features")
        header.append("Knn adjacency graph")
        header.append("Total number of points")
        header.append("Total accuracy")
        for name in list(self.getNamedDict(stat.getAccuracyPerClass()).keys()):
            header.append(name)
        header.append("Number of superpoints")
        header.append("Avg number of points per superpoint")
        for name in list(self.getNamedDict(stat.nbSppPerClass).keys()):
            header.append(name)

        value = list()
        value.append(self.regStrength)
        value.append(self.lambdaWeight)
        value.append(self.knnGeo)
        value.append(self.knnAdj)
        value.append(stat.getNbPt())
        value.append(stat.getTotalAccuracy())
        value = value + list(stat.getAccuracyPerClass())
        value.append(stat.getNbSpp())
        value.append(stat.getAvgNbOfPtPerSpp())
        value = value + list(stat.nbSppPerClass)

        return [header, value]


    def getFormattedReport(self):
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
        report += "Number of points: {} \n".format([self.stats[0].getNbPt(), self.stats[1].getNbPt()])
        report += "Total accuracy : {} \n".format([self.stats[0].getTotalAccuracy(), self.stats[1].getTotalAccuracy()])
        report += "Accuracy per class:\n"
        report += "Training: {} \n".format(self.getNamedDict(self.stats[0].getAccuracyPerClass()))
        report += "Testing: {} \n".format(self.getNamedDict(self.stats[1].getAccuracyPerClass()))
        report += "\n"
        report += "## Superpoints analysis\n"
        report += "\n"
        report += "Number of superpoints: {} \n".format([self.stats[0].getNbSpp(), self.stats[1].getNbSpp()])
        report += "Average number of points per superpoints: {} \n".format([self.stats[0].getAvgNbOfPtPerSpp(), self.stats[1].getAvgNbOfPtPerSpp()])
        report += "\n"
        report += "Number of superpoint per class:\n"
        report += "Training: {} \n".format(self.getNamedDict(self.stats[0].nbSppPerClass))
        report += "Testing: {} \n".format(self.getNamedDict(self.stats[1].nbSppPerClass))

        return report
