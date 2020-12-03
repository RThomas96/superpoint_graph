import os.path
import numpy as np
from pathlib import Path
from operator import truediv
from collections import Counter
from colorLabelManager import ColorLabelManager
from datetime import datetime
from sklearn.metrics import confusion_matrix
from confusionMatrix import ConfusionMatrix
import math

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
        return self.CM.getAccuracy()

    def getAccuracyPerClass(self): 
        return self.CM.getIoUPerClass()

class SPPComputationReportManager:
    def __init__(self, args, nbLabels):
        " Parameters "
        self.regStrength = args.reg_strength 
        self.knnGeo = args.knn_geofeatures 
        self.knnAdj = args.knn_adj
        self.lambdaWeight = args.lambda_edge_weight
        self.voxel_width = args.voxel_width

        self.stats = StatManagerOnSPP(nbLabels)

    def computeStatOnSpp(self, nbPtPerLabelForEachSpp):
        for predicted in nbPtPerLabelForEachSpp:
            groundTruth = predicted.argmax()
            self.stats.addBatchPrediction(predicted, groundTruth)

    def getCsvReport(self):

        label2Name = ColorLabelManager().label2Name

        header = list()
        header.append("Voxel width")
        header.append("Regularization strength")
        header.append("Lambda edge weight")
        header.append("Knn geometric features")
        header.append("Knn adjacency graph")
        header.append("Total number of points")
        header.append("Total accuracy")
        for name in list(label2Name.values()):
            header.append(name)
        header.append("Number of superpoints")
        header.append("Avg number of points per superpoint")
        for name in list(label2Name.values()):
            header.append(name)

        value = list()
        value.append(self.voxel_width)
        value.append(self.regStrength)
        value.append(self.lambdaWeight)
        value.append(self.knnGeo)
        value.append(self.knnAdj)
        value.append(self.stats.getNbPt())
        value.append(self.stats.getTotalAccuracy())
        value = value + list(self.stats.getAccuracyPerClass())
        value.append(self.stats.getNbSpp())
        value.append(self.stats.getAvgNbOfPtPerSpp())
        value = value + list(self.stats.nbSppPerClass)

        return [header, value]
