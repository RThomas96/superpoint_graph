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
    # Multi-class confusion matrix with format
    #
    #                predictions
    #                     v
    #                [[x, x, x],
    #  groundtruth >  [x, x, x],
    #                 [x, x, x]]
    #
    def __init__(self, number_of_labels = 2):
        self.number_of_labels = number_of_labels
        self.confusionMatrix = np.matrix(np.zeros(shape=(self.number_of_labels,self.number_of_labels)))

    def addPrediction(self, prediction, groundTruth):
        #CM = np.matrix(confusion_matrix(prediction, groundTruth))
        #self.confusionMatrix += CM
        for i, pred in enumerate(prediction):
            self.addBatchPrediction(pred, 1, groundTruth[i])

    # Add "nbPrediction" times the value "prediction" into the confusion matrix, if the groundTruth is "ground truth"
    def addBatchPrediction(self, prediction, nbPrediction, groundTruth):
        self.confusionMatrix[groundTruth, prediction] += nbPrediction

    # Add a prediction when the ground truth is a vector
    def addBatchPredictionVec(self, prediction, groundTruth):
        self.confusionMatrix.getA()[:, prediction] += groundTruth

    def getNbValues(self):
        return self.confusionMatrix.sum()

    def getNbPredictionsPerClass(self):
        return self.confusionMatrix.sum(axis=0)

    def getNbGroundTruthPerClass(self):
        return self.confusionMatrix.sum(axis=1)

    def getTruePositivPerClass(self):
        return np.diagonal(self.confusionMatrix)

    def getTruePositiv(self, label):
        return self.getTruePositivPerClass()[label]

    def getFalsePositivPerClass(self):
        res = np.copy(self.confusionMatrix)
        res[np.diag_indices_from(res)] = 0.
        return res.sum(axis=0)

    def getFalseNegativPerClass(self):
        res = np.copy(self.confusionMatrix)
        res[np.diag_indices_from(res)] = 0.
        return res.sum(axis=1)

    def getFalsePositiv(self, label):
        return self.getFalsePositivPerClass()[label]

    def getAccuracy(self): 
        return (self.getTruePositivPerClass().sum() / self.getNbValues())

    # precision = tp / (tp + fp)
    def getPrecisionPerClass(self): 
        all = self.getTruePositivPerClass() + self.getFalsePositivPerClass()
        res = (self.getTruePositivPerClass() / all)
        return self.removeNanForUsedClass(res)

    # recall = tp / (tp + fn)
    def getRecallPerClass(self): 
        all = self.getTruePositivPerClass() + self.getFalseNegativPerClass()
        res = (self.getTruePositivPerClass() / all)
        return self.removeNanForUsedClass(res)

    # iou = tp / (tp + fp + fn)
    def getIoUPerClass(self):
        all = self.getTruePositivPerClass() + self.getFalsePositivPerClass() + self.getFalseNegativPerClass()
        return [float(val) / all[i] for i, val in enumerate(self.getTruePositivPerClass()) ]

    def getAvgIoU(self):
        #from pudb import set_trace; set_trace()
        values = np.array(self.getIoUPerClass())
        values[self.getNotUseClass()] = 0 # We do not want Nan values
        nb_class_seen = len(self.getUsedClass()) 
        return (sum(values) / nb_class_seen)

    def getAvgRecall(self):
        values = np.array(self.getRecallPerClass())
        values[self.getNotUseClass()] = 0 # We do not want Nan values
        nb_class_seen = len(self.getUsedClass()) 
        return (sum(values) / nb_class_seen)

    def getAvgPrecision(self):
        values = np.array(self.getPrecisionPerClass())
        values[self.getNotUseClass()] = 0 # We do not want Nan values
        nb_class_seen = len(self.getUsedClass()) 
        return (sum(values) / nb_class_seen)

    def getStats(self):
        return self.getAccuracy(), self.getAvgIoU(), self.getAvgPrecision(), self.getAvgRecall(), self.getIoUPerClass(), self.getPrecisionPerClass(), self.getRecallPerClass()

    # Return classes without any prediction or ground truth
    def getNotUseClass(self):
        row = np.array(self.confusionMatrix).sum(axis=1)
        column = np.array(self.confusionMatrix).sum(axis=0)
        add = row + column
        return np.where(add == 0)[0]

    def getUsedClass(self):
        row = np.array(self.confusionMatrix).sum(axis=1)
        column = np.array(self.confusionMatrix).sum(axis=0) 
        add = row + column
        return np.where(add != 0)[0]

    def removeNanForUsedClass(self, vec):
        for i in self.getUsedClass():
            if math.isnan(vec[i]):
                vec[i] = 0
        return vec
