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
        CM = np.matrix(confusion_matrix(predictions, groundTruth))
        self.confusionMatrix += CM

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
    def getPrecisionPerClass(self, withoutNan = False): 
        all = self.getTruePositivPerClass() + self.getFalsePositivPerClass()
        if withoutNan: 
            all[all == 0] = 1
        return (self.getTruePositivPerClass() / all)

    # recall = tp / (tp + fn)
    def getRecallPerClass(self, withoutNan = False): 
        all = self.getTruePositivPerClass() + self.getFalseNegativPerClass()
        if withoutNan: 
            all[all == 0] = 1
        return (self.getTruePositivPerClass() / all)

    # iou = tp / (tp + fp + fn)
    def getIoUPerClass(self, withoutNan = False):
        all = self.getTruePositivPerClass() + self.getFalsePositivPerClass() + self.getFalseNegativPerClass()
        if withoutNan:
            all[all == 0] = 1
        return (self.getTruePositivPerClass() / all )

    def getAvgIoU(self):
        values = self.getIoUPerClass(withoutNan=True)
        valWithNan = self.getIoUPerClass(withoutNan=False)
        # A class with 0 ground truth point has nan IoU
        nb_class_seen = len([i for i in valWithNan if not math.isnan(i)]) 
        return (sum(values) / nb_class_seen)

    def getAvgRecall(self):
        values = self.getRecallPerClass(withoutNan=True)
        valWithNan = self.getRecallPerClass(withoutNan=False)
        # A class with 0 ground truth point has nan Recall
        nb_class_seen = len([i for i in valWithNan if not math.isnan(i)]) 
        return (sum(values) / nb_class_seen)

    def getAvgPrecision(self):
        values = self.getPrecisionPerClass(withoutNan=True)
        valWithNan = self.getPrecisionPerClass(withoutNan=False)
        # A class with 0 ground truth point has nan Precision
        nb_class_seen = len([i for i in valWithNan if not math.isnan(i)]) 
        #from pudb import set_trace; set_trace()
        return (sum(values) / nb_class_seen)

    def getAccuracyPerClass(self, withoutNan=True):
        return self.getIoUPerClass(withoutNan)
