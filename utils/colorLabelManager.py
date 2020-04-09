import os

class ColorLabelManager:
        def __init__(self):
                #self.filePath = os.getcwd() + "/" + os.path.dirname(__file__) + "/colorCode" 
                self.filePath = "/home/thomas/Data/Cajun/Evaluation/Methods/superpoint_graph/utils/colorCode" 
                self.colorDict, self.labelDict, self.nameDict, self.aggregationDict = self.parseColorFile()
                #self.nbColor = len(self.colorDict)
                self.nbColor = len(self.aggregationDict)
                self.aggregation = True if max([len(x) for x in self.aggregationDict.values()]) > 1 else False
                if self.aggregation:
                        self.aggregateDict()

        def parseColorFile(self):
                colorFile = open(self.filePath, "r") 
                colorDict, labelDict, nameDict, aggregationDict = {}, {}, {}, {}
                # First label is "unknown" label and black color
                # This label is writed by prediction writer, and not in the color file
                labelDict[0] = [0, 0, 0]

                # Name dict is used per per_class_iou, and then do not need to contain 0
                # Note that in report 0 cannot be renamed
                #nameDict[0] = 'Inconnue'
                for i, line in enumerate(colorFile):
                        values = line.split()
                        key = values[1]+values[2]+values[3]
                        # No +1 here cause colorDict is used by upSample script
                        # And there is no unknown label for now in labelised files
                        # To add unknown labels modify here AND AT AGGREGATE LABEL THAT ADD +1
                        colorDict[key] = values[0]
                        # +1 here cause first label is "unknown" label 
                        labelDict[i+1] = [values[1], values[2], values[3]]
                        nameDict[i] = values[7]
                        aggregationDict.setdefault(values[0],[]).append(i)
                return colorDict, labelDict, nameDict, aggregationDict

        def aggregateLabels(self, labels):
                for i, label in enumerate(labels):
                        for key, value in self.aggregationDict.items():
                                if label in value:
                                        labels[i] = key+1
                                        break

        def aggregateDict(self):
                newLabelDict, newNameDict, newAggregationDict = {}, {}, {}
                newLabelDict[0] = [0, 0, 0]
                for i, keep in enumerate(sorted(self.aggregationDict)):
                        newLabelDict[int(i)+1] = self.labelDict[int(keep)+1]
                        newNameDict[int(i)] = self.nameDict[int(keep)]
                        newAggregationDict[int(i)] = self.aggregationDict[keep]
                self.labelDict, self.nameDict, self.aggregationDict = newLabelDict, newNameDict, newAggregationDict