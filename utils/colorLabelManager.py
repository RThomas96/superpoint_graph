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
                for i, line in enumerate(colorFile):
                        values = line.split()
                        key = values[1]+values[2]+values[3]
                        colorDict[key] = values[0]
                        # +1 here cause first label is "unknown" label 
                        labelDict[i+1] = [values[1], values[2], values[3]]
                        nameDict[i] = values[7]
                        aggregationDict.setdefault(values[0],[]).append(i)
                return colorDict, labelDict, nameDict, aggregationDict

        def aggregateLabels(self, labels):
                for label in labels:
                        for key, value in self.aggregationDict.items():
                                if label in value:
                                        label = key
                                        break

        def aggregateDict(self):
                newLabelDict, newNameDict, newAggregationDict = {}, {}, {}
                newLabelDict[0] = [0, 0, 0]
                for i, keep in enumerate(sorted(self.aggregationDict)):
                        newLabelDict[int(i)+1] = self.labelDict[int(keep)+1]
                        newNameDict[int(i)] = self.nameDict[int(keep)]
                        newAggregationDict[int(i)] = self.aggregationDict[keep]
                self.labelDict, self.nameDict, self.aggregationDict = newLabelDict, newNameDict, newAggregationDict