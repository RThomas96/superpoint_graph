import os

class ColorLabelManager:
        def __init__(self):
                #TODO
                self.filePath = "/home/thomas/Data/Cajun/Data/Evaluation/Methods/superpoint_graph/utils/colorCode" 
                self.label2Color, self.label2Name, self.aggregationDict = self.parseColorFile()
                self.nbColor = len(self.aggregationDict)
                self.needAggregation = True if max([len(x) for x in self.aggregationDict.values()]) > 1 else False
                if self.needAggregation:
                        self.aggregateDict()

        def parseColorFile(self):
                colorFile = open(self.filePath, "r") 
                label2Color, label2Name, aggregationDict = {}, {}, {}

                label2Color[0] = [0, 0, 0]
                label2Name[0] = 'Inconnue'
                for i, line in enumerate(colorFile):
                        values = line.split()
                        label2Color[i+1] = [values[1], values[2], values[3]]
                        label2Name[i+1] = values[7]
                        aggregationDict.setdefault(values[0],[]).append(i)
                return label2Color, label2Name, aggregationDict

        def aggregateLabels(self, labels):
            return [self.aggregationDict[i] for i in labels]

        def aggregateDict(self):
                newLabel2Color, newLabel2Name = {}, {}
                newLabel2Color[0] = [0, 0, 0]
                newLabel2Name[0] = 'Inconnue'

                keys = list(self.aggregationDict.keys())
                intKeys = [int(x) for x in keys]
                sortedKeys = sorted(intKeys)

                for i, keep in enumerate(sortedKeys):
                        newLabel2Color[int(i)+1] = self.label2Color[keep]
                        newLabel2Name[int(i)+1] = self.label2Name[keep]
                        #newAggregationDict[int(i)] = self.aggregationDict[str(keep)]

                # Here we invert the aggregation dict to simplify label conversion
                reverse_dict = {0:0}
                for key in self.aggregationDict.keys():
                    for value in self.aggregationDict[key]:
                        reverse_dict[value+1] = int(key)
                
                self.label2Color, self.label2Name, self.aggregationDict = newLabel2Color, newLabel2Name, reverse_dict
