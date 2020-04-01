import os 

class ColorLabelManager:
        def __init__(self):
                #self.filePath = os.getcwd() + "/" + os.path.dirname(__file__) + "/colorCode" 
                self.filePath = "/home/thomas/Data/Cajun/Evaluation/Methods/superpoint_graph/utils/colorCode" 
                self.colorDict = self.parseColorFile()
                self.nbColor = len(self.colorDict)

        def parseColorFile(self):
                colorFile = open(self.filePath, "r") 
                colorDict = {}
                for line in colorFile:
                        values = line.split()
                        key = values[1]+values[2]+values[3]
                        colorDict[key] = values[0]
                return colorDict
