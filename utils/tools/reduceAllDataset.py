# Reduce density of all dataset

import argparse
import pdal # For laz reading
import sys
from pathManager import PathManager

parser = argparse.ArgumentParser(description='Simple tool to reduce density of a dataset')
parser.add_argument('ROOT_PATH', help='name of the folder containing the data directory')
args = parser.parse_args()
pathManager = PathManager(args)

for folder in pathManager.folders:
    for i, fileName in enumerate(pathManager.allDataFileName[folder]):
        print(folder + " | " + str(i) + "/" + str(len(pathManager.allDataFileName[folder])) + " : " + fileName)
        dataType = pathManager.allDataFileType[folder][i]
        dataFolder = pathManager.rootPath + "/data/raw/" + folder + "/" 
        dataFile = dataFolder + fileName + '.' + dataType
        dataOutFile = dataFolder + fileName + '-reduced.' + dataType
        json = """
        [
            "%s",
            {
                "type":"filters.decimation",
                "step": 10
            },
            {
                "type":"writers.las",
                "filename":"%s",
                "forward":"all"
            }
        ]
        """ % (dataFile, dataOutFile)
        
        pipeline = pdal.Pipeline(json)
        count = pipeline.execute()
