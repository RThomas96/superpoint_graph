import argparse
import sys
import os
import shutil
sys.path.append("./partition/")
sys.path.append("./learning/")
from superpointComputation import main as superpointComputation 
from visualize_result import main as visualize
from train import main as train 

parser = argparse.ArgumentParser(description='Superpoint computation programm')
parser.add_argument('--keep', action='store_true', help='Do not delete all files')
args = parser.parse_args()

rootDir = "projects/test2"

if not args.keep:
    # Delete all subfolders except "data"
    subdirs = os.listdir(rootDir)
    print("Delete subfolders: ", end="")
    for directory in subdirs: 
        if directory != "data":
            shutil.rmtree(os.path.join(rootDir,directory))
            print(directory, end=" ")

    print("")

#superpointComputation(["test2", "--voxelize", "--voxel_width", "0.05"])
superpointComputation(["test2", "--voxelize", "--voxel_width", "0.1"])

train(["test2", "--epoch", "10"])

visualize(["test2", "test", "LPA3-1", "--outType", "sptgd"])

train(["test2", "--epoch", "15", "--resume"])
