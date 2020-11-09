import argparse
import sys
import os
import shutil
sys.path.append("./partition/")
sys.path.append("./learning/")
from superpointComputation import main as superpointComputation 
from train import main as train 

parser = argparse.ArgumentParser(description='Superpoint computation programm')
parser.add_argument('--keep', action='store_true', help='Do not delete all files')
args = parser.parse_args()

rootDir = "projects/test"

if not args.keep:
    # Delete all subfolders except "data"
    subdirs = os.listdir(rootDir)
    print("Delete subfolders: ", end="")
    for directory in subdirs: 
        if directory != "data":
            shutil.rmtree(os.path.join(rootDir,directory))
            print(directory, end=" ")
    print("")

superpointComputation(["test"])

train(["test", "--epoch", "5"])
