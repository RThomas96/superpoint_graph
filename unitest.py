import sys
import os
import shutil
sys.path.append("./partition/")
sys.path.append("./learning/")
from superpointComputation import main as superpointComputation 
from train import main as train 

rootDir = "projects/test"

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
