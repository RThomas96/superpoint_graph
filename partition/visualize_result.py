"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
    
this functions outputs varied ply file to visualize the different steps
"""
import os.path
import numpy as np
import argparse
import sys
sys.path.append("./partition/")
from plyfile import PlyData, PlyElement
import provider

def openPredictions(res_file, h5FolderPath):
    try:
        pred_red  = np.array(provider.h5py.File(res_file, 'r').get(h5FolderPath))        
        if (len(pred_red) != len(components)):
            raise ValueError("It looks like the spg is not adapted to the result file") 
        return provider.reduced_labels2full(pred_red, components, len(xyz))
    except OSError:
        raise ValueError("%s does not exist in %s" % (h5FolderPath, res_file))

parser = argparse.ArgumentParser(description='Generate ply file from prediction file')
parser.add_argument('ROOT_PATH', help='Folder name which contains data')
parser.add_argument('predFileName', help='Name of the prediction file used, they are store at /results/ folder')
parser.add_argument('file_path', help='Full path of file to display, from data folder, must be "test/X"')
parser.add_argument('-ow', '--overwrite', action='store_true', help='Wether to read existing files or overwrite them')
parser.add_argument('--outType', default='p', help='which cloud to output: s = superpoints, p = predictions')
args = parser.parse_args()

outSuperpoints = 's' in args.outType
outPredictions = 'p' in args.outType
#---path to data---------------------------------------------------------------
root = os.path.dirname(os.path.realpath(__file__)) + '/../' + args.ROOT_PATH

folder = os.path.split(args.file_path)[0] + '/'
file_name = os.path.split(args.file_path)[1]
h5FolderPath = folder + file_name

fea_file   = root + "/features/"          + folder + file_name + '.h5'
spg_file   = root + "/superpoint_graphs/" + folder + file_name + '.h5'
res_file   = root + "/results/" + args.predFileName + '.h5'
outPredFileName = file_name + "_pred.ply"
outPredFile   = root + "/visualisation/predictions/" + args.predFileName + "/" + outPredFileName 

outSPntFileName = file_name + "_partition.ply"
outSPntFile   = root + "/visualisation/superpoints/" + outSPntFileName 


if not os.path.isfile(fea_file) :
    raise ValueError("%s does not exist and is needed" % fea_file)
if not os.path.isfile(spg_file):    
    raise ValueError("%s does not exist and is needed to output the partition  or result ply" % spg_file) 
if not os.path.isfile(res_file):
    raise ValueError("%s does not exist and is needed to output the result ply" % res_file) 

geof, xyz, rgb, graph_nn, labels = provider.read_features(fea_file)
graph_spg, components, in_component = provider.read_spg(spg_file)

pred_full = openPredictions(res_file, folder + file_name)

#if not bool(args.upsample):
def checkIfExist(file, fileName):
    if os.path.isfile(file) and not args.overwrite:
        print("{} result file already exist and overwrite option isn't set".format(fileName))
        print("Nothing to do")
        return False
    elif os.path.isfile(file):
        print("{} result file already exist and will be OVERWRITE".format(fileName))
    else :
        print("writing the file {}...".format(fileName))
    return True

if outPredictions:
    n_labels = 10    
    if checkIfExist(outPredFile, outPredFileName):
        provider.prediction2ply(outPredFile, xyz, pred_full+1, n_labels, "custom_dataset")

if outSuperpoints:
    if checkIfExist(outSPntFile, outSPntFileName):
        provider.partition2ply(outSPntFile, xyz, components)

#if bool(args.upsample):
#    if args.dataset=='s3dis':
#        data_file   = root + 'data/' + folder + file_name + '/' + file_name + ".txt"
#        xyz_up, rgb_up = read_s3dis_format(data_file, False)
#    elif args.dataset=='sema3d':#really not recommended unless you are very confident in your hardware
#        data_file  = data_folder + file_name + ".txt"
#        xyz_up, rgb_up = read_semantic3d_format(data_file, 0, '', 0, args.ver_batch)
#    elif args.dataset=='custom_dataset':
#        data_file  = data_folder + file_name + ".ply"
#        xyz_up, rgb_up = read_ply(data_file)
#    del rgb_up
#    pred_up = interpolate_labels(xyz_up, xyz, pred_full, args.ver_batch)
#    print("writing the upsampled prediction file...")
#    prediction2ply(ply_file + "_pred_up.ply", xyz_up, pred_up+1, n_labels, args.dataset)