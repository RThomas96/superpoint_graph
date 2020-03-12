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
parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
parser.add_argument('--ROOT_PATH', default='/mnt/bigdrive/loic/S3DIS', help='folder containing the ./data folder')
parser.add_argument('--file_path', default='Area_1/conferenceRoom_1', help='file to output (must include the area / set in its path)')
parser.add_argument('--file_res', default='Area_1/conferenceRoom_1', help='file to output (must include the area / set in its path)')
args = parser.parse_args()
#---path to data---------------------------------------------------------------
#root of the data directory
root = args.ROOT_PATH+'/'
folder = os.path.split(args.file_res)[0] + '/'
file_name = os.path.split(args.file_res)[1]
res_file_name = file_name

folder = os.path.split(args.file_path)[0] + '/'
file_name = os.path.split(args.file_path)[1]

n_labels = 10    
#---load the values------------------------------------------------------------
fea_file   = root + "features/"          + folder + file_name + '.h5'
spg_file   = root + "superpoint_graphs/" + folder + file_name + '.h5'
res_file   = root + "results/" + folder + res_file_name + '.h5'
#ply_folder = root + "clouds/"            + folder 
ply_folder = root + "visualisation/result"              + folder 
ply_file   = ply_folder                  + file_name

if not os.path.isdir(root + "clouds/"):
    os.mkdir(root + "clouds/" )
if not os.path.isdir(ply_folder ):
    os.mkdir(ply_folder)
if (not os.path.isfile(fea_file)) :
    raise ValueError("%s does not exist and is needed" % fea_file)

geof, xyz, rgb, graph_nn, labels = provider.read_features(fea_file)

if not os.path.isfile(spg_file):    
    raise ValueError("%s does not exist and is needed to output the partition  or result ply" % spg_file) 
else:
    graph_spg, components, in_component = provider.read_spg(spg_file)

if not os.path.isfile(res_file):
    raise ValueError("%s does not exist and is needed to output the result ply" % res_file) 
try:
    pred_red  = np.array(provider.h5py.File(res_file, 'r').get(folder + file_name))        
    if (len(pred_red) != len(components)):
        raise ValueError("It looks like the spg is not adapted to the result file") 
    pred_full = provider.reduced_labels2full(pred_red, components, len(xyz))
except OSError:
    raise ValueError("%s does not exist in %s" % (folder + file_name, res_file))

#if not bool(args.upsample):
print("writing the prediction file...")
provider.prediction2ply(ply_file + "_pred.ply", xyz, pred_full+1, n_labels, "custom_dataset")

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