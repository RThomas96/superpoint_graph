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
parser.add_argument('--output_type', default='p', help='which cloud to output: i = input rgb pointcloud \
                    , f = geometric features, p = partition\
                    , e = error, s = SPG')
args = parser.parse_args()
#---path to data---------------------------------------------------------------
#root of the data directory
root = args.ROOT_PATH+'/'
rgb_out = 'i' in args.output_type
fea_out = 'f' in args.output_type
par_out = 'p' in args.output_type
err_out = 'e' in args.output_type
spg_out = 's' in args.output_type
folder = os.path.split(args.file_path)[0] + '/'
file_name = os.path.split(args.file_path)[1]

n_labels = 10    
#---load the values------------------------------------------------------------
fea_file   = root + "features/"          + folder + file_name + '.h5'
spg_file   = root + "superpoint_graphs/" + folder + file_name + '.h5'
#ply_folder = root + "clouds/"            + folder 
ply_folder = root + "data/"              + folder 
ply_file   = ply_folder                  + file_name

if not os.path.isdir(root + "clouds/"):
    os.mkdir(root + "clouds/" )
if not os.path.isdir(ply_folder ):
    os.mkdir(ply_folder)
if (not os.path.isfile(fea_file)) :
    raise ValueError("%s does not exist and is needed" % fea_file)

geof, xyz, rgb, graph_nn, labels = provider.read_features(fea_file)

graph_spg, components, in_component = provider.read_spg(spg_file)

#---write the output clouds----------------------------------------------------
if rgb_out:
    print("writing the RGB file...")
    provider.write_ply(ply_file + "_rgb.ply", xyz, rgb)
    
if fea_out:
    print("writing the features file...")
    provider.geof2ply(ply_file + "_geof.ply", xyz, geof)
    
if par_out:
    print("writing the partition file...")
    provider.partition2ply(ply_file + "_partition.ply", xyz, components)
    
if spg_out:
    print("writing the SPG file...")
    provider.spg2ply(ply_file + "_spg.ply", graph_spg)