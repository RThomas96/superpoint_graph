import os
import numpy as np
import h5py
from plyfile import PlyData, PlyElement # Ply I/O
import csv
import pandas as pd


#------------------------------------------------------------------------------
def reduceDensity(inFile, outFile, voxel_width, regular_density = True):

    filename, extension = os.path.splitext(outFile)
    if extension == ".laz":
        extension = ".las"# Las and laz writer are the same
    filter=""
    if regular_density:
        filter= """
        {
            "type":"filters.voxelcenternearestneighbor",
            "cell":"%s"
        },
        """ % (voxel_width)
    else:
        filter= """
        {
            "type":"filters.voxelcentroidnearestneighbor",
            "cell":"%s"
        },
        """ % (voxel_width)

    json = """
    [
        "%s",
    """ % (inFile)
    
    json += filter
    
    if extension == ".las":
        json += """
            {
                "type":"writers%s",
                "filename":"%s",
                "forward":"all"
            }
        ]
        """ % (extension, outFile)
    else:
        json += """
            {
                "type":"writers%s",
                "filename":"%s",
                "precision" : "7"
            }
        ]
        """ % (extension, outFile)
    import pdal # Laz I/O
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    
    #"dims": ["x=float64", "y=float64", "z=float64", "Red=uint16_t", "Green=uint16_t", "Blue=uint16_t", "classification=uint16_t"],

#------------------------------------------------------------------------------
def read_file(filename, extension):
    if extension == "laz":
        return read_laz(filename)
    else:
        return read_ply(filename)

#------------------------------------------------------------------------------
def read_ply(filename):
    """convert from a ply file. include the label and the object number"""
    #---read the ply file--------
    plydata = PlyData.read(filename)
    xyz = np.stack([plydata['vertex'][n] for n in['x', 'y', 'z']], axis=1).astype(np.float32)
    try:
        rgb = np.stack([plydata['vertex'][n]
                        for n in ['red', 'green', 'blue']]
                       , axis=1).astype(np.uint8)
    except ValueError:
        rgb = np.stack([plydata['vertex'][n]
                        for n in ['r', 'g', 'b']]
                       , axis=1).astype(np.float32)
    if np.max(rgb) > 1:
        rgb = rgb
    try:
        labels = plydata['vertex']['classification'].astype(np.uint32)
        return xyz, rgb, labels, []
    except ValueError:
        try:
            labels = plydata['vertex']['label']
            return xyz, rgb, labels, []
        except ValueError:
            return xyz, rgb, [], []

#------------------------------------------------------------------------------
def read_laz(filename):

    # The filter for PDAL, in json format
    json = """
    [
        {
            "type":"readers.las",
            "filename":"%s"
        }
    ]
    """ % (filename)

    import pdal # Laz I/O
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    arrays = np.array(pipeline.arrays)

    x = np.reshape(arrays['X'], (count,1))
    y = np.reshape(arrays['Y'], (count,1))
    z = np.reshape(arrays['Z'], (count,1))

    r = np.reshape(arrays['Red'], (count,1))
    g = np.reshape(arrays['Green'], (count,1))
    b = np.reshape(arrays['Blue'], (count,1))

    #labels = np.reshape(arrays['Classification'], (count,1))
    labels = np.array(arrays['Classification']).flatten()

    xyz = np.hstack((x,y,z)).astype('f4')
    rgb = np.hstack((r/255,g/255,b/255)).astype('u1')
    #rgb = rgb/255
    return xyz, rgb, labels, [] 
#------------------------------------------------------------------------------
def read_las(filename):
    """convert from a las file with no rgb"""
    #---read the ply file--------
    try:
        inFile = laspy.file.File(filename, mode='r')
    except NameError:
        raise ValueError("laspy package not found. uncomment import in /partition/provider and make sure it is installed in your environment")
    N_points = len(inFile)
    x = np.reshape(inFile.x, (N_points,1))
    y = np.reshape(inFile.y, (N_points,1))
    z = np.reshape(inFile.z, (N_points,1))
    xyz = np.hstack((x,y,z)).astype('f4')
    return xyz

#------------------------------------------------------------------------------
#def write_file(filename, xyz, rgb, labels, extension):
def write_file(*args):
    filename = args[0]
    xyz = args[1]
    rgb = args[2]
    if len(args) == 4 and isinstance(args[3], str):
        extension = args[3]
        if extension == "laz":
            write_laz_simple(filename, xyz, rgb)
        else:
            write_ply_simple(filename, xyz, rgb)
    elif len(args) == 5 and isinstance(args[4], str):
        labels = args[3]
        extension = args[4]
        if extension == "laz":
            write_laz_labels(filename, xyz, rgb, labels)
        else:
            write_ply_labels(filename, xyz, rgb, labels)

#------------------------------------------------------------------------------
def write_ply_labels(filename, xyz, rgb, labels):
    """write into a ply file. include the label"""
    """ Label type is f4 cause u1 cannot write -1 value, at it is used for unknown label """
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1')
            , ('blue', 'u1'), ('label', 'f4')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop+3][0]] = rgb[:, i_prop]
    vertex_all[prop[6][0]] = labels
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)

#------------------------------------------------------------------------------
def write_ply_simple(filename, xyz, rgb):
    """write into a ply file"""
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop+3][0]] = rgb[:, i_prop]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)

#------------------------------------------------------------------------------
def write_laz_labels(filename, xyz, rgb, labels):
    """write into a laz file"""
    prop = [('X', 'f4'), ('Y', 'f4'), ('Z', 'f4'), ('Red', 'u1'), ('Green', 'u1')
            , ('Blue', 'u1'), ('Classification', 'f4')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop+3][0]] = rgb[:, i_prop]
    vertex_all[prop[6][0]] = labels

    # Create empty dir if needed
    os.makedirs(os.path.split(filename)[0], exist_ok=True)

    # Write our data to an LAZ file
    output =u"""{
      "pipeline":[
        {
          "type":"writers.las",
          "filename":"%s",
          "offset_x":"auto",
          "offset_y":"auto",
          "offset_z":"auto",
          "scale_x":0.00000001,
          "scale_y":0.00000001,
          "scale_z":0.00000001
        }
      ]
    }""" % (filename)
    
    import pdal # Laz I/O
    p = pdal.Pipeline(output, [vertex_all])
    count = p.execute()

#------------------------------------------------------------------------------
def write_laz_simple(filename, xyz, rgb):
    """write into a laz file"""
    prop = [('X', '<f4'), ('Y', '<f4'), ('Z', '<f4'), ('Red', '<u1'), ('Green', '<u1')
            , ('Blue', '<u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop+3][0]] = rgb[:, i_prop]

    # Create folders
    os.makedirs(os.path.split(filename)[0], exist_ok=True)

    # Write our data to an LAZ file
    output =u"""{
      "pipeline":[
        {
          "type":"writers.las",
          "filename":"%s",
          "offset_x":"auto",
          "offset_y":"auto",
          "offset_z":"auto",
          "scale_x":0.00000001,
          "scale_y":0.00000001,
          "scale_z":0.00000001
        }
      ]
    }""" % (filename)
    
    import pdal # Laz I/O
    p = pdal.Pipeline(output, [vertex_all])
    count = p.execute()

#------------------------------------------------------------------------------
def write_features(file_name, geof, xyz, rgb, graph_nn, labels):
    """write the geometric features, labels and clouds in a h5 file"""
    if os.path.isfile(file_name):
        os.remove(file_name)
    data_file = h5py.File(file_name, 'w')
    data_file.create_dataset('geof', data=geof, dtype='float32', compression="gzip", compression_opts=7)
    data_file.create_dataset('source', data=graph_nn["source"], dtype='uint32', compression="gzip", compression_opts=7)
    data_file.create_dataset('target', data=graph_nn["target"], dtype='uint32', compression="gzip", compression_opts=7)
    data_file.create_dataset('distances', data=graph_nn["distances"], dtype='float32', compression="gzip", compression_opts=7)
    data_file.create_dataset('xyz', data=xyz, dtype='float32', compression="gzip", compression_opts=7)
    if len(rgb) > 0:
        data_file.create_dataset('rgb', data=rgb, dtype='uint8', compression="gzip", compression_opts=7)

    #if len(labels) > 0:
    #    if len(labels) > 0 and len(labels.shape)>1 and labels.shape[1]>1:
    #        data_file.create_dataset('labels', data=labels, dtype='uint32')
    #    else:
    #        data_file.create_dataset('labels', data=labels, dtype='uint8')
    if len(labels) > 0 and len(labels.shape)>1 and labels.shape[1]>1:
        data_file.create_dataset('labels', data=labels, dtype='uint32', compression="gzip", compression_opts=7)
    else:
        data_file.create_dataset('labels', data=labels, dtype='uint8', compression="gzip", compression_opts=7)
    data_file.close()
#------------------------------------------------------------------------------
def read_features(file_name):
    """read the geometric features, clouds and labels from a h5 file"""
    data_file = h5py.File(file_name, 'r')
    #fist get the number of vertices
    n_ver = len(data_file["geof"][:, 0])
    has_labels = len(data_file["labels"])
    #the labels can be empty in the case of a test set
    if has_labels:
        labels = np.array(data_file["labels"])
    else:
        labels = []
    #---fill the arrays---
    geof = data_file["geof"][:]
    xyz = data_file["xyz"][:]
    rgb = data_file["rgb"][:]
    source = data_file["source"][:]
    target = data_file["target"][:]
    try:
        distances = data_file["distances"][:]
    except KeyError:
        distances = []

    #---set the graph---
    graph_nn = dict([("is_nn", True)])
    graph_nn["source"] = source
    graph_nn["target"] = target
    graph_nn["distances"] = distances
    return geof, xyz, rgb, graph_nn, labels
#------------------------------------------------------------------------------
def write_spg(file_name, graph_sp, components, in_component):
    """save the partition and spg information"""
    if os.path.isfile(file_name):
        os.remove(file_name)
    data_file = h5py.File(file_name, 'w')
    grp = data_file.create_group('components')
    n_com = len(components)
    for i_com in range(0, n_com):
        grp.create_dataset(str(i_com), data=components[i_com], dtype='uint32')
        #grp.create_dataset(str(i_com), data=components[i_com].astype('uint32'), dtype='uint32')
    data_file.create_dataset('in_component'
                             , data=in_component, dtype='uint32')
    data_file.create_dataset('sp_labels'
                             , data=graph_sp["sp_labels"], dtype='uint32')
    data_file.create_dataset('sp_centroids'
                             , data=graph_sp["sp_centroids"], dtype='float32')
    data_file.create_dataset('sp_length'
                             , data=graph_sp["sp_length"], dtype='float32')
    data_file.create_dataset('sp_surface'
                             , data=graph_sp["sp_surface"], dtype='float32')
    data_file.create_dataset('sp_volume'
                             , data=graph_sp["sp_volume"], dtype='float32')
    data_file.create_dataset('sp_point_count'
                             , data=graph_sp["sp_point_count"], dtype='uint64')
    data_file.create_dataset('source'
                             , data=graph_sp["source"], dtype='uint32')
    data_file.create_dataset('target'
                             , data=graph_sp["target"], dtype='uint32')
    data_file.create_dataset('se_delta_mean'
                             , data=graph_sp["se_delta_mean"], dtype='float32')
    data_file.create_dataset('se_delta_std'
                             , data=graph_sp["se_delta_std"], dtype='float32')
    data_file.create_dataset('se_delta_norm'
                             , data=graph_sp["se_delta_norm"], dtype='float32')
    data_file.create_dataset('se_delta_centroid'
                             , data=graph_sp["se_delta_centroid"], dtype='float32')
    data_file.create_dataset('se_length_ratio'
                             , data=graph_sp["se_length_ratio"], dtype='float32')
    data_file.create_dataset('se_surface_ratio'
                             , data=graph_sp["se_surface_ratio"], dtype='float32')
    data_file.create_dataset('se_volume_ratio'
                             , data=graph_sp["se_volume_ratio"], dtype='float32')
    data_file.create_dataset('se_point_count_ratio'
                             , data=graph_sp["se_point_count_ratio"], dtype='float32')
#-----------------------------------------------------------------------------
def read_spg(file_name):
    """read the partition and spg information"""
    data_file = h5py.File(file_name, 'r')
    graph = dict([("is_nn", False)])
    graph["source"] = np.array(data_file["source"], dtype='uint32')
    graph["target"] = np.array(data_file["target"], dtype='uint32')
    graph["sp_centroids"] = np.array(data_file["sp_centroids"], dtype='float32')
    graph["sp_length"] = np.array(data_file["sp_length"], dtype='float32')
    graph["sp_surface"] = np.array(data_file["sp_surface"], dtype='float32')
    graph["sp_volume"] = np.array(data_file["sp_volume"], dtype='float32')
    graph["sp_point_count"] = np.array(data_file["sp_point_count"], dtype='uint64')
    graph["se_delta_mean"] = np.array(data_file["se_delta_mean"], dtype='float32')
    graph["se_delta_std"] = np.array(data_file["se_delta_std"], dtype='float32')
    graph["se_delta_norm"] = np.array(data_file["se_delta_norm"], dtype='float32')
    graph["se_delta_centroid"] = np.array(data_file["se_delta_centroid"], dtype='float32')
    graph["se_length_ratio"] = np.array(data_file["se_length_ratio"], dtype='float32')
    graph["se_surface_ratio"] = np.array(data_file["se_surface_ratio"], dtype='float32')
    graph["se_volume_ratio"] = np.array(data_file["se_volume_ratio"], dtype='float32')
    graph["se_point_count_ratio"] = np.array(data_file["se_point_count_ratio"], dtype='float32')
    in_component = np.array(data_file["in_component"], dtype='uint32')
    n_com = len(graph["sp_length"])
    graph["sp_labels"] = np.array(data_file["sp_labels"], dtype='uint32')
    grp = data_file['components']
    components = np.empty((n_com,), dtype=object)
    for i_com in range(0, n_com):
        components[i_com] = np.array(grp[str(i_com)], dtype='uint32').tolist()
    return graph, components, in_component

#-----------------------------------------------------------------------------
def writeCsv(file, header, data):
    isFile = os.path.isfile(file)
    with open(file, 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if not isFile:
            spamwriter.writerow(header)
        spamwriter.writerow(data)

#-----------------------------------------------------------------------------
def duplicateLastLineCsv(file, epoch = -1):
    data = [] 
    with open(file, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in csvreader:
            data.append(row)

    data = data[-1]
    if epoch == -1:
        data[0] = str(float(data[0]) + 1) # Increment epoch
    else:
        data[0] = epoch
    writeCsv(file, [], data)
