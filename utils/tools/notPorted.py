
# Not working yet ############################################################
#------------------------------------------------------------------------------
def error2ply(filename, xyz, rgb, labels, prediction):
    """write a ply with green hue for correct classifcation and red for error"""
    if len(prediction.shape) > 1 and prediction.shape[1] > 1:
        prediction = np.argmax(prediction, axis = 1)
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis = 1)
    color_rgb = rgb/255
    for i_ver in range(0, len(labels)):
        
        color_hsv = list(colorsys.rgb_to_hsv(color_rgb[i_ver,0], color_rgb[i_ver,1], color_rgb[i_ver,2]))
        if (labels[i_ver] == prediction[i_ver]) or (labels[i_ver]==0):
            color_hsv[0] = 0.333333
        else:
            color_hsv[0] = 0
        color_hsv[1] = min(1, color_hsv[1] + 0.3)
        color_hsv[2] = min(1, color_hsv[2] + 0.1)
        color_rgb[i_ver,:] = list(colorsys.hsv_to_rgb(color_hsv[0], color_hsv[1], color_hsv[2]))
    color_rgb = np.array(color_rgb*255, dtype='u1')
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i+3][0]] = color_rgb[:, i]        
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
def spg2ply(filename, spg_graph):
    """write a ply displaying the SPG by adding edges between its centroid"""
    vertex_prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_val = np.empty((spg_graph['sp_centroids']).shape[0], dtype=vertex_prop)
    for i in range(0, 3):
        vertex_val[vertex_prop[i][0]] = spg_graph['sp_centroids'][:, i]
    edges_prop = [('vertex1', 'int32'), ('vertex2', 'int32')]
    edges_val = np.empty((spg_graph['source']).shape[0], dtype=edges_prop)
    edges_val[edges_prop[0][0]] = spg_graph['source'].flatten()
    edges_val[edges_prop[1][0]] = spg_graph['target'].flatten()
    ply = PlyData([PlyElement.describe(vertex_val, 'vertex'), PlyElement.describe(edges_val, 'edge')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
def scalar2ply(filename, xyz, scalar):
    """write a ply with an unisgned integer scalar field"""
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('scalar', 'f4')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    vertex_all[prop[3][0]] = scalar
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    
    ply.write(filename)
#------------------------------------------------------------------------------
def write_ply_obj(filename, xyz, rgb, labels, object_indices):
    """write into a ply file. include the label and the object number"""
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1')
            , ('green', 'u1'), ('blue', 'u1'), ('label', 'u1')
            , ('object_index', 'uint32')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop+3][0]] = rgb[:, i_prop]
    vertex_all[prop[6][0]] = labels
    vertex_all[prop[7][0]] = object_indices
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
    
#------------------------------------------------------------------------------
def embedding2ply(filename, xyz, embeddings):
    """write a ply with colors corresponding to geometric features"""
    
    if embeddings.shape[1]>3:
        pca = PCA(n_components=3)
        #pca.fit(np.eye(embeddings.shape[1]))
        pca.fit(np.vstack((np.zeros((embeddings.shape[1],)),np.eye(embeddings.shape[1]))))
        embeddings = pca.transform(embeddings)
        
    #value = (embeddings-embeddings.mean(axis=0))/(2*embeddings.std())+0.5
    #value = np.minimum(np.maximum(value,0),1)
    #value = (embeddings)/(3 * embeddings.std())+0.5
    value = np.minimum(np.maximum((embeddings+1)/2,0),1)
    
    
    color = np.array(255 * value, dtype='uint8')
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i+3][0]] = color[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    
    ply.write(filename)

#------------------------------------------------------------------------------
def edge_class2ply2(filename, edg_class, xyz, edg_source, edg_target):
    """write a ply with edge weight color coded into the midway point"""
    
    n_edg = len(edg_target)
    
    midpoint = (xyz[edg_source,]+xyz[edg_target,])/2
    
    color = np.zeros((edg_source.shape[0],3), dtype = 'uint8')
    color[edg_class==0,] = [0,0,0]
    color[(edg_class==1).nonzero(),] = [255,0,0]
    color[(edg_class==2).nonzero(),] = [125,255,0]
    color[(edg_class==3).nonzero(),] = [0,125,255]
    
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(n_edg, dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = np.hstack(midpoint[:, i])
    for i in range(3, 6):
        vertex_all[prop[i][0]] = color[:,i-3]
    
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    
    ply.write(filename)
    
#------------------------------------------------------------------------------
def interpolate_labels_batch(data_file, xyz, labels, ver_batch):
    """interpolate the labels of the pruned cloud to the full cloud"""
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis = 1)
    i_rows = None
    labels_f = np.zeros((0, ), dtype='uint8')
    #---the clouds can potentially be too big to parse directly---
    #---they are cut in batches in the order they are stored---
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(xyz)
    while True:
        try:
            if ver_batch>0:
                if i_rows is None:
                    print("read lines %d to %d" % (0, ver_batch))
                else:
                    print("read lines %d to %d" % (i_rows, i_rows + ver_batch))
                #vertices = np.genfromtxt(data_file
                #         , delimiter=' ', max_rows=ver_batch
                #        , skip_header=i_rows)
                vertices = pd.read_csv(data_file
                         , sep=' ', nrows=ver_batch
                         , header=i_rows).values
            else:
                #vertices = np.genfromtxt(data_file
                 #        , delimiter=' ')
                vertices = pd.read_csv(data_file
                         , delimiter=' ').values
                break
        except (StopIteration, pd.errors.ParserError):
            #end of file
            break
        if len(vertices)==0:
            break
        xyz_full = np.array(vertices[:, 0:3], dtype='float32')
        del vertices
        distances, neighbor = nn.kneighbors(xyz_full)
        del distances
        labels_f = np.hstack((labels_f, labels[neighbor].flatten()))
        if i_rows is None:
            i_rows = ver_batch
        else:
            i_rows = i_rows + ver_batch
    return labels_f
#------------------------------------------------------------------------------
def interpolate_labels(xyz_up, xyz, labels, ver_batch):
    """interpolate the labels of the pruned cloud to the full cloud"""
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis = 1)
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(xyz)
    distances, neighbor = nn.kneighbors(xyz_up)
    return labels[neighbor].flatten()
#------------------------------------------------------------------------------
def perfect_prediction(components, labels):
    """assign each superpoint with the majority label"""
    full_pred = np.zeros((labels.shape[0],),dtype='uint32')
    for i_com in range(len(components)):
        # Here we have labels has a 2D array cause each point is composed of multiple points with differente labels
        label_com = labels[components[i_com],1:].sum(0).argmax()
        full_pred[components[i_com]]=label_com
    return full_pred
#----------------------------------------------------
#SEAL utilities

def compute_gt_connected_components(n_ver, edg_source, edg_target, is_transition, cutoff):
    components, in_component = libcp.connected_comp(n_ver,
                                                  edg_source.astype('uint32'),
                                                  edg_target.astype('uint32'),
                                                  is_transition.astype('uint8'), 40) #rough guess
    return components, in_component
#----------------------
def write_gt_connected_components(file_name, components, in_component):
    """save the label-based connected components of the ground truth"""
    if os.path.isfile(file_name):
        os.remove(file_name)
    data_file = h5py.File(file_name, 'w')
    grp = data_file.create_group('components')
    for i_com in range(len(components)):
        grp.create_dataset(str(i_com), data=components[i_com], dtype='uint32')
    data_file.create_dataset('in_component', data=in_component, dtype='uint32')
#-------------------------------------

def read_gt_connected_components(file_name):
    """read the label-based connected components of the ground truth"""
    data_file = h5py.File(file_name, 'r')
    in_component = np.array(data_file["in_component"], dtype='uint32')
    n_com = np.amax(in_component)
    components = np.empty((n_com,), dtype=object)
    for i_com in range(0, n_com):
        components[i_com] = np.array(grp[str(i_com)], dtype='uint32').tolist()
    return components, in_component
