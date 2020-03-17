# Full pipeline description of Superpoint graph

## File specificities

Related file: partion/superpointComputation.py

Arguments:

ROOT\_PATH'	       : Name of the folder containing the data directory to be processed, must be at root folder\
--knn\_geofeatures     : Number of neighbors for the geometric features computation aka operations on eigen values, default=45\
--knn\_adj             : Number of neighbors for the adjacency structure of the minimal partition between superpoints , default=10\
--lambda\_edge\_weight : TODO Parameter determine the edge weight for minimal part, default=1.\
--reg\_strength        : TODO Regularization strength for the minimal partition, default=0.03\
--d\_se\_max           : Max length of super edges, default=0\
--voxel\_width         : Voxel size when subsampling (in m), default=0.03\
--ver\_batch           : TODO Batch size for reading large files, 0 do disable batch loading, default=0\
--overwrite            : Wether to read existing files or overwrite them

## I. Superpoint graph computation

### Read point cloud file

>Actually only .ply format is accepted, but .las is available in the code

### Reduce point cloud density with voxelisation with voxels of "voxel\_width" size

* Compute min and max xyz, and then each voxel min and max
* For each point check if it is on a voxel, if so mark the voxel as usefull
* For each point add his coordinates, rgb, etc to the right voxel
* For each usefull voxel compute average coordinates, rgb, etc. If present, the final label is the majority one.

> C++ code with openMP\
> Do not work with non aligned cloud

### Two knn\_graphs computation

Computation of two knn\_graph, one with "knn\_geofeatures" neighbors, and one with "knn\_adj".
Next, the number "knn\_geofeatures" will be refered as $knnGeo and "knn\_adj" as $knnadj
TODO This step has an option "Voronoi".

* Computation of $knngeo neighbors relations and distances with sklearn (a python lib)
* Only neighbors relation is returned and then used for geometric features computation. As the form of simple array, the $knnGeo first values are indexes of neighbors with the first point and so on
* The whole graph, relations and distances, is then cut to be only a $knnadj" graph and returned 
* The graph is then built and returned, points are then reordered into "sources", neighbors as "target" and distances as "edges"

> "knn\_adj" MUST be lower than "knn\_geofeatures"\
> BUG: "knn\_geofeatures" cannot be geater than 100

### Geometrics features computation

Related file: ply\_c.cpp, compute\_geof, l. 385

* For each point compute the covariance matrix of neighborhood
* Compute eigen values and vectors
* Compute 4 metrics based on eigen values: 
	* Linearity: <img src="https://render.githubusercontent.com/render/math?math=\frac{\sqrt{\lambda_0} - \sqrt{\lambda_1}}{\sqrt{\lambda_0}}">
	* Planarity: <img src="https://render.githubusercontent.com/render/math?math=\frac{\sqrt{\lambda_1} - \sqrt{\lambda_2}}{\sqrt{\lambda_0}}">
	* Scattering: <img src="https://render.githubusercontent.com/render/math?math=\frac{\sqrt{\lambda_2}}{\sqrt{\lambda_0}}">
	* Verticality: see code l. 441
* Add normalized color to features

### Optimisation problem resolution 

* Add "edge weight" to the graph, which is: ( edgeLength + "--lambda\_edge\_weight" )/ mean(edgeLength) )
* Use CutPursuit to solve the problem TODO

### Superpoint graph computation

* Compute 4 features for each superpoints: 
	* centroid: mean(xyz) 
	* length: sqrt(sum(xyz))
	* surface: TODO
	* volume: TODO
* Compute 8 features for each superedges: 
	* deltaCentroid = centroids.source - centroids.target
	* lengthRatio = length.source / length.target
	* surfaceRatio = surface.source / surface.target
	* volumeRatio = volume.source / volume.target
	* pointCountRatio = pointCount.source / pointCount.target
	* delta = xyzSource - xyzTarget
	* deltaMean = np.mean(delta, axis=0)
	* deltaStd = np.std(delta, axis=0)
	* deltaNorm = np.mean(np.sqrt(np.sum(delta^2, axis=1)))
* Write superpoint graph on a file TODO: archi

## II. Prepare data for learning

Put all normalized values between -0.5 and 0.5 into a file, with one array per superpoint (in the code superpoint = component).
Each array correspond to a superpoint, and contain a set of values for each points in this superpoint.
Values for each points have this format: [[xyz, rgb, e, lpsv, xyzn, distanceToCenter], [...], ...], when each subarray are values of a point.

* xyz: all points coordinates not normalized (usefull to stay variant to object size)
* rgb: normalized points color (rgb = rgb/255.)
* e: elevation, computed with Ransac Regressor to get the ground TODO
* lpsv: geometrics features previously computed to create superpoints, there are normalized
* xyzn: normalized points coordinates
* distanceToCenter: sum of distance of each point with the room center when center is the mean of x,y of all points
