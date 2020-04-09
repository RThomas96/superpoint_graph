# Data augmentation study

## Problem

When you try to apply the method on a relatively small/simple cloud subset, the partitionning algorithm is able to compute a small amought of huge superpoints, which make the labelisation task easier, but the training task much harder.
We need to find a way to increase the number of data from a small amount of clouds.

## Propositions

### Increase the number of superpoints

The most straightforward solution is to increase the number of superpoints.
There is multiple ways of doing this:

* Change the regularization strength of the energy formula used for the superpoint computation
* Use a new cloud oversegmentation algorithm with graph structured deep learning metric 
* Try to get the bigger superpoints possible and then voxelise each of then

An improvement can be to compute other superpoints at the end of the classification, as big as possible. Then the final labelisation is the label in majority in these huge superpoints. Thuse we can eliminate some outliers due to superpoints anormally small size.
