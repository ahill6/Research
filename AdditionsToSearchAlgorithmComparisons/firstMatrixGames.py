#!/usr/bin/env python
'''
CREATED:2011-11-12 08:23:33 by Brian McFee <bmcfee@cs.ucsd.edu>

Spatial tree demo for matrix data
'''

import numpy, random
from spatialtree import spatialtree


# First, create a random data matrix
N = 5000
D = 20

X = numpy.random.randn(N,D)


# Apply a random projection so the data's not totally boring
P = numpy.random.randn(D, D)

X = numpy.dot(X, P)

# Construct a tree.  By default, we get a KD-spill-tree with height
# determined automatically, and spill = 25%

print 'Building tree...'
T = spatialtree(X)
print 'done.'

# Show some useful information about the tree
print '# items in tree    : ', len(T)
print 'Dimensionality     : ', T.getDimension()
print 'Height of tree     : ', T.getHeight()
print 'Spill percentage   : ', T.getSpill()
print 'Split rule         : ', T.getRule()

# If we want to compare accuracy against brute-force search,
# we can make a height=0 tree:
T_root = spatialtree(X, height=0)
    
in_tree_count = 0
in_tree_recall = 0

for countvar in range(100):
    rand = random.randint(0,499)
    knn_a = T.k_nearest(X, k=10, index = rand)
    knn_t = T_root.k_nearest(X, k=10, index=rand)
    in_tree_count += 1
    in_tree_recall += (len(set(knn_a) & set(knn_t)) * 1.0 / len(set(knn_t)))
    
# Find the 10 approximate nearest neighbors of the 500th data point
# returned list is row#'s of X closest to the query index, 
# sorted by increasing distance
#knn_a = T.k_nearest(X, k=10, index=429)
#print 'KNN approx (index) : ', knn_a

# Now, get the true nearest neighbors
#knn_t = T_root.k_nearest(X, k=10, index=429)
#print 'KNN true   (index) : ', knn_t

# Recall rate:
print 'In Tree Recall             : ', in_tree_recall/in_tree_count

# We can also search with a new vector not already in the tree
out_of_tree_count = 0
out_of_tree_recall = 0

for countvar in range(100):
    query = numpy.dot(numpy.random.randn(D), P)
    knn_a = T.k_nearest(X, k=10, vector=query)
    knn_t = T_root.k_nearest(X, k=10, vector=query)
    out_of_tree_count += 1
    out_of_tree_recall += (len(set(knn_a) & set(knn_t)) * 1.0 / len(set(knn_t)))

# Generate a random test query
#query = numpy.dot(numpy.random.randn(D), P)

# Find approximate nearest neighbors
#knn_a = T.k_nearest(X, k=10, vector=query)
#print 'KNN approx (vector): ', knn_a

# And the true neighbors
#knn_t = T_root.k_nearest(X, k=10, vector=query)
#print 'KNN true   (vector): ', knn_t

# Recall rate:
print 'Out of Tree Recall             : ', out_of_tree_recall/out_of_tree_count

