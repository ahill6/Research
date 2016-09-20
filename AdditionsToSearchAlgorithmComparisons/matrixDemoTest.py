#!/usr/bin/env python
'''
CREATED:2011-11-12 08:23:33 by Brian McFee <bmcfee@cs.ucsd.edu>
MODIFIED: starting 2016-09-08 by Andrew Hill <ahill6@ncsu.edu>
Spatial tree demo for matrix data
'''
import numpy, random
from spatialtree import spatialtree

def matrixDemoTestWorker(size=None, dimensions=None, tree_type=None, spill_rate=None, samp=None):
    this_run = dict()
    
    # Create random matrix
    N = size or 5000
    D = dimensions or 20 
    tree= tree_type or 'kd'
    spill = spill_rate or .25
    #samples = samp or 100
    samples = 100
    X = numpy.random.randn(N,D)
    
    #Random projection to liven up the data
    P = numpy.random.randn(D, D)
    X = numpy.dot(X, P)
    
    # Construct the type of tree specified with spill specified.
    # Defaults are KD-spill-tree with spill = 25%
    print "Building tree...", tree, spill
    T = spatialtree(X, rule=tree, spill=spill)
    print "Done"
    
    T_root = spatialtree(X, rule=tree, spill=spill, height = 0)
    
    # Test recall on items in tree
    in_tree_count = 0
    in_tree_recall = 0
    
    for countvar in range(samples):
        rand = random.randint(0,499)
        knn_a = T.k_nearest(X, k=10, index = rand)
        knn_t = T_root.k_nearest(X, k=10, index=rand)
        in_tree_count += 1
        in_tree_recall += (len(set(knn_a) & set(knn_t)) * 1.0 / len(set(knn_t)))
        
    index = 'in_' + tree + '_' + str(spill) + '_' + str(N) + '_' + str(D)
    this_run[index] = in_tree_recall/in_tree_count 
        
        # We can also search with a new vector not already in the tree
    out_of_tree_count = 0
    out_of_tree_recall = 0
        
    for countvar in range(samples):
        query = numpy.dot(numpy.random.randn(D), P)
        knn_a = T.k_nearest(X, k=10, vector=query)
        knn_t = T_root.k_nearest(X, k=10, vector=query)
        out_of_tree_count += 1
        out_of_tree_recall += (len(set(knn_a) & set(knn_t)) * 1.0 / len(set(knn_t)))
        
    index = 'out_' + tree + '_' + str(spill) + '_' + str(N) + '_' + str(D)
    this_run[index] = out_of_tree_recall/out_of_tree_count 
    
    return this_run

def matrixTestMaster(samples, trials):
#def matrixTestMaster(samples, trials, all='Y'):
    #call the method with different tree types, spill levels, et al  many times and average/collate the data 
    results = dict()
    #trees = ['kd', 'pca', '2-means', 'rp', 'sway']
    #spill_rates = [0.00, 0.01, 0.05, 0.10, 0.15, 0.2, 0.25]
    #trees = ['kd', 'pca', '2-means', 'rp', 'sway', 'sway2']
    trees = ['sway', 'sway2']
    spill_rates = [0.01, 0.05, 0.10]
    
    for x in trees:
        for y in spill_rates:
            for z in xrange(trials):
                #matrixDemoTestWorker(size=None, dimensions=None, tree_type=None, spill_rate=None)
                res = matrixDemoTestWorker(tree_type=x, spill_rate=y)
                for i in res:
                    if i in results:
                        results[i] += res[i]
                    else:
                        results[i] = res[i]
    
    for x in results:
        results[x] /= trials
    return results
    
def pretty_print(d):
    
    for k in sorted(d.keys()):
        print k,"\t\t : ", d[k]

data = matrixTestMaster(0, 1)
pretty_print(data)