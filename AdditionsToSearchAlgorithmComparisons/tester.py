#!/usr/bin/env python
'''
CREATED:2011-11-12 08:23:33 by Brian McFee <bmcfee@cs.ucsd.edu>
MODIFIED: starting 2016-09-08 by Andrew Hill <ahill6@ncsu.edu>
Spatial tree demo for matrix data
'''
import numpy, random, sys, pprint
from spatialtree import spatialtree
from fileio import summarize, csv_reader

def matrixDemoTestWorker(size = None, dimensions = None, tree_type = None, spill_rate = None, samp = None, k_neighbors = None, tree_depth=None):
    #this_run = dict()
    this_run = [0 for i in xrange(2)]
    
    # Create random matrix
    N = size or 5000
    D = dimensions or 20 
    
    # Create testing variables
    tree= tree_type or 'kd'
    spill = spill_rate or .25
    samples = samp or 100
    k_near = k_neighbors or 10
    k = 5
    max_value = 100
    filename = "testdata.csv"
    
    # Python interprets spill_layer = 0 as False, and so sets spill to .25
    if spill_rate == 0:
        spill = 0
    
    
    """
    X = numpy.random.randn(N, D)
    P = numpy.random.randn(D, D)
    X = numpy.dot(X, P)
    """
    
    #If you want to use data from a file, uncomment this and put the filename in here
    X = csv_reader("testdata.csv")
    D = len(X[0])
    N = len(X)
    P = numpy.random.randn(D,D)
    
    
    # Apply a few random projections so the data's not totally boring
    # Goal: embed k dimensional data in D-space
    """
    for i in xrange(k):
        P = numpy.random.randn(D, D)
        X = numpy.dot(X, P)
    """
    
    # Construct the type of tree specified with spill specified.
    # Defaults are KD-spill-tree with spill = 25%
    print "Building tree...", tree, spill, tree_depth
    T = spatialtree(X, rule=tree, spill=spill, height = tree_depth)
    print "Done"
    
    T_root = spatialtree(X, rule=tree, spill=spill, height = 0)
    
    print("Running tests...")
    # Test recall on items in tree
    in_tree_recall = 0
    out_of_tree_recall = 0
    
    index = '' + tree + '_' + str(spill) + '_' + str(tree_depth)
    f = open(index+".txt", 'w')
    
    for countvar in range(samples):
        rand = random.randint(0,N-1)
        knn_a = T.k_nearest(X, k=k_near, index = rand)
        knn_t = T_root.k_nearest(X, k=k_near, index=rand)
        value = len(set(knn_a) & set(knn_t))*1.0/len(set(knn_t))
        f.write(str(value))
        if countvar != samples-1:
            f.write(", ")
        in_tree_recall += value
      
        """       
        true_pos += len(set(knn_a) & set(knn_t))
        false_pos += in_tree_count - true_pos
        false_neg += in_tree_count - true_pos
        true_neg += samples - false_neg
        """
        
    f.write("\n")
        
    # We can also search with a new vector not already in the tree
    for countvar in range(samples):
        query = numpy.dot(numpy.random.randn(D), P)
        knn_a = T.k_nearest(X, k=k_near, vector=query)
        knn_t = T_root.k_nearest(X, k=k_near, vector=query)
        value = len(set(knn_a) & set(knn_t))*1.0/len(set(knn_t))
        #f.write(str(format(len(set(knn_a) & set(knn_t)) * 1.0 / len(set(knn_t)), '.4f')))
        f.write(str(value))
        if countvar != samples-1:
            f.write(', ')
        out_of_tree_recall += value
    f.write("\n")
    f.close()    
    
    print "in tree_recall\t", in_tree_recall*1.0/samples 
    print "out of tree_recall\t", out_of_tree_recall*1.0/samples 
    print("Done")
    
    #return this_run
    

def matrixTestMaster(samples, trials, size = None, dimensions = None, tree_type = None, spill_rate = None, samp = None, k_neighbors = None):
    #call the method with different tree types, spill levels, et al  many times and average/collate the data 
    #All cases listed first for ease of reference
    #results = dict()
    #trees = ['kd', 'pca', '2-means', 'rp', 'sway']
    #spill_rates = [0, 0.01, 0.05, 0.10, 0.15, 0.2, 0.25]
    #tree_depth = [5, 6, 7, 8, 9, 10, 11, 12, 13]
    tree_depths = [5,13]
    trees = ['pca','rp', 'sway']
    spill_rates = [0.05]
    myArray=[[[[0 for a in range(2)] for k in range(len(tree_depths))] for j in range(len(spill_rates))] for i in range(len(trees))]
    
    
    for x in xrange(len(trees)):
        for y in xrange(len(spill_rates)):
            for z in xrange(len(tree_depths)):
                for a in xrange(trials):
                    matrixDemoTestWorker(tree_type = trees[x], spill_rate = spill_rates[y], tree_depth = tree_depths[z])

    # call the fileio summarize method to give quartiles of data collected
    for t in tree_depths:
        summarize(str(t))

def pretty_print(d):
    for k in sorted(d.keys()):
        print k,"\t\t : ", d[k]


def testMaster():
    # read in the data
    data  = 1
    # divide the data into 5 groups for cross validation
    
    # for each, use the other 4 groups for training, test on the remaining group
    
    # collate and analyze results
    
    
#matrixTestMaster(0, 1)
matrixDemoTestWorker()
#summarize(".01_5")