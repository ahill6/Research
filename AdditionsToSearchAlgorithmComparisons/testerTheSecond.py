#!/usr/bin/env python
'''
CREATED:2011-11-12 08:23:33 by Brian McFee <bmcfee@cs.ucsd.edu>
MODIFIED: starting 2016-09-08 by Andrew Hill <ahill6@ncsu.edu>
Spatial tree demo for matrix data
'''
import numpy, random, sys, pprint, timeit
from spatialtree import spatialtree
from fileio import summarize, csv_reader, make_stats


results_memo = {}
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
    filename = "mccabes_mc12.csv"
    
    # Python interprets spill_layer = 0 as False, and so sets spill to .25
    if spill_rate == 0:
        spill = 0
    
        
    # read in the data
    X = csv_reader(filename)

    # divide the data into 5 groups for cross validation
    i = 0
    random.shuffle(X)
    
    Y = [[] for i in xrange(5)]
    for item in X:
        Y[i%5].append(item)
        i += 1
    """
    for i in xrange(5):
        print(len(Y[i]))
    """
    timer = open(tree, 'w')

    # for each, use the other 4 groups for training, test on the remaining group
    for item in Y:
        t = []
        for x in Y:
            if x != item:
                t.extend(x)
        training = numpy.array(t)
        print "Building tree...", tree, spill, tree_depth
        #T = spatialtree(training, rule=tree, spill=spill, height = tree_depth)
        start_time = timeit.default_timer()
        T = spatialtree(training, rule=tree, spill=spill, height = tree_depth)
        elapsed = timeit.default_timer() - start_time
        timer.write(str(elapsed)+",")
        #print "Done
        T_root = spatialtree(training, rule=tree, spill=spill, height = 0) # this should be training and not X, right?
        recall = 0
        index = '' + tree + '_' + str(spill) + '_' + str(tree_depth)
        f = open(index+".txt", 'w')
        for test in item:
            knn_a = T.k_nearest(training, k=k_near, vector = test)
            knn_t = T_root.k_nearest(training, k=k_near, vector = test)
            #true_pos = len(set(knn_a) & set(knn_t))*1.0/len(set(knn_t))
            true_pos = len(set(knn_a) & set(knn_t))*1.0
            false_pos = len(set(knn_t)) - true_pos
            true_neg = len(training) - false_pos
            f.write(str(true_pos)+'-'+str(false_pos)+'-'+str(true_neg)+',')
            recall += true_pos/len(set(knn_t))
        #print_recall = recall*1.0/len(item)
        #f.write(str(print_recall)+',')
        #print(""+tree+" recall\t", print_recall) 
        #results_memo[index] = print_recall
       
    f.write("\n")
    f.close()
    timer.close()
    # collate and analyze results
    

def matrixTestMaster(samples, trials, size = None, dimensions = None, tree_type = None, spill_rate = None, samp = None, k_neighbors = None):
    #call the method with different tree types, spill levels, et al  many times and average/collate the data 
    #All cases listed first for ease of reference
    #results = dict()
    #trees = ['kd', 'pca', '2-means', 'rp', 'where', 'random']
    #spill_rates = [0, 0.01, 0.05, 0.10, 0.15, 0.2, 0.25]
    #tree_depth = [5, 6, 7, 8, 9, 10, 11, 12, 13]
    tree_depths = [5, 9, 13]
    trees = ['pca', 'where', 'random']
    spill_rates = [0, 0.01, 0.25]
    myArray=[[[[0 for a in range(2)] for k in range(len(tree_depths))] for j in range(len(spill_rates))] for i in range(len(trees))]
    
    
    for x in xrange(len(trees)):
        for y in xrange(len(spill_rates)):
            for z in xrange(len(tree_depths)):
                for a in xrange(trials):
                    matrixDemoTestWorker(tree_type = trees[x], spill_rate = spill_rates[y], tree_depth = tree_depths[z])

    # call the fileio summarize method to give quartiles of data collected
    #for t in trees:
        #summarize(str(t))
        #summarize(t+"_0.05_5")
    #pretty_print(results_memo)
    make_stats()

def pretty_print(d):
    for k in sorted(d.keys()):
        print k,"\t\t : ", d[k]


def testMaster():
    # read in the data
    data  = 1
    # divide the data into 5 groups for cross validation
    
    # for each, use the other 4 groups for training, test on the remaining group
    
    # collate and analyze results
    
    
matrixTestMaster(0, 1)
#summarize(".01_5")