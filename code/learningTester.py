#!/usr/bin/env python
'''
CREATED:2011-11-12 08:23:33 by Brian McFee <bmcfee@cs.ucsd.edu>
MODIFIED: starting 2016-09-08 by Andrew Hill <ahill6@ncsu.edu>
Spatial tree demo for matrix data
'''
import numpy, random, sys, pprint, os
from spatialtree import spatialtree
from localio import summarize, csv_reader2, make_stats, abcd_master, menzies_stats, stats_on_stats, csv_reader_remove_duplicates, csv_reader_remove_duplicates_and_normalize
from numpy.linalg.linalg import LinAlgError
from copy import deepcopy
from contextlib import contextmanager
from time import time
from math import sqrt

results_memo = {}

@contextmanager
def timer(outfile):
  t1 = time()
  yield
  t2 = time()
  outfile.write(str(t2-t1)+",")


def distance(one, two):
    return sqrt(numpy.sum(numpy.subtract(one, two) ** 2) / len(one))

def matrixDemoTestWorker(trials=None, tree_type=None, spill_rate=None, k_neighbors=None, tree_depth=None, files=None, basepath=None):
    run = 0
    # Create testing variables
    tree = tree_type or 'kd'
    spill = spill_rate or .25
    k_near = k_neighbors
    depth = tree_depth or 0
    filename = files

    # Python interprets spill_layer = 0 as False, and so sets spill to .25
    if spill_rate == 0:
        spill = 0

    # read in the data
    # filedata, reduceddata, nothing = csv_reader2(filename, mini=True)
    #filedata, reduceddata, nothing = csv_reader_remove_duplicates(filename, mini=True)
    filedata, reduceddata, nothing = csv_reader_remove_duplicates_and_normalize(filename, mini=True)
    if reduceddata == "error" and tree == 'entropic':
        print("insufficient eigenvectors for entropic")
        return
    N = len(filedata)
    D = len(filedata[0])

    displayfile = os.path.basename(filename).split(".")[0] # get only the filename from the path
    index = 'k_' + str(k_near) + '_' + str(spill) + '_' + str(tree_depth) + "_" + str(displayfile) + '_' + tree

    #create a folder for it, then put all this stuff in.
    if not os.path.exists(basepath):
        os.makedirs(basepath)

    path = basepath + index

    #f = open(path + ".txt", 'w')
    #f.write(displayfile + "," + index + "\n")
    timerfile = open(path + "_times.txt", 'w')

    for runs in range(trials):
        c = list(zip(filedata, reduceddata))
        random.shuffle(c)
        filedata, reduceddata = zip(*c)

        # make a version of filedata that doesn't have the class in it so things don't get sorted by class
        runfiledata = deepcopy(filedata)
        for row in runfiledata:
            del row[-1]

        # divide the data into 5 groups for cross validation

        validationGroups = [[] for i in range(5)]
        evectValidationGroups = [[] for i in range(5)]
        testingValidationGroups = [[] for i in range(5)]

        i = 0
        for item in runfiledata:
            validationGroups[i % 5].append(item)
            i += 1
        i = 0
        for item in filedata:
            testingValidationGroups[i % 5].append(item)
            i += 1
        i = 0
        for item in reduceddata:
            evectValidationGroups[i % 5].append(item)
            i += 1

        for k in range(len(validationGroups)):
            run += 1
            index = str(run) + '-k_' + str(k_near) + '_' + str(spill) + '_' + str(tree_depth) + "_" + str(
                displayfile) + '_' + tree
            path = basepath + index
            f = open(path + ".txt", 'w')
            f.write(displayfile + "," + index + "\n")

            #print("Building ", tree_type)
            t = []
            for x in range(len(evectValidationGroups)):
                if x != k:
                    t.extend(evectValidationGroups[x])
            trainingevect = numpy.array(t)
            t2 = []
            for x in range(len(validationGroups)):
                if x != k:
                    t2.extend(validationGroups[x])
            training = numpy.array(t2)
            t3 = []
            for x in range(len(testingValidationGroups)):
                if x != k:
                    t3.extend(testingValidationGroups[x])
            testing = numpy.array(t3)

            #need to get this part working
            if tree == 'entropic':
                with timer(timerfile):
                    T = spatialtree(trainingevect, spill=spill, rule=tree, height=tree_depth)
            else:
                with timer(timerfile):
                    T = spatialtree(training, spill=spill, rule=tree, height=tree_depth)

            # To compare accuracy against brute force, make a height=0 tree (will do a linear search for knn
            #T_root = spatialtree(training, height=0)

            recall = 0

            # Generate test points from the test set
            for test_point in range(len(validationGroups[k])):
                test = validationGroups[k][test_point]
                testevect = evectValidationGroups[k][test_point]

                #actual classification
                actual = testingValidationGroups[k][test_point][-1]

                #find approximate knn
                if tree != 'entropic':
                    knn_approx = T.k_nearest(training, k=k_near, vector=test)

                else:
                    knn_approx = T.k_nearest(trainingevect, k=k_near, vector=testevect)

                #predicted classification (class should be 1 or 0.  Sum, divide by n and round to 0 or 1 for majority vote)
                #predicted = round(sum([training[kl][-1] if training[kl][-1] > 0 else 0 for kl in knn_approx])/k_near)

                if tree != 'entropic':
                    dist = sum([distance(training[kl], test) for kl in knn_approx])
                else:
                    dist = sum([distance(trainingevect[kl], testevect) for kl in knn_approx])

                if dist < 1e-10:
                    predicted = testing[knn_approx[0]][-1]
                else:
                    if tree != 'entropic':
                        predicted = round(sum(
                            [testing[kl][-1] * distance(training[kl], test) for kl
                             in knn_approx]) / dist)
                    else:
                        predicted = round(sum([testing[kl][-1]*distance(trainingevect[kl], testevect) for kl in knn_approx]) / dist)

                #first round, merely classifying as buggy or not
                if predicted >= 0.5 and actual >= 0.5: # datasets are number of bugs, average of over 1/2 means buggy
                    predicted = actual = 1
                else:
                    predicted = int(max(min(predicted, 1), 0))
                    actual = int(max(min(actual, 1), 0))

                # Now, get the true nearest neighbors (want to compare results with this????)
                #knn_t = T_root.k_nearest(training, k=k_near, vector=test2)

                f.write(str(actual) + "," + str(predicted) + "\n")
                #f2.write(str(actual) + "," + str(predicted) + "\n")
            f.close()
            #f2.close()
    timerfile.close()


def matrixTestMaster(trials, size=None, dimensions=None, tree_type=None, spill_rate=None, samp=None,
                     k_neighbors=None):
    # call the method with different tree types, spill levels, et al  many times and average/collate the data
    # All cases listed first for ease of reference
    # results = dict()
    # trees = ['kd', 'pca', '2-means', 'rp', 'where', 'random', 'spectral', 'entropic']
    # spill_rates = [0, 0.01, 0.05, 0.10, 0.15, 0.2, 0.25]
    # tree_depth = [5, 6, 7, 8, 9, 10, 11, 12, 13]
    #files = ['accumulo.csv', 'bookkeeper.csv', 'camel.csv', 'cassandra.csv', 'cxf.csv', 'derby.csv', 'felix.csv', 'hive.csv', 'openjpa.csv', 'pig.csv', 'wicket.csv']
    #nearest_neighbors = [1, 5, 10, 15, 30, 50, 100]
    tree_depths = [5, 7, 9, 11, 13]
    trees = ['kd', 'pca', '2-means', 'rp', 'where', 'random', 'spectral', 'entropic']
    spill_rates = [0, 0.01, 0.05, 0.10, 0.15, 0.2, 0.25]
    files = ['bookkeeper.csv', 'camel.csv', 'cassandra.csv', 'cxf.csv', 'derby.csv', 'felix.csv', 'hive.csv', 'openjpa.csv', 'pig.csv', 'wicket.csv']
    nearest_neighbors = [3]
    outpath = 'C:\\Users\\Andrew\\Documents\\Schools\\Grad School\\NCSU - Comp Sci\\Research\\Overlaping Trees\\Data\\10 dups removed weighted normalized\\'

    preface = '.\\Mining Datasets\\Bellweather\\'
    """
    for dirName, subdirList, fileList in os.walk(preface):
        print(dirName, fileList)
    sys.exit(0)
    """
    errorLog = open(outpath + "errors.txt", 'w')

    for c in nearest_neighbors:
        for b in files:
            for x in range(len(trees)):
                for y in range(len(spill_rates)):
                    for z in range(len(tree_depths)):
                            try:
                                print(c, b, x, y, z)
                                matrixDemoTestWorker(tree_type=trees[x], spill_rate=spill_rates[y], tree_depth=tree_depths[z],
                                                            files=preface+b, trials=trials, k_neighbors=c, basepath=outpath)
                            except LinAlgError as err:
                                print("error")
                                errorLog.write(str(err) + " in " + str(c) + "_" + b + "_" + str(trees[x]) + "_" + str(
                                    spill_rates[y]) + "_" + str(tree_depths[z])+"\n")
                                pass
                            except Exception as err :
                                print("error")
                                errorLog.write(str(err) + " in " + str(c) + "_" + b + "_" + str(trees[x]) + "_" + str(
                                    spill_rates[y]) + "_" + str(tree_depths[z])+"\n")
                                pass
    # I think some k=10 accumulo runs are missing

    #make single file
    """
    fs = ['accumulo', 'bookkeeper', 'camel', 'cassandra', 'cxf', 'derby', 'felix', 'hive', 'openjpa', 'pig',
             'wicket']
    for m in fs:
        abcd_master(outpath, m)
        menzies_stats(outpath+m+"\\")

    stats_on_stats(outpath, fs, nearest_neighbors, spill_rates, tree_depths, trees)
    #run stats on those single files
    #localio.stats_master(path, make_stats_from_collected_files)
    """
    #make_stats(num_entries = n, num_decisions=d)
    errorLog.close()

matrixTestMaster(5)