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
from sklearn import tree
from random import shuffle

results_memo = {}

@contextmanager
def timer(outfile):
  t1 = time()
  yield
  t2 = time()
  outfile.write(str(t2-t1)+",")


def distance(one, two):
    return sqrt(numpy.sum(numpy.subtract(one, two) ** 2) / len(one))

def cart_test(files, preface, outpath):
    #create a folder for it, then put all this stuff in.
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    if not os.path.exists(outpath+"\\"):
        os.makedirs(outpath+"\\")

    for f1 in files:
        print(f1)
        first = True
        filedata = []
        testing = []
        training = []
        displayfile = os.path.basename(f1).split(".")[0]  # get only the filename from the path

        index = "cart" + "_" + str(displayfile)
        path = outpath + index

        timerMake = open(path + "_make_times.txt", 'w')
        timerRecall = open(path + "_recall_times.txt", 'w')
        testing, non, sense = csv_reader_remove_duplicates(preface + f1, mini=False)

        for f2 in files:
            if f1 != f2:
                if first:
                    filedata, reduceddata, nothing = csv_reader_remove_duplicates(preface+f2, mini=False)
                    first = False
                else:
                    a, b, c = csv_reader_remove_duplicates(preface+f2, mini=False)
                    filedata.extend(a)
                    reduceddata.extend(b)

        training = deepcopy(filedata)

        path = outpath + index
        f = open(path + ".txt", 'w')
        # make a version of filedata that doesn't have the class in it so things don't get sorted by class
        indep = []
        dep = []
        for row in training:
            indep.append(row[:-1])
            dep.append(row[-1])
        print(training)
        print(testing)
        with timer(timerMake):
            cartTree = tree.DecisionTreeRegressor().fit(indep, dep)
        i = 0
        j = 0
        for test_point in testing:
            i += 1
            actual = 1 if int(test_point[-1]) > 0.5 else 0
            test_point = numpy.array(test_point[:-1])

            with timer(timerRecall):
                predicted = 1 if cartTree.predict(test_point)[0] > 0.5 else 0
            if actual != predicted:
                j += 1

            f.write(str(actual) + "," + str(predicted) + "\n")
        print(i)
        print(j)
        print(len(testing))
        f.close()
        timerMake.close()
        timerRecall.close()

def matrixDemoTestWorker(trials=None, tree_type=None, spill_rate=None, k_neighbors=None, tree_depth=None, files=None, basepath=None):
    run = 0
    # Create testing variables
    treet = tree_type or 'kd'
    spill = spill_rate or .25
    k_near = k_neighbors
    depth = tree_depth or 0
    filename = files

    # Python interprets spill_layer = 0 as False, and so sets spill to .25
    if spill_rate == 0:
        spill = 0

    # read in the data
    filedata = csv_reader2(filename, mini=False)
    reduceddata = []
    #filedata, reduceddata, nothing = csv_reader2(filename, mini=False)
    #filedata, reduceddata, nothing = csv_reader_remove_duplicates_and_normalize(filename, mini=True)
    """
    if reduceddata == "error" and treet == 'entropic':
        print("insufficient eigenvectors for entropic")
        return
    """
    N = len(filedata)
    D = len(filedata[0])

    displayfile = os.path.basename(filename).split(".")[0] # get only the filename from the path
    index = 'k_' + str(k_near) + '_' + str(spill) + '_' + str(tree_depth) + "_" + str(displayfile) + '_' + treet

    #create a folder for it, then put all this stuff in.
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    #if not os.path.exists(basepath+"global\\"):
        #os.makedirs(basepath+"global\\")
    path = basepath + index

    f = open(path + ".txt", 'w')
    f.write(displayfile + "," + index + "\n")
    #timerfile = open(path + "_times.txt", 'w')
    timerMake = open(path + "_times.txt", 'w')
    #timerfile_global = open(path +"_global_times.txt", 'w')
    #timerRecall = open(path+"recall_times.txt", 'w')

    for runs in range(trials):
        #c = list(zip(filedata, reduceddata))
        #random.shuffle(c)
        #filedata, reduceddata = zip(*c)
        random.shuffle(filedata)

        # make a version of filedata that doesn't have the class in it so things don't get sorted by class
        runfiledata = deepcopy(filedata)
        for row in runfiledata:
            del row[-1]


        indep = []
        dep = []
        for row in filedata:
            indep.append(row[:-1])
            dep.append(row[-1])


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
        """
        for item in reduceddata:
            evectValidationGroups[i % 5].append(item)
            i += 1
        """

        for k in range(len(validationGroups)):
            run += 1
            index = str(run) + '-k_' + str(k_near) + '_' + str(spill) + '_' + str(tree_depth) + "_" + str(
                displayfile) + '_' + treet
            index = str(run) + "-" + displayfile +  "_cart"
            path = basepath + index
            f = open(path + ".txt", 'w')
            #f_global = open(basepath + index + '_global.txt', 'w')
            f.write(displayfile + "," + index + "\n")
            #f_global.write(displayfile + "," + index + "\n")

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


            with timer(timerMake):
                cartTree = tree.DecisionTreeRegressor().fit(indep, dep)

            """
            if treet == 'entropic':
                with timer(timerfile):
                    T = spatialtree(trainingevect, spill=spill, rule=treet, height=tree_depth)
                with timer(timerfile_global):
                    T_global = spatialtree(trainingevect, height=0)
            else:
                with timer(timerfile):
                    T = spatialtree(training, spill=spill, rule=treet, height=tree_depth)
                T_global = spatialtree(training, height=0)

            # To compare accuracy against brute force, make a height=0 tree (will do a linear search for knn)
            #T_root = spatialtree(training, height=0)
            """
            recall = 0

            # Generate test points from the test set
            for test_point in range(len(validationGroups[k])):
                test = validationGroups[k][test_point]
                #testevect = evectValidationGroups[k][test_point]

                #actual classification
                actual = testingValidationGroups[k][test_point][-1]

                """
                #find approximate knn
                if treet != 'entropic':
                    with timer(timerRecall):
                        knn_approx = T.k_nearest(training, k=k_near, vector=test)
                    with timer(timerfile_global):
                        knn_global = T_global.k_nearest(training, k=k_near, vector=test)
                else:
                    knn_approx = T.k_nearest(trainingevect, k=k_near, vector=testevect)
                    knn_global = T_global.k_nearest(training, k=k_near, vector=testevect)
                    #print("a")
                """
                # do CART
                predicted = 1 if cartTree.predict(test)[0] > 0.5 else 0

                #predicted classification (class should be 1 or 0.  Sum, divide by n and round to 0 or 1 for majority vote)
                #predicted = round(sum([training[kl][-1] if training[kl][-1] > 0 else 0 for kl in knn_approx])/k_near)
                """
                if treet != 'entropic':
                    dist = sum([distance(training[kl], test) for kl in knn_approx])
                else:
                    dist = sum([distance(trainingevect[kl], testevect) for kl in knn_approx])

                if dist < 1e-10:
                    predicted = testing[knn_approx[0]][-1]
                else:
                    if treet != 'entropic':
                        predicted = round(sum(
                            [testing[kl][-1] * distance(training[kl], test) for kl
                             in knn_approx]) / dist)
                    else:
                        predicted = round(sum([testing[kl][-1]*distance(trainingevect[kl], testevect) for kl in knn_approx]) / dist)
                """
                #predicted = round(sum([testing[kl][-1] if testing[kl][-1] > 0 else 0 for kl in knn_approx])/k_near)
                #predicted_global = round(sum([testing[kl][-1] if testing[kl][-1] > 0 else 0 for kl in knn_global])/k_near)

                #first round, merely classifying as buggy or not

                if predicted >= 0.5 and actual >= 0.5: # datasets are number of bugs, average of over 1/2 means buggy
                    predicted = actual = 1
                else:
                    predicted = int(max(min(predicted, 1), 0))
                    actual = int(max(min(actual, 1), 0))
                """
                if predicted_global >= 0.5:
                    predicted_global = 1
                else:
                    predicted_global = int(max(min(predicted_global, 1), 0))
                actual = 1 if actual >= 0.5 else 0
                """

                f.write(str(actual) + "," + str(predicted) + "\n")
                #f_global.write(str(actual) + "," + str(predicted_global) + "\n")
            f.close()
            #f_global.close()
    timerMake.close()
    #timerfile.close()
    #timerfile_global.close()
    #timerRecall.close()


def transfer_learning(trials=None, tree_type=None, spill_rate=None, k_neighbors=None, tree_depth=None, files=None, basepath=None, mini=False, preface=None):
    # Create testing variables
    treet = tree_type or 'kd'
    spill = spill_rate or .25
    k_near = k_neighbors
    depth = tree_depth or 0
    filename = files

    # Python interprets spill_layer = 0 as False, and so sets spill to .25
    if spill_rate == 0:
        spill = 0
    for f in files:
        # read in the data
        #filedata, reduceddata, nothing = csv_reader2(preface+f, mini)
        filedata = None
        reduceddata = None
        testset = None
        testset, reducedtesting, nothing = csv_reader_remove_duplicates(preface+f, mini)

        shorttesting = deepcopy(testset)
        testing = numpy.array(testset)

        for row in shorttesting:
            del row[-1]
        for g in files:
            if f != g:
                if filedata is None:
                    filedata, reduceddata, nothing = csv_reader_remove_duplicates(preface+g, mini)
                else:
                    a, b, c = csv_reader_remove_duplicates(preface+g, mini)
                    filedata.extend(a)
                    reduceddata.extend(b)
                    nothing.extend(c)
        # extra remove duplicates

        """
        data = [[float(r) for r in line.rstrip('\n').split(',')] for line in temp_file]
        data.sort()
        data = list(d for d, _ in itertools.groupby(data))
        shuffle(data)
        """

        #filedata, reduceddata, nothing = csv_reader_remove_duplicates_and_normalize(filename, mini=True)
        if reduceddata == "error" and treet == 'entropic':
            print("insufficient eigenvectors for entropic")
            return

        N = len(filedata)
        D = len(filedata[0])

        displayfile = f.split(".")[0] # get only the filename from the path
        index = 'k_' + str(k_near) + '_' + str(spill) + '_' + str(tree_depth) + "_" + str(displayfile) + '_' + treet

        path = basepath + index

        timerMake = open(path + "_make_times_transfer.txt", 'w')
        timerRecall = open(path + "_recall_times_transfer.txt", 'w')
        #timerMakeGlobal = open(path + "_make_times_global_transfer.txt", 'w')
        #timerRecallGlobal = open(path + "_recall_times_global_transfer.txt", 'w')

        # make a version of filedata that doesn't have the class in it so things don't get sorted by class
        runfiledata = deepcopy(filedata)
        for row in runfiledata:
            del row[-1]

        index = '1-k' + str(k_near) + '_' + str(spill) + '_' + str(tree_depth) + "_" + str(
            displayfile) + '_' + treet
        path = basepath + index
        f = open(path + "_transfer.txt", 'w')
        #f_global = open(basepath + index + '_global_transfer.txt', 'w')
        f.write(displayfile + "," + index + "\n")
        #f_global.write(displayfile + "," + index + "\n")

        indep = []
        dep = []
        for row in filedata:
            indep.append(row[:-1])
            dep.append(row[-1])

        training = numpy.array(runfiledata)

        with timer(timerMake):
            cartTree = tree.DecisionTreeRegressor().fit(indep, dep)
        """
        with timer(timerMakeGlobal):
            T_global = spatialtree(training, height=0)
        with timer(timerMake):
            T = spatialtree(training, spill=spill, rule=treet, height=tree_depth)
        """
        # To compare accuracy against brute force, make a height=0 tree (will do a linear search for knn
        # T_root = spatialtree(training, height=0)

        recall = 0

        # Generate test points from the test set
        for test_point in range(len(testing)):
            test = shorttesting[test_point]

            # actual classification
            actual = testing[test_point][-1]

            # find approximate knn
            if treet != 'entropic':
                """
                with timer(timerRecall):
                    knn_approx = T.k_nearest(training, k=k_near, vector=test)
                with timer(timerRecallGlobal):
                    knn_global = T_global.k_nearest(training, k=k_near, vector=test)
                """
                with timer(timerRecall):
                    predicted = 1 if cartTree.predict(test)[0] > 0.5 else 0
            else:
                print("ERROR AT RECALL IN CROSS TESTING")
                return

            # predicted classification (class should be 1 or 0.  Sum, divide by n and round to 0 or 1 for majority vote)
            # predicted = round(sum([training[kl][-1] if training[kl][-1] > 0 else 0 for kl in knn_approx])/k_near)
            """
            if treet != 'entropic':
                dist = sum([distance(training[kl], test) for kl in knn_approx])
            else:
                dist = sum([distance(trainingevect[kl], testevect) for kl in knn_approx])

            if dist < 1e-10:
                predicted = testing[knn_approx[0]][-1]
            else:
                if treet != 'entropic':
                    predicted = round(sum(
                        [testing[kl][-1] * distance(training[kl], test) for kl
                         in knn_approx]) / dist)
                else:
                    predicted = round(sum([testing[kl][-1]*distance(trainingevect[kl], testevect) for kl in knn_approx]) / dist)
            """

            """
            predicted = round(
                sum([filedata[kl][-1] if filedata[kl][-1] > 0 else 0 for kl in knn_approx]) / k_near)
            predicted_global = round(
                sum([filedata[k2][-1] if filedata[k2][-1] > 0 else 0 for k2 in knn_global]) / k_near)
            """
            # first round, merely classifying as buggy or not
            if predicted >= 0.5 and actual >= 0.5:  # datasets are number of bugs, average of over 1/2 means buggy
                predicted = actual = 1
            else:
                predicted = int(max(min(predicted, 1), 0))
                actual = int(max(min(actual, 1), 0))
            """
            if predicted_global >= 0.5:
                predicted_global = 1
            else:
                predicted_global = int(max(min(predicted_global, 1), 0))
            """
            # Now, get the true nearest neighbors (want to compare results with this????)
            # knn_t = T_root.k_nearest(training, k=k_near, vector=test2)

            f.write(str(actual) + "," + str(predicted) + "\n")
            #f_global.write(str(actual) + "," + str(predicted_global) + "\n")
        f.close()
        #f_global.close()
    timerMake.close()
    #timerMakeGlobal.close()
    timerRecall.close()
    #timerRecallGlobal.close()




def matrixTestMaster(trials, size=None, dimensions=None, tree_type=None, spill_rate=None, samp=None,
                     k_neighbors=None):
    # call the method with different tree types, spill levels, et al  many times and average/collate the data
    # All cases listed first for ease of reference
    # results = dict()
    # trees = ['kd', 'pca', '2-means', 'rp', 'where', 'random', 'spectral', 'entropic']
    # spill_rates = [0, 0.01, 0.05, 0.10, 0.15, 0.2, 0.25]
    # tree_depth = [5, 6, 7, 8, 9, 10, 11, 12, 13]
    #files = ['accumulo.csv', 'bookkeeper.csv', 'camel.csv', 'cassandra.csv', 'cxf.csv', 'derby.csv', 'felix.csv', 'hive.csv', 'openjpa.csv', 'pig.csv', 'wicket.csv']
    #files = ['ant.csv', 'arc.csv', 'berek.csv', 'camel.csv', 'elearning.csv', 'ivy.csv', 'jedit.csv', 'log4j.csv',
             #'lucene.csv', 'poi.csv', 'prop6.csv', 'synapse.csv', 'tomcat.csv', 'xalan.csv', 'xerces.csv']
    #nearest_neighbors = [1, 5, 10, 15, 30, 50, 100]
    tree_depths = [5]
    trees = ['kd']
    spill_rates = [0.25]
    #files = ['accumulo.csv', 'bookkeeper.csv', 'camel.csv', 'cassandra.csv', 'cxf.csv', 'derby.csv', 'felix.csv', 'hive.csv', 'openjpa.csv', 'pig.csv', 'wicket.csv']
    files = ['ant2.csv', 'arc2.csv', 'berek2.csv', 'camel2.csv', 'elearning2.csv', 'ivy2.csv','jedit2.csv', 'log4j2.csv', 'lucene2.csv', 'poi2.csv', 'prop62.csv', 'synapse2.csv', 'xerces2.csv']
    #files = ['jm1.csv', 'kc2.csv']
    nearest_neighbors = [3]
    outpath = 'C:\\Users\\Andrew\\Documents\\Schools\\Grad School\\NCSU - Comp Sci\\Research\\Overlaping Trees\\Data\\1st Run\\'
    #outpath = 'C:\\Users\\Andrew\\Documents\\Schools\\Grad School\\NCSU - Comp Sci\\Research\\Overlaping Trees\\Data\\Global vs Local\\'

    preface = '.\\Mining Datasets\\Bellweather\\Promise Datasets\\CK2\\'
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
                                print(err)
                                sys.exit()
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

def transfer_learning_master(size=None, dimensions=None, tree_type=None, spill_rate=None, samp=None,
                     k_neighbors=None):
    tree_depths = [5]
    trees = ['cart']
    spill_rates = [0.25]
    files = ['accumulo.csv', 'bookkeeper.csv', 'camel.csv', 'cassandra.csv', 'cxf.csv', 'derby.csv', 'felix.csv', 'hive.csv', 'openjpa.csv', 'pig.csv', 'wicket.csv']
    #files = ['ant2.csv', 'arc2.csv', 'berek2.csv', 'camel2.csv', 'elearning2.csv', 'ivy2.csv','jedit2.csv', 'log4j2.csv', 'lucene2.csv', 'poi2.csv', 'synapse2.csv', 'xerces2.csv']
    #files = ['jm1.csv', 'kc2.csv']
    nearest_neighbors = [3]
    outpath = 'C:\\Users\\Andrew\\Documents\\Schools\\Grad School\\NCSU - Comp Sci\\Research\\Overlaping Trees\\Data\\First Dataset\\'
    preface = '.\\Mining Datasets\\Bellweather\\'
    errorLog = open(outpath + "errors.txt", 'w')

    for c in nearest_neighbors:
        for x in range(len(trees)):
            for y in range(len(spill_rates)):
                for z in range(len(tree_depths)):
                    try:
                        print(c, x, y, z)
                        transfer_learning(tree_type=trees[x], spill_rate=spill_rates[y], tree_depth=tree_depths[z],
                                             files=files, k_neighbors=c, basepath=outpath, preface=preface)
                    except LinAlgError as err:
                        print("error")
                        errorLog.write(str(err) + " in " + str(c) + "_" + str(trees[x]) + "_" + str(
                            spill_rates[y]) + "_" + str(tree_depths[z])+"\n")
                        pass
                    except Exception as err :
                        print("error")
                        errorLog.write(str(err) + " in " + str(c) + "_" + str(trees[x]) + "_" + str(
                            spill_rates[y]) + "_" + str(tree_depths[z])+"\n")
                        pass

    errorLog.close()

matrixTestMaster(5)
#transfer_learning_master()
