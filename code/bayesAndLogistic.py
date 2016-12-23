from sklearn import linear_model, naive_bayes
from localio import csv_reader2, csv_reader_remove_duplicates_and_normalize
from copy import deepcopy
from random import shuffle, sample
from spatialtree import spatialtree
import sys, os, numpy

def naiveBayes(training, testing, outfileGauss=None, outfileMultinom=None):
    GaussNB = naive_bayes.GaussianNB()
    #MultinomNB = naive_bayes.MultinomialNB()
    indep = []
    dep = []

    for row in training:
        indep.append(row[:-1])
        dep.append(row[-1])

    GaussNB.fit(indep, dep)
    #MultinomNB.fit(indep, dep)

    for t in testing:
        predictedGauss = 1 if GaussNB.predict(t[:-1])[0] > 0.5 else 0 # could technically pass all of testing as a single matrix-like, but that could end crazy
        #predictedMultinom = 1 if MultinomNB.predict(t[:-1])[0] > 0.5 else 0
        actual = 1 if t[-1] > 0.5 else 0
        outfileGauss.write(str(actual) + ',' + str(predictedGauss)+"\n")
        #outfileMultinom.write(str(actual) + ',' + str(predictedMultinom)+"\n")

def logisticRegression(training, testing, outfile):
    indep = []
    dep = []

    for row in training:
        indep.append(row[:-1])
        dep.append(row[-1])

    clf = linear_model.LogisticRegression(C=1e5)
    clf.fit(indep, dep)

    for t in testing:
        predicted = 1 if clf.predict(t[:-1])[0] > 0.5 else 0 # could technically pass all of testing as a single matrix-like, but that could end crazy
        actual = 1 if t[-1] > 0.5 else 0
        outfile.write(str(actual) + ',' + str(predicted)+"\n")

def transferLearning(preface, files, outpath, undersample=False, globalLocal=False, knn=False):
    for f1 in files:
        filedata = None
        print("Starting transfer learning for: " + f1)
        for f2 in files:
            if f1 != f2:
                if filedata == None:
                    filedata = csv_reader2(preface + f2, mini=False)
                    # filedata = csv_reader_remove_duplicates_and_normalize(preface+f2, mini=False)
                else:
                    filedata.extend(csv_reader2(preface + f2, mini=False))
                    # filedata.extend(csv_reader_remove_duplicates_and_normalize(preface+f2, mini=False))
        training = deepcopy(filedata)
        label = f1.split('.')[0]

        if undersample or globalLocal:
            bugs = [i for i, x in enumerate(training) if x[-1] > 0.5]
            nonbugs = [i for i, x in enumerate(training) if x[-1] < 0.5]
            if undersample:
                undersampleTraining = [filedata[x] for x in bugs + sample(nonbugs, len(bugs))]
                bugs = [i for i, x in enumerate(undersampleTraining) if x[-1] > 0.5]
                nonbugs = [i for i, x in enumerate(undersampleTraining) if x[-1] < 0.5]
                training = deepcopy(undersampleTraining)
        testing = csv_reader2(preface + f1, mini=False)

        if undersample:
            label += '_Und'
        if globalLocal:
            label += '_GL'
        if not knn:
            outLogReg = open(outpath + label + "_Tran.txt", 'w')
            outGauss = open(outpath + label + "_Tran_Gauss.txt", 'w')
            # outMultiNom = open(outpath + label + "_transfer_Multinom.txt", 'w')
            outLogReg.write("Logistic Transfer,Logistic Transfer\n")
            outGauss.write("GaussianNB Transfer,GaussianNB Transfer\n")
            # outMultiNom.write("MultinomialNB Transfer,MultinomialNB Transfer\n")
        else:
            outKNN1 = open(outpath + label + "_Tran_KNN1.txt", 'w')
            outKNN1.write("KNN, KNN")
            outKNN3 = open(outpath + label + "_Tran_KNN3.txt", 'w')
            outKNN3.write("KNN, KNN")
            outKNN5 = open(outpath + label + "_Tran_KNN5.txt", 'w')
            outKNN5.write("KNN, KNN")
            outKNN10 = open(outpath + label + "_Tran_KNN10.txt", 'w')
            outKNN10.write("KNN, KNN")

        if globalLocal:
            runfiledata = deepcopy(training)
            for row in runfiledata:
                del row[-1]
            runfiledata = numpy.array(runfiledata)
            nearest = min(max(len(runfiledata) // 10, 10), len(runfiledata)//3)
            T = spatialtree(runfiledata, spill=0.25, rule='kd')
            for test in testing:
                miniTraining = [training[k] for k in T.k_nearest_with_both(runfiledata, bugs, k=nearest, vector=test[:-1]) if k != -1]
                if len(miniTraining) < 1:
                    continue
                if not knn:
                    naiveBayes(miniTraining, [test], outGauss)
                    logisticRegression(miniTraining, [test], outLogReg)
                else:
                    knna = T.k_nearest(runfiledata, k=10, vector=test[:-1])
                    predicted1 = 1 if round(sum([1 if training[kl][-1] > 0 else 0 for kl in knna[:1]])) else 0
                    predicted3 = 1 if round(sum([1 if training[kl][-1] > 0 else 0 for kl in knna[:3]])/3.0) else 0
                    predicted5 = 1 if round(sum([1 if training[kl][-1] > 0 else 0 for kl in knna[:5]])/5.0) else 0
                    predicted10 = 1 if round(sum([1 if training[kl][-1] > 0 else 0 for kl in knna])/10.0) else 0
                    actual = 1 if test[-1] > 0.5 else 0
                    outKNN1.write(str(actual) + ',' + str(predicted1) + "\n")
                    outKNN3.write(str(actual) + ',' + str(predicted3) + "\n")
                    outKNN5.write(str(actual) + ',' + str(predicted5) + "\n")
                    outKNN10.write(str(actual) + ',' + str(predicted10) + "\n")
        else:
            if not knn:
                naiveBayes(training, testing, outGauss)
                logisticRegression(training, testing, outLogReg)
            else:
                runfiledata = deepcopy(training)
                for row in runfiledata:
                    del row[-1]
                runfiledata = numpy.array(runfiledata)
                if not globalLocal:
                    T = spatialtree(runfiledata, spill=0.25, rule='kd')
                for test in testing:
                    knna = T.k_nearest(runfiledata, k=10, vector=test[:-1])
                    predicted1 = 1 if round(sum([1 if training[kl][-1] > 0 else 0 for kl in knna[:1]])) else 0
                    predicted3 = 1 if round(sum([1 if training[kl][-1] > 0 else 0 for kl in knna[:3]]) / 3.0) else 0
                    predicted5 = 1 if round(sum([1 if training[kl][-1] > 0 else 0 for kl in knna[:5]]) / 5.0) else 0
                    predicted10 = 1 if round(sum([1 if training[kl][-1] > 0 else 0 for kl in knna]) / 10.0) else 0
                    actual = 1 if test[-1] > 0.5 else 0
                    outKNN1.write(str(actual) + ',' + str(predicted1) + "\n")
                    outKNN3.write(str(actual) + ',' + str(predicted3) + "\n")
                    outKNN5.write(str(actual) + ',' + str(predicted5) + "\n")
                    outKNN10.write(str(actual) + ',' + str(predicted10) + "\n")


def regularLearning(preface, files, outpath, undersample=False, globalLocal=False, knn=False):
    for f1 in files:
        print("Starting regular learning for: " + f1)
        filedata = csv_reader2(preface + f1, mini=False)
        #filedata = csv_reader_remove_duplicates_and_normalize(preface+f1, mini=False)
        retries = 0

        for k in range(25):
            shuffle(filedata)
            index = len(filedata)//5
            training = deepcopy(filedata[index:])
            if undersample or globalLocal:
                bugs = [i for i, x in enumerate(training) if x[-1] > 0.5]
                nonbugs = [i for i, x in enumerate(training) if x[-1] < 0.5]
                if len(bugs) == 0 or len(nonbugs) == 0:
                    k -= 1
                    retries += 1
                    continue
                if undersample:
                    undersampleTraining = [filedata[x] for x in bugs + sample(nonbugs, min(len(bugs), len(nonbugs)))]
                    bugs = [i for i, x in enumerate(undersampleTraining) if x[-1] > 0.5]
                    nonbugs = [i for i, x in enumerate(undersampleTraining) if x[-1] < 0.5]
                    training = deepcopy(undersampleTraining)
            testing = deepcopy(filedata[:index])
            bugs = [i for i, x in enumerate(training) if x[-1] > 0.5]
            nonbugs = [i for i, x in enumerate(training) if x[-1] < 0.5]
            if len(bugs) == 0 or len(nonbugs) == 0:
                k -= 1
                retries += 1
                continue

            label = str(k) + "-" + f1.split('.')[0]

            if not os.path.exists(outpath):
                os.makedirs(outpath)
            if undersample:
                label += '_Und'
            if globalLocal:
                label += '_GL'
            if not knn:
                outLogReg = open(outpath + label + ".txt", 'w')
                outGauss = open(outpath + label + "_Gauss.txt", 'w')
                #outMultiNom = open(outpath + label + "Multinom.txt", 'w')
                outLogReg.write("Logistic,Logistic\n")
                outGauss.write("GaussianNB,GaussianNB\n")
                #outMultiNom.write("MultinomialNB,MultinomialNB\n")
            else:
                outKNN1 = open(outpath + label + "_Tran_KNN1.txt", 'w')
                outKNN1.write("KNN, KNN")
                outKNN3 = open(outpath + label + "_Tran_KNN3.txt", 'w')
                outKNN3.write("KNN, KNN")
                outKNN5 = open(outpath + label + "_Tran_KNN5.txt", 'w')
                outKNN5.write("KNN, KNN")
                outKNN10 = open(outpath + label + "_Tran_KNN10.txt", 'w')
                outKNN10.write("KNN, KNN")

            if globalLocal:
                runfiledata = deepcopy(training)
                for row in runfiledata:
                    del row[-1]
                nearest = min(max(len(runfiledata) // 10, 10), len(runfiledata)//3)
                runfiledata = numpy.array(runfiledata)
                T = spatialtree(runfiledata, spill=0.25, rule='kd', height=5)
                for test in testing:
                    x = T.k_nearest_with_both(runfiledata, bugs, k=nearest, vector=test[:-1])
                    if x != -1:
                        miniTraining = [training[k] for k in x]
                    else:
                        continue
                    """
                    miniTraining = [training[k] for k in T.k_nearest_with_both(runfiledata, bugs, k=nearest, vector=test[:-1]) if k != -1]
                    if len(miniTraining) < 1:
                        continue
                    """
                    if not knn:
                        naiveBayes(miniTraining, [test], outGauss)
                        logisticRegression(miniTraining, [test], outLogReg)
                    else:
                        knna = T.k_nearest(runfiledata, k=10, vector=test[:-1])
                        predicted1 = 1 if round(sum([1 if training[kl][-1] > 0 else 0 for kl in knna[:1]])) else 0
                        predicted3 = 1 if round(sum([1 if training[kl][-1] > 0 else 0 for kl in knna[:3]]) / 3.0) else 0
                        predicted5 = 1 if round(sum([1 if training[kl][-1] > 0 else 0 for kl in knna[:5]]) / 5.0) else 0
                        predicted10 = 1 if round(sum([1 if training[kl][-1] > 0 else 0 for kl in knna]) / 10.0) else 0
                        actual = 1 if test[-1] > 0.5 else 0
                        outKNN1.write(str(actual) + ',' + str(predicted1) + "\n")
                        outKNN3.write(str(actual) + ',' + str(predicted3) + "\n")
                        outKNN5.write(str(actual) + ',' + str(predicted5) + "\n")
                        outKNN10.write(str(actual) + ',' + str(predicted10) + "\n")
            else:
                if not knn:
                    naiveBayes(training, testing, outGauss)
                    logisticRegression(training, testing, outLogReg)
                else:
                    runfiledata = deepcopy(training)
                    for row in runfiledata:
                        del row[-1]
                    runfiledata = numpy.array(runfiledata)
                    if not globalLocal:
                        T = spatialtree(runfiledata, spill=0.25, rule='kd')
                    for test in testing:
                        knna = T.k_nearest(runfiledata, k=10, vector=test[:-1])
                        predicted1 = 1 if round(sum([1 if training[kl][-1] > 0 else 0 for kl in knna[:1]])) else 0
                        predicted3 = 1 if round(sum([1 if training[kl][-1] > 0 else 0 for kl in knna[:3]]) / 3.0) else 0
                        predicted5 = 1 if round(sum([1 if training[kl][-1] > 0 else 0 for kl in knna[:5]]) / 5.0) else 0
                        predicted10 = 1 if round(sum([1 if training[kl][-1] > 0 else 0 for kl in knna]) / 10.0) else 0
                        actual = 1 if test[-1] > 0.5 else 0
                        outKNN1.write(str(actual) + ',' + str(predicted1) + "\n")
                        outKNN3.write(str(actual) + ',' + str(predicted3) + "\n")
                        outKNN5.write(str(actual) + ',' + str(predicted5) + "\n")
                        outKNN10.write(str(actual) + ',' + str(predicted10) + "\n")



def driversSeat():
    # read in files
    #files = ['accumulo.csv', 'bookkeeper.csv', 'camel.csv', 'cassandra.csv', 'cxf.csv', 'derby.csv', 'felix.csv', 'hive.csv', 'openjpa.csv', 'pig.csv', 'wicket.csv']
    #files = ['ant2.csv', 'arc2.csv', 'berek2.csv', 'camel2.csv', 'elearning2.csv', 'ivy2.csv','jedit2.csv', 'log4j2.csv', 'lucene2.csv', 'poi2.csv', 'synapse2.csv', 'xerces2.csv']
    #files = ['jm1.csv', 'kc2.csv']
    #outpath = 'C:\\Users\\Andrew\\Documents\\Schools\\Grad School\\NCSU - Comp Sci\\Research\\Overlaping Trees\\Data\\Major Comparison\\'
    #inpath = '.\\Mining Datasets\\Bellweather\\'
    #inpath = '.\\Mining Datasets\\Bellweather\\Promise Datasets\\MC\\'
    #inpath = '.\\Mining Datasets\\Bellweather\\Promise Datasets\\CK2\\'
    #inpath = '.\\Mining Datasets\\Bellweather\\Promise Datasets\\CK2\\'


    file1 = ['accumulo.csv', 'bookkeeper.csv', 'camel.csv', 'cassandra.csv', 'cxf.csv', 'derby.csv', 'felix.csv',
             'hive.csv', 'openjpa.csv', 'pig.csv', 'wicket.csv']
    file2 = ['ant2.csv', 'arc2.csv', 'berek2.csv', 'camel2.csv', 'elearning2.csv', 'ivy2.csv','jedit2.csv', 'log4j2.csv', 'lucene2.csv', 'poi2.csv', 'synapse2.csv', 'xerces2.csv']
    #file3 = ['jm1.csv', 'kc2.csv']
    file3 = ['cm1.csv', 'pc1.csv', 'pc3.csv', 'pc4.csv']
    outpath = 'C:\\Users\\Andrew\\Documents\\Schools\\Grad School\\NCSU - Comp Sci\\Research\\Overlaping Trees\\Data\\Major Comparison Test\\'
    inpath1 = '.\\Mining Datasets\\Bellweather\\'
    inpath2 = '.\\Mining Datasets\\Bellweather\\Promise Datasets\\CK2\\'
    inpath3 = '.\\Mining Datasets\\Bellweather\\Promise Datasets\\MC\\'

    sets = [(file1, inpath1), (file2, inpath2), (file3, inpath3)]
    sets = [(file3, inpath3)]
    locGlobOptions = [True, False]
    undersampOptions = [True, False]
    knnOption = False
    for s in sets:
        for r in locGlobOptions:
            for t in undersampOptions:
                print(s, r, t)
                regularLearning(s[1], s[0], outpath, t, r, knnOption)
                transferLearning(s[1], s[0], outpath, t, r, knnOption)




    #regularLearning(inpath, files, outpath)
    #transferLearning(inpath, files, outpath)
    #regularLearningLocal(inpath, files, outpath)
    #transferLearningLocal(inpath, files, outpath)
    #regularLearningUnderSample(inpath, files, outpath)
    #transferLearningUnderSample(inpath, files, outpath)
    #regularLearningLocalUndersample(inpath, files, outpath)
    #transferLearningLocalUndersample(inpath, files, outpath)

    #postprocessing(outpath, files)

def postprocessing(path, files):
    abcd_master()
    adsfdsf

driversSeat()
