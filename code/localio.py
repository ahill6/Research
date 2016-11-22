from math import floor
from collections import defaultdict
from random import sample, random, shuffle
from stats import rdivDemo
from contextlib import contextmanager
from abcd import Abcd
from copy import deepcopy
import sys, os, numpy, csv, difflib, pprint, re, itertools
#import classes


# Helper methods
def find_median(lst):
    if len(lst) % 2 == 1:
        mid = int(floor(len(lst) / 2))
        return lst[mid], lst[:mid], lst[(mid + 1):]
    else:
        mid = int(floor(len(lst) / 2))
        avg = (float(lst[mid - 1]) + float(lst[mid])) / 2.0
        return avg, lst[:mid], lst[mid:]


def del_txt():
    path = '.'
    files = [f for f in os.listdir(path) if f.endswith('.txt') & ("_" in f) & ('stats' not in f) & ('times' not in f)]
    for f in files:
        os.remove(f)

def quickcheck(basepath, filenames):
    for f in filenames:
        max = 0
        count = 0
        total = 0
        first = True
        # at the moment only works for csv files b/c lazy at moment
        with open(basepath+f+".csv", 'r') as temp_file:
            if first:
                temp_file.readline()
                first = False
            for line in temp_file:
                tmp = eval(line.rstrip("\n").split(',')[-1])
                if tmp > max:
                    max = tmp
                if tmp > 1:
                    count += 1
                total += 1
            print(f + ": " + str(count) + "/" + str(total))
            print(count/(total+0.0))


def datafile_stats(basepath, filenames):
    """
    data = numpy.ndfromtxt(
        filename,
        names = True, #  If `names` is True, the field names are read from the first valid line
        comments = '#', # Skip characters after #
        delimiter = ',', # comma separated values
        dtype = None)  # guess the dtype of each column
    """
    """
    data = []
    f = open(filename, 'r')
    headers = f.readline()
    for line in f:
        a = line.strip('\n').split(',')
        data.append(a)
    """
    for f in filenames:
        first = True
        # at the moment only works for csv files b/c lazy at moment
        with open(basepath+f+".csv", 'r') as temp_file:
            if first:
                temp_file.readline()
                first = False
            data = [[float(r) for r in line.rstrip('\n').split(',')] for line in temp_file]

            meds = numpy.median(data, axis=0)
            mins = numpy.amin(data, axis=0)
            maxs = numpy.amax(data, axis=0)
            stds = numpy.std(data, axis=0)
            cov = numpy.corrcoef(data, rowvar=0)
            evals, evects = numpy.linalg.eig(cov)
            ev = numpy.unique(evects)
            data2 = numpy.dot(data, evects)
            corr = numpy.corrcoef(data2, rowvar=0)
            with open(basepath + f + "_stats.txt", 'w') as cout:
                cout.write(f + "\n")
                cout.write("Medians:" + "\n" + str(meds) + "\n")
                cout.write("Minima:" + "\n" + str(mins) + "\n")
                cout.write("Maxima:" + "\n" + str(maxs) + "\n")
                cout.write("Std Devs:" + "\n" + str(stds) + "\n")
                cout.write("Covariance:" + "\n" + str(cov) + "\n")


def csv_reader2(filename, mini=False):
    first = True
    data = []
    with open(filename) as temp_file:
        if first:
            temp_file.readline()
            first = False
        for line in temp_file:
            tmp = [float(r) for r in line.rstrip("\n").split(',')]
            if tmp not in data:
                data.append(tmp)
        #data = [[float(r) for r in line.rstrip('\n').split(',')] for line in temp_file]
    return data
    meds = numpy.median(data, axis=0)
    mins = numpy.amin(data, axis=0)
    maxs = numpy.amax(data, axis=0)
    stds = numpy.std(data, axis=0)
    cov = numpy.corrcoef(data, rowvar=0)
    evals, evects = numpy.linalg.eig(cov)
    ev = numpy.unique(evects)
    data2 = numpy.dot(data, evects)
    corr = numpy.corrcoef(data2, rowvar=0)

    useless = set()
    # useless2 = [ind for ind, mn, md, mx, sd in enumerate(zip(mins,meds, maxs, stds)) if (md-md < sd) if (mx-md < sd)]
    # print(useless2)

    # HERE RUN THROUGH COVARIANCE AND FIGURE OUT WHO NEEDS TO GO
    limit = .999
    for j, row in enumerate(corr):
        useless.update([(i, j) for i, val in enumerate(row) if (val > limit and val != 1.0)])

    # print(useless)

    C = set()
    while not not useless:
        a = sample(useless, 1)[0]
        C.update([a[0], a[1]])
        tmp = set()
        for x in useless:
            if x == a or a[0] in x or a[1] in x:
                tmp.add(x)
        useless -= tmp

    sampledata = [data[i] for i in sorted(sample(range(len(data)), min(len(data), 1000)))]
    sampledata2 = [data2[i] for i in sorted(sample(range(len(data2)), min(len(data2), 1000)))]
    reduced = [[item[i] for i in range(len(item)) if i not in list(C)] for item in data]
    #print(C)

    if len(data) != len(data2) or len(data[0]) != len(data2[0]):
        data2, sampledata2 = "error", "error"

    if mini:
        return sampledata, sampledata2, reduced
    else:
        return data, data2, reduced

def csv_reader_remove_duplicates(filename, mini=False):
    first = True
    with open(filename) as temp_file:
        if first:
            temp_file.readline()
            first = False
        data = [[float(r) for r in line.rstrip('\n').split(',')] for line in temp_file]
        data.sort()
        data = list(d for d, _ in itertools.groupby(data))
        shuffle(data)

    meds = numpy.median(data, axis=0)
    mins = numpy.amin(data, axis=0)
    maxs = numpy.amax(data, axis=0)
    stds = numpy.std(data, axis=0)
    cov = numpy.corrcoef(data, rowvar=0)
    evals, evects = numpy.linalg.eig(cov)
    ev = numpy.unique(evects)
    data2 = numpy.dot(data, evects)
    corr = numpy.corrcoef(data2, rowvar=0)
    data2 = data2.tolist()
    useless = set()
    # useless2 = [ind for ind, mn, md, mx, sd in enumerate(zip(mins,meds, maxs, stds)) if (md-md < sd) if (mx-md < sd)]
    # print(useless2)

    # HERE RUN THROUGH COVARIANCE AND FIGURE OUT WHO NEEDS TO GO
    limit = .999
    for j, row in enumerate(corr):
        useless.update([(i, j) for i, val in enumerate(row) if (val > limit and val != 1.0)])


    # print(useless)

    C = set()
    while not not useless:
        a = sample(useless, 1)[0]
        C.update([a[0], a[1]])
        tmp = set()
        for x in useless:
            if x == a or a[0] in x or a[1] in x:
                tmp.add(x)
        useless -= tmp

    sampledata = [data[i] for i in sorted(sample(range(len(data)), min(len(data), 1000)))]
    sampledata2 = [data2[i] for i in sorted(sample(range(len(data2)), min(len(data2), 1000)))]
    reduced = [[item[i] for i in range(len(item)) if i not in list(C)] for item in data]
    #print(C)

    if len(data) != len(data2) or len(data[0]) != len(data2[0]):
        data2, sampledata2 = "error", "error"

    if mini:
        return sampledata, sampledata2, reduced
    else:
        return data, data2, reduced

def csv_reader_remove_duplicates_and_normalize(filename, mini=False):
    first = True
    with open(filename) as temp_file:
        if first:
            temp_file.readline()
            first = False
        data = [[float(r) for r in line.rstrip('\n').split(',')] for line in temp_file]
        data.sort()
        data = list(d for d, _ in itertools.groupby(data))
        shuffle(data)

    meds = numpy.median(data, axis=0)
    mins = numpy.amin(data, axis=0)
    maxs = numpy.amax(data, axis=0)
    stds = numpy.std(data, axis=0)

    data = numpy.array(data)
    x_normed = data / data.max(axis=0)
    data = x_normed.tolist()

    cov = numpy.corrcoef(data, rowvar=0)
    evals, evects = numpy.linalg.eig(cov)
    ev = numpy.unique(evects)
    data2 = numpy.dot(data, evects)
    corr = numpy.corrcoef(data2, rowvar=0)

    useless = set()
    # useless2 = [ind for ind, mn, md, mx, sd in enumerate(zip(mins,meds, maxs, stds)) if (md-md < sd) if (mx-md < sd)]
    # print(useless2)

    # HERE RUN THROUGH COVARIANCE AND FIGURE OUT WHO NEEDS TO GO
    limit = .999
    for j, row in enumerate(corr):
        useless.update([(i, j) for i, val in enumerate(row) if (val > limit and val != 1.0)])

    # print(useless)

    C = set()
    while not not useless:
        a = sample(useless, 1)[0]
        C.update([a[0], a[1]])
        tmp = set()
        for x in useless:
            if x == a or a[0] in x or a[1] in x:
                tmp.add(x)
        useless -= tmp

    sampledata = [data[i] for i in sorted(sample(range(len(data)), min(len(data), 1000)))]
    sampledata2 = [data2[i] for i in sorted(sample(range(len(data2)), min(len(data2), 1000)))]
    reduced = [[item[i] for i in range(len(item)) if i not in list(C)] for item in data]
    #print(C)

    if len(data) != len(data2) or len(data[0]) != len(data2[0]):
        data2, sampledata2 = "error", "error"

    if mini:
        return sampledata, sampledata2, reduced
    else:
        return data, data2, reduced

def pretty_print(f, d):
    for k in sorted(d.keys()):
        f.write(k, "\t\t : ", d[k])


def summarize(txt):
    path = '.'
    # files = [f for f in os.listdir(path) if f.endswith('.txt') & (txt in f) & ('summary' not in f)]
    files = [f for f in os.listdir(path) if f.endswith('.txt') & ("_" in f) & ('summary' not in f)]

    """for f in files:
        print f
        print txt in f"""

    if len(files) > 0:
        summary = open('summary.txt', 'w')
        print("Summarizing...")

        for f in files:
            # open file
            f = open(f, 'r')
            print(f.name)

            # read in something
            cin = f.readline()
            cin = cin[:-2]
            results = cin.strip("\n").split(',')

            # order that, find percentiles (min, 25%, median, 75%, max)
            results = sorted(results)
            min = results[0]
            _25 = find_median(find_median(results)[1])[0]
            median = find_median(results)[0]
            _75 = find_median(find_median(results)[2])[0]
            max = results[-1]

            # print that data to a summary file
            output = "" + str(min) + "," + str(_25) + "," + str(median) + "," + str(_75) + "," + str(max)

            summary.write(f.name + "\n" + output + "\n")

            # close data file and repeat
            f.close()
        pretty_print(summary, txt)
        summary.close()
        print("Done")

def statstree_master(f,path='.'):
    #need to figure out how to get some entry and decision numbers in here
    # make a directory tree to traverse
    #files = [f for f in os.listdir(path) if f.endswith('.txt') if ("_" in f) if ('summary' not in f) if ("stat" not in f)]
    #files = [f for f in os.listdir(path) if '.' not in f if '__' not in f]
    rootDir = 'C:\\Users\\Andrew\\PycharmProjects\\spatialtree\\1000samples\\'
    for dirName, subdirList, fileList in os.walk(rootDir):
        if '\\' in dirName and 'idea' not in dirName and '__' not in dirName:
            print(dirName+"\\")
            f(path=dirName+"\\")
    #print(files)
    # for every folder, summarize the data

def make_stats(num_entries=None, num_decisions=None, path='.'):
    path = path or ".\\1000samples\\GeneratedPaperSimulant\\k-10\\"

    # files = [f for f in os.listdir(path) if f.endswith('.txt') & (txt in f) & ('summary' not in f)]
    files = [f for f in os.listdir(path) if f.endswith('.txt') if ("_" in f) if ('summary' not in f) if
             ("stat" not in f)]
    timers = [f for f in os.listdir(path) if "_" not in f]

    if len(files) > 0:
        stats = open(path+'stats.txt', 'w')
        medians = open(path+"medians.txt", 'w')
        times = open(path+'times.txt', 'w')
        print("Making Stats...")

        for g in timers:
            h = open(path+g, 'r')
            times.write(g + "\n" + h.readline() + "\n")
            h.close()
        times.close()

        for f in sorted(files):
            # open file
            f = open(path+f, 'r')
            # print f.name

            true_pos = []
            false_pos = []
            false_neg = []
            true_neg = []
            recall = []
            pf = []
            prec = []
            acc = []
            select = []
            neg_pos = []

            # read in something
            cin = f.readline()
            cin = cin[:-2]
            results = cin.strip("\n").split(',')
            mean_recall = 0
            mean_pf = 0
            mean_prec = 0
            mean_acc = 0
            mean_select = 0
            mean_neg_pos = 0

            # calculate stats (quartiles, all others)
            for item in results:
                tp, fp, tn = item.split('-')
                tp = float(tp)
                fp = float(fp)
                tn = float(tn)
                fn = fp

                true_pos.append(tp)
                false_pos.append(fp)
                false_neg.append(fp)  # this would normally be otherwise
                true_neg.append(tn)

                recall.append((tp + 0.0) / (fn + tp))
                pf.append((fp + 0.0) / (tn + fp))
                prec.append((tp + 0.0) / (tp + fp))
                acc.append((tn + tp + 0.0) / (tn + fn + fp + tp))
                select.append((fp + tp + 0.0) / (tn + fn + fp + tp))
                neg_pos.append((tn + fp + 0.0) / (fn + tp))

                mean_recall += (tp + 0.0) / (fn + tp)
                mean_pf += (fp + 0.0) / (tn + fp)
                mean_prec += (tp + 0.0) / (tp + fp)
                mean_acc += (tn + tp + 0.0) / (tn + fn + fp + tp)
                mean_select += (fp + tp + 0.0) / (tn + fn + fp + tp)
                mean_neg_pos += (tn + fp + 0.0) / (fn + tp)

            # find iqr
            recall = sorted(recall)
            min = recall[0]
            _25 = find_median(find_median(recall)[1])[0]
            median = find_median(recall)[0]
            _75 = find_median(find_median(recall)[2])[0]
            max = recall[-1]

            # print that data to a summary file
            total = len(recall)
            stats.write(f.name + "\n")
            medians.write(f.name + "\n")
            medians.write(str(min) + '\t' + str(_25) + "\t" + str(median) + "\t" + str(_75) + "\t" + str(max) + "\n")
            # summary.write(" ".join(str(_25),str(median),str(_75)) + "\n")
            stats.write("rec " + str(mean_recall / total) + "\n")
            stats.write("pf " + str(mean_pf / total) + "\n")
            stats.write("prec " + str(mean_prec / total) + "\n")
            stats.write("acc " + str(mean_acc / total) + "\n")
            stats.write("select " + str(mean_select / total) + "\n")
            stats.write("neg/pos " + str(mean_neg_pos / total) + "\n")

            # close data file and repeat
            f.close()
        # pretty_print(summary, txt)
        stats.close()
        medians.close()
        print("Done")

        table_please(infile=stats.name, num_decisions=num_decisions,
                     num_entries=num_entries)  # need some way to pass num_entries and num_decisions to table_please
        table_please(infile=medians.name, num_decisions=num_decisions,
                     num_entries=num_entries)  # need some way to pass num_entries and num_decisions to table_please

def prep_stats(path='.'):
    path = path or ".\\1000samples\\GeneratedPaperSimulant\\k-10\\"

    # files = [f for f in os.listdir(path) if f.endswith('.txt') & (txt in f) & ('summary' not in f)]
    files = [f for f in os.listdir(path) if f.endswith('.txt') if
             any(f.split('.')[0] in s for s in ['kd', 'pca', '2-means', 'rp', 'where', 'random', 'spectral', 'entropic', 'entropic2'])]

    if len(files) > 0:
        recallstats = open(path+'recall.txt', 'w')
        precstats = open(path + 'prec.txt', 'w')
        accstats = open(path + 'acc.txt', 'w')
        selectstats = open(path + 'select.txt', 'w')
        print("Prepping Stats...")

        """
        for g in timers:
            h = open(path+g, 'r')
            times.write(g + "\n" + h.readline() + "\n")
            h.close()
        times.close()
        """

        for f in sorted(files):
            # open file
            f = open(path+f, 'r')
            # print f.name

            true_pos = []
            false_pos = []
            false_neg = []
            true_neg = []
            recall = []
            prec = []
            acc = []
            select = []

            # read in something
            cin = f.readline()
            cin = cin[:-2]
            results = cin.strip("\n").split(',')

            # calculate stats (quartiles, all others)
            for item in results:
                tp, fp, tn = item.split('-')
                tp = float(tp)
                fp = float(fp)
                tn = float(tn)
                fn = fp

                true_pos.append(tp)
                false_pos.append(fp)
                false_neg.append(fp)  # this would normally be otherwise
                true_neg.append(tn)

                recall.append(str((tp + 0.0) / (fn + tp)))
                prec.append(str((tp + 0.0) / (tp + fp)))
                acc.append(str((tn + tp + 0.0) / (tn + fn + fp + tp)))
                select.append(str((fp + tp + 0.0) / (tn + fn + fp + tp)))

            # print that data to a summary file
            total = len(recall)
            name = os.path.split(f.name)[1].split('_')[0].split('.')[0]
            recallstats.write(name + ",") # this needs to be 1 word (3 characters?) - and not all the nonsense now in name
            precstats.write(name + ",")  # this needs to be 1 word (3 characters?) - and not all the nonsense now in name
            accstats.write(name + ",")  # this needs to be 1 word (3 characters?) - and not all the nonsense now in name
            selectstats.write(name + ",")  # this needs to be 1 word (3 characters?) - and not all the nonsense now in name
            recallstats.write(",".join(recall) + "\n")
            precstats.write(",".join(prec) + "\n")
            accstats.write(",".join(acc) + "\n")
            selectstats.write(",".join(select) + "\n")

            # close data file and repeat
            f.close()
        # pretty_print(summary, txt)
        recallstats.close()
        precstats.close()
        accstats.close()
        selectstats.close()
        print("Done")

        # Call Menzies' magic stats maker


def strip_csv(file, col, inpath, outpath, times):
    f2 = file.split('.')[0]
    cout = open(outpath + f2 + '2.csv', 'w')
    with open(inpath + file, 'r') as f:
        for line in f:
            results = line.strip("\n").split(',')
            for t in range(times):
                del results[col]
            cout.write(','.join([r for r in results]) + "\n")
    cout.close()


def table_please(num_entries=None, num_decisions=None, infile=None):
    results = dict()

    path =  os.path.split(infile)[0]

    def nested_set(dic, keys, value):
        for key in keys[:-1]:
            dic = dic.setdefault(key, {})
        dic[keys[-1]] = value
        """
        if not isinstance(value, (list, tuple)):
            dic[keys[-1]] = value
        else:
            dic[keys[-1]].append(value)
        """

    def medians(cin):
        a, x, y, z, b = cin.split()
        e = [y, eval(z) - eval(x)]
        nested_set(results, [s, d, m], e)

    def stats(cin):
        x, y = cin.split()
        nested_set(results, [s, d, x, m], y)

    def graphing(cin):
        x, y, z = cin.split()
        e = [y, eval(z) - eval(x)]
        nested_set(results, [s, d, m], e)

    file = infile or 'summary.txt'
    if 'median' in infile:
        process = medians
        out = path+'\\medians.ods'
    elif 'stat' in infile:
        process = stats
        out = path+'\\stats.ods'
    else:
        raise ValueError('Invalid filename.', infile)

    f = open(file, 'r')
    lst = []
    m = -1
    s = -1
    d = -1

    # read in the data and put into a multi-D dictionary that will allow for pretty printing
    for cin in f:
        try:
            m, s, d = cin.strip('.txt\n').split('_')
        except:
            process(cin)

    # data is in multidimensional dictionary.  Send to printer (writer)
    write_table_from_dict(results, num_entries, num_decisions, out)


def write_table_from_dict(dictionary, entries, decisions, outfile):
    def medians():
        spl = list(dictionary.keys())[0]
        dpt = list(dictionary[spl].keys())[0]
        tmp = list(dictionary[spl][dpt].keys())
        cout.write(',' + ',,'.join(tmp))
        header = 'median,iqr,' * len(tmp)
        cout.write("\nSpill-depth," + header + "\n")
        for spill in dictionary:
            for depth in dictionary[spill]:
                tmp = [str(x[0]) + ',' + str(x[1]) for x in dictionary[spill][depth].values()]
                cout.write(str(spill) + ' ' + str(depth) + ',' + ','.join(tmp) + "\n")

    def stats():
        for spill in dictionary:
            for depth in dictionary[spill]:
                ind = list(dictionary[spill][depth].keys())[0]
                tmp = list(dictionary[spill][depth][ind].keys())
                cout.write(str(spill) + ' ' + str(depth) + ',' + ','.join(tmp) + "\n")
                for statistic in dictionary[spill][depth]:
                    tmp = list(dictionary[spill][depth][statistic].values())
                    cout.write(statistic + ',' + ','.join(tmp) + "\n")
                newline = ',' * (len(dictionary[spill][depth][statistic]) + 1) + "\n"
                cout.write(newline * 2)

    cout = open(outfile, 'w')

    # this is a terrible kludge.  Fix eventually.
    if 'median' in outfile:
        printprocess = medians
    else:
        printprocess = stats
    # cout = open('table.ods','w')

    # print number of entries and decisions for reference/interest
    cout.write("" + str(entries) + ' entries , ' + str(decisions) + " decisions\n")
    # print the whole dictionary in a particular order/way (using the keys as headers)
    printprocess()
    #del_txt()


def graph_data():
    path = '.'
    files = [f for f in os.listdir(path) if f.endswith('.txt') if ("_" in f) if ('summary' not in f) if
             ("stat" not in f)]
    # timers = [f for f in os.listdir(path) if "." not in f]

    if len(files) > 0:
        chart = open("chart_data.txt", 'w')
        print("Prepping Data for Graphs...")

        """
        for g in timers:
            h = open(g, 'r')
            times.write(g + "\n" + h.readline()+"\n")
            h.close()
        times.close()
        """

        for f in sorted(files):
            # open file
            g = open(f, 'r')
            # print f.name

            true_pos = []
            false_pos = []
            false_neg = []
            true_neg = []
            recall = []
            pf = []
            prec = []
            acc = []
            select = []
            neg_pos = []

            # read in something
            cin = f.readline()
            cin = cin[:-2]
            results = cin.strip("\n").split(',')
            mean_recall = 0
            mean_pf = 0
            mean_prec = 0
            mean_acc = 0
            mean_select = 0
            mean_neg_pos = 0

            # calculate stats (quartiles, all others)
            for item in results:
                tp, fp, tn = item.split('-')
                tp = float(tp)
                fp = float(fp)
                tn = float(tn)
                fn = fp

                true_pos.append(tp)
                false_pos.append(fp)
                false_neg.append(fp)  # this would normally be otherwise
                true_neg.append(tn)

                recall.append((tp + 0.0) / (fn + tp))
                pf.append((fp + 0.0) / (tn + fp))
                prec.append((tp + 0.0) / (tp + fp))
                acc.append((tn + tp + 0.0) / (tn + fn + fp + tp))
                select.append((fp + tp + 0.0) / (tn + fn + fp + tp))
                neg_pos.append((tn + fp + 0.0) / (fn + tp))

                mean_recall += (tp + 0.0) / (fn + tp)
                mean_pf += (fp + 0.0) / (tn + fp)
                mean_prec += (tp + 0.0) / (tp + fp)
                mean_acc += (tn + tp + 0.0) / (tn + fn + fp + tp)
                mean_select += (fp + tp + 0.0) / (tn + fn + fp + tp)
                mean_neg_pos += (tn + fp + 0.0) / (fn + tp)

            # find iqr
            recall = sorted(recall)
            min = recall[0]
            _25 = find_median(find_median(recall)[1])[0]
            median = find_median(recall)[0]
            _75 = find_median(find_median(recall)[2])[0]
            max = recall[-1]

            # print that data to a summary file
            total = len(recall)
            chart.write(f.name + "\n")
            chart.write(str(min) + '\t' + str(_25) + "\t" + str(median) + "\t" + str(_75) + "\t" + str(max) + "\n")
            # summary.write(" ".join(str(_25),str(median),str(_75)) + "\n")
            chart.write("rec " + str(mean_recall / total) + "\n")
            chart.write("pf " + str(mean_pf / total) + "\n")
            chart.write("prec " + str(mean_prec / total) + "\n")
            chart.write("acc " + str(mean_acc / total) + "\n")
            chart.write("select " + str(mean_select / total) + "\n")
            chart.write("neg/pos " + str(mean_neg_pos / total) + "\n")

            # close data file and repeat
            f.close()
        # pretty_print(summary, txt)
        chart.close()
        print("Done")


def make_me_a_table(num_entries, num_decisions, infile=None):
    rob = ResultFactory('name')

    file = infile or 'summary.txt'
    f = open(file, 'r')
    lst = []

    # read in the data and put into separate Results objects
    with cin as f:
        try:
            m, s, d = cin.strip('.txt').split(
                '_')  # strip removes all instances of everything passed, not exact matches.
            if hasattr(bob, 'method'):
                bob.add_values(lst)
                lst = []
            bob = rob.make_result(method=m, spill=s, depth=d)
        except:
            x, y = cin.split()
            lst.append(dict(name=x, value=y))

    # data is in Result objects in ResultFactory.  Send to printer (writer)
    write_table(rob, num_entries, num_decisions)


def write_table(data, num_entries, num_decisions):
    # methods = # num of distinct methods?  Do I need to write a table, then go back and write the header?
    # put everything into a dictionary, then pull it out in a comprehensible order
    for x in range(2 * len(methods) + 1):
        out.write(',')
    cout = "\n,data = " + str(num_entries) + ', ' + str(num_decisions) + "decisions"
    for x in range(2 * len(methods)):
        cout += ','
    out.write(cout + "\n,,")
    for x in methods:
        out.write(x + ',,')
    out.write(',,\n,Spill-depth,')
    cout = 'median,iqr,' * len(methods)
    out.write(cout + ',\n')
    for a in spill:
        for b in tree_depth:
            out.write(',' + str(b) + ' ' + str(a) + ',')
            for t in trees:  # need to guarantee these go in order
                out.write(median[t] + ',' + iqr[t] + ',')
            out.write(',\n')
    cout = [',' for i in 2 * len(methods) + 1].tostring
    out.write(cout)
    out.write(cout)
    out.write(cout)
    """
    for a in spill:
        for b in tree_depth: # write this to do all or just 5, 13?
            out.write(',' + str(b) + ' ' + str(a) + ',')
            for m in methods:
                out.write(m + ',')
            out.write(',,\n,')
            outpf = [i for i in pf if a==a if b==b]tostring #??
            for m in methods:
                # how to put the label at the beginning of the line, then loop through the method values?
                out.write('pf:,' + pf[a][b][m] + ',') # how to reference?
                out.write(pf[a][b][m] + ',') # how to reference?
                out.write(pf[a][b][m] + ',') # how to reference?
                out.write(pf[a][b][m] + ',') # how to reference?
    """

def sort_files():
    rootDir = 'C:\\Users\\Andrew\\PycharmProjects\\spatialtree\\1000samples\\unconstrained\\'
    lst = []
    for dirName, subdirList, fileList in os.walk(rootDir):
        if '\\' in dirName and 'idea' not in dirName and '__' not in dirName:
            lst.append(dirName+"\\")

    for item in lst:
        files = [f for f in os.listdir(item) if f.endswith('.txt') if ("_" in f) ]
        for f in files:
            # read off the next spill/depth
            if f.split('.')[1] == 'txt':
                tp = f.split('.')[0].split('_')
                print(tp)
                if tp[0] == '0':
                    tmp = "0_"
                else:
                    tmp = tp[1] + '_' + tp[2]
                newdir = os.path.split(item)[0] + '\\' + tmp

            else:
                newdir = os.path.split(item)[0] + '\\' + f.split('.')[1] + '\\'
            fnew = newdir + f.split('_')[0] + '.txt'
            os.makedirs(os.path.dirname(newdir), exist_ok=True)
            os.rename(item + f, fnew)

# strip_csv('arc.csv',-1)
def menzies_stats(path):
    for method in ['pd', 'prec', 'pf', 'acc', 'f', 'g']:
        fname = method + ".csv"
        d = dict()

        #input current file data
        if not os.path.isfile(path + fname):
            print("No file " + fname + " in " + path)
            pass
        with open(path + fname, 'r') as file:
            for cin in file:
                results = cin.strip("\n").split(',')
                label = results[0]
                results = results[1:]
                d[label] = [eval(x) for x in results]
        #make stats
        if '' in d:
            del d['']

        data = []
        for k in d.keys():
            inp = [str(k)]
            inp.extend(list(d[k]))
            data.append(inp)
        with open(path + method +"_stats.txt", 'w') as out:
            rdivDemo(data, out)
        print("done")

def menzies_time_stats(path, infiles):
    d = dict()
    for zz in infiles:
        for fname in zz:
            #input current file data
            if not os.path.isfile(path + fname):
                print("No file " + fname + " in " + path)
                pass
            with open(path + fname, 'r') as file:
                for cin in file:
                    results = cin.strip("\n").split(',')
                    label = '.'.join(fname.strip("\n").split('.')[:2])
                    d[label] = [eval(x) for x in results if x != '']
                #make stats

        data = []
        for k in d.keys():
            inp = [str(k)]
            inp.extend(list(d[k]))
            data.append(inp)
        with open(path + "time_stats.txt", 'w') as out:
            rdivDemo(data, out)
    print("done")

def menzies_stats_mod(path):
    for method in ['pd', 'prec', 'pf', 'acc', 'f', 'g']:
        #fname = method + ".csv"
        fname = method + ".txt"
        d = dict()

        #input current file data
        if not os.path.isfile(path + fname):
            print("No file " + fname + " in " + path)
            pass
        with open(path + fname, 'r') as file:
            for cin in file:
                results = cin.strip("\n").split(',')
                results = results[:-2]
                label = results[0]
                results = results[1:]
                d[label] = [eval(x) for x in results]
        #make stats
        del d['']

        data = []
        for k in d.keys():
            inp = [str(k)]
            inp.extend(list(d[k]))
            data.append(inp)
        with open(path + method +"_stats.txt", 'w') as out:
            rdivDemo(data, out)
        print("done")

def summary_master(path, f):
    #need to figure out how to get some entry and decision numbers in here
    # make a directory tree to traverse
    #files = [f for f in os.listdir(path) if f.endswith('.txt') if ("_" in f) if ('summary' not in f) if ("stat" not in f)]
    #files = [f for f in os.listdir(path) if '.' not in f if '__' not in f]
    #rootDir = path or 'C:\\Users\\Andrew\\PycharmProjects\\spatialtree\\1000samples\\'
    out = open(path+"data.txt", 'w')
    for dirName, subdirList, fileList in os.walk(path):
        if '\\' in dirName and 'idea' not in dirName and '__' not in dirName and 'compilation' not in dirName and 'unconstrained' not in dirName \
                and 'holding' not in dirName:
            print(dirName)
            f(path=dirName+"\\", out=out)
    out.close()

def directory_master(path, func, want=None, dontwant=None, **kwargs):
    # make a directory tree to traverse
    for dirName, subdirList, fileList in os.walk(path):
        infiles = [x for x in fileList if '.txt' in x if 'idea' not in x if 'error' not in x if (want == None or want in x) if dontwant not in x]
        print(dirName)
        func(path+"\\", infiles, **kwargs)

def directory_master2(path, func, want=None, dontwant=None, **kwargs):
    # make a directory tree to traverse
    for dirName, subdirList, fileList in os.walk(path):
        infiles = [x for x in fileList if '.txt' in x if 'idea' not in x if 'error' not in x if (want == None or want in x) if 'cart' in x if dontwant not in x]
        print(dirName)
        func(path+"\\", infiles, **kwargs)

def summarizeData(dir, *args):
    print("Under Construction")
    #read everything sent in, add to files in each arg category

#!!!!!!requires a contextmanager to open/close the files before/after this


#@contextmanager
def filemanager(dir, fname):
    print("Under Construction")



def abcd_master(dir, wanted):
    #path = os.path.dirname(dir)
    path = dir + "\\" + wanted

    if not os.path.exists(path):
        os.makedirs(path)
    path += "\\"

    # open files for all stats
    acc = open(path + "acc.csv", 'w')
    pd = open(path + "pd.csv", 'w')
    pf = open(path + "pf.csv", 'w')
    prec = open(path + "prec.csv", 'w')
    f = open(path + "f.csv", 'w')
    g = open(path + "g.csv", 'w')
    errorout = open(path + "abcdErrors.txt", 'w')

    directory_master(dir, abcd_caller, want = wanted, dontwant="times", acc=acc, pd=pd, pf=pf, prec=prec, f=f, g=g, errorout=errorout)
    #directory_master(dir, abcd_cart, want=wanted, dontwant="times", acc=acc, pd=pd, pf=pf, prec=prec, f=f, g=g,
                     #errorout=errorout)


    #with filemanager(dir, "times"):
        #directory_master(dir, summarizeData, want="times")

    #come back at some point and make these a context manager
    acc.close()
    pd.close()
    pf.close()
    prec.close()
    f.close()
    g.close()
    errorout.close()


def abcd_caller(dir, files, **kwargs):
    log = None
    oldlabel = None
    files.sort(key=lambda x: x.replace(".txt", "").strip().split('-')[1])
    #files.sort(key=lambda x: x.replace(".txt", "").strip())

    i = 0

    for f in files:
        #read in file
        label = f.replace(".txt", "").split('-')[1]
        #label = f.replace(".txt","")
        log = None
        if os.stat(dir + f).st_size != 0:
            with open(dir + f, 'r') as cin:
                print(f)
                for line in cin:
                    words = re.sub(r"[\n\r]", "", line).split(",")
                    one, two = words[0], words[1]
                    if log:
                        log(one, two)
                    else:
                        log = Abcd(one, two)

                    # put the pieces in each file
                if label != oldlabel:
                    try:
                        kwargs['acc'].write("\n" + label + "," + str(numpy.dot(numpy.array([s.acc for x, s in sorted(log.scores().items())]), [log.yes, log.no]) / (log.yes + log.no)))
                        kwargs['pd'].write("\n" + label + "," +  str(numpy.dot(numpy.array([s.pd for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['pf'].write("\n" + label + "," +  str(numpy.dot(numpy.array([s.pf for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['prec'].write("\n" + label + "," +  str(numpy.dot(numpy.array([s.prec for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['f'].write("\n" + label + "," +  str(numpy.dot(numpy.array([s.f for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['g'].write("\n" + label + "," +  str(numpy.dot(numpy.array([s.g for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        oldlabel = deepcopy(label)
                    except ValueError:
                        kwargs['errorout'].write("ERROR WITH " + f + "\n")
                        pass
                else:
                    try:
                        kwargs['acc'].write("," + str(numpy.dot(numpy.array([s.acc for x, s in sorted(log.scores().items())]),[log.yes, log.no]) / (log.yes + log.no)))
                        kwargs['pd'].write("," +  str(numpy.dot(numpy.array([s.pd for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['pf'].write("," +  str(numpy.dot(numpy.array([s.pf for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['prec'].write("," +  str(numpy.dot(numpy.array([s.prec for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['f'].write("," +  str(numpy.dot(numpy.array([s.f for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['g'].write("," +  str(numpy.dot(numpy.array([s.g for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                    except ValueError:
                        kwargs['errorout'].write("ERROR WITH " + f + "\n")
                        pass

def write_all(path, out):
    #open all files with names in X
    dirParts = path.split("\\")
    zeroup = dirParts[-2]
    oneup = dirParts[-3]
    twoup = dirParts[-4]
    bigfilename = zeroup + "-" + oneup + "-" + twoup
    files = [f for f in os.listdir(path+"\\") if 'recall.txt' in f]
    for f in files:
        with open(path+"\\"+f, 'r') as file:
            for cin in file:
                results = cin.strip("\n").split(',')
                label = results[0]+bigfilename
                results = results[1:]
                out.write(label + ",")
                out.write(",".join(results)+"\n")

    # for each, write a name, then put all data in this file
    #difflib.ndiff(, b)
"""
def combine_summary_files_and_stats(path, out):
    statfiles = ['acc.csv', 'f.csv', 'g.csv', 'prec.csv', 'pd.csv', 'pf.csv']
    for f in statfiles:
        first = None
    outfile = f.split('.')[0] + ".txt"
	out = open(path + "\\" + outfile, 'w')

    with open(path + "\\" + f, 'r') as file:
            for cin in file:
                line = cin.strip("\n").split(',')
                label = line[0]
                data = line[1]
                print(first, label, data)
        print(outfile, out)
		print(first == label)
		sys.exit(0)
                if first == label:
                    out.write("\n" + label + "," + data)
                    first = deepcopy(label)
                else:
                    out.write("," + data)
	out.close()
"""


def abcd_cart(dir, files, **kwargs):
    log = None
    oldlabel = None
    #files.sort(key=lambda x: x.replace(".txt", "").strip().split('-')[1])
    files.sort(key=lambda x: x.replace(".txt", "").strip())

    i = 0

    for f in files:
        #read in file
        #label = f.replace(".txt", "").split('-')[1]
        label = f.replace(".txt","")
        log = None
        if os.stat(dir + f).st_size != 0:
            with open(dir + f, 'r') as cin:
                print(f)
                for line in cin:
                    words = re.sub(r"[\n\r]", "", line).split(",")
                    one, two = words[0], words[1]
                    if log:
                        log(one, two)
                    else:
                        log = Abcd(one, two)

                    # put the pieces in each file
                if label != oldlabel:
                    try:
                        kwargs['acc'].write("\n" + label + "," + str(numpy.dot(numpy.array([s.acc for x, s in sorted(log.scores().items())]), [log.yes, log.no]) / (log.yes + log.no)))
                        kwargs['pd'].write("\n" + label + "," +  str(numpy.dot(numpy.array([s.pd for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['pf'].write("\n" + label + "," +  str(numpy.dot(numpy.array([s.pf for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['prec'].write("\n" + label + "," +  str(numpy.dot(numpy.array([s.prec for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['f'].write("\n" + label + "," +  str(numpy.dot(numpy.array([s.f for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['g'].write("\n" + label + "," +  str(numpy.dot(numpy.array([s.g for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        oldlabel = deepcopy(label)
                    except ValueError:
                        kwargs['errorout'].write("ERROR WITH " + f + "\n")
                        pass
                else:
                    try:
                        kwargs['acc'].write("," + str(numpy.dot(numpy.array([s.acc for x, s in sorted(log.scores().items())]),[log.yes, log.no]) / (log.yes + log.no)))
                        kwargs['pd'].write("," +  str(numpy.dot(numpy.array([s.pd for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['pf'].write("," +  str(numpy.dot(numpy.array([s.pf for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['prec'].write("," +  str(numpy.dot(numpy.array([s.prec for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['f'].write("," +  str(numpy.dot(numpy.array([s.f for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['g'].write("," +  str(numpy.dot(numpy.array([s.g for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                    except ValueError:
                        kwargs['errorout'].write("ERROR WITH " + f + "\n")
                        pass

def collator(dir, wanted, name):
    # INPUT:    dir, "wanted" files, name for summary file
    # OUTPUT:   none, the files have been put into a summary file of passed name
    files = [f for f in os.listdir(dir) if wanted in f]
    cout = open(dir + name, 'w')

    for f in files:
        print(f)
        with open(dir + f, 'r') as temp_file:
            data = [[r for r in line.strip("\n").split(',') if r] for line in temp_file]
            if data:
                data = data[0]
            else:
                pass
            cout.write(f.replace(".txt.","").strip() + ",")
            cout.write(','.join(data) + "\n")
    cout.close()

def stats_on_stats(fileroot, datasets, knns, spills, depths, methods):
    # read in stats summaries one-at-a-time, keep a tally of how often each shows up

    # helper methods
    # adds values to the dictionaries that are counters
    def update(input):
        parsin = input[1]
        parsebit = parsin.split('_')
        dataset = parsebit[4]
        dep = allowed_depths[int(parsebit[3])] if int(parsebit[3]) < 5 else int(parsebit[3])
        index = "_".join(parsebit[0:4])+"_"+parsebit[5]
        try:
            bigsummary[stattype][index] += 1
        except KeyError:
            nested_set(bigsummary, [stattype, index], 1)

        try:
            knn[stattype][dataset][int(parsebit[1])] += 1
        except KeyError:
            nested_set(knn, [stattype, dataset, int(parsebit[1])], 1)
        try:
            spill[stattype][dataset][eval(parsebit[2])] += 1
        except KeyError:
            nested_set(spill, [stattype, dataset, eval(parsebit[2])], 1)
        try:
            depth[stattype][dataset][dep] += 1
        except KeyError:
            nested_set(depth, [stattype, dataset, dep], 1)
        try:
            method[stattype][dataset][parsebit[5]] += 1
        except KeyError:
            nested_set(method, [stattype, dataset, parsebit[5]], 1)

    # allows inserted to nested dictionary
    def nested_set(dic, keys, value):
        for key in keys[:-1]:
            dic = dic.setdefault(key, {})
        dic[keys[-1]] = value

    files = []
    allowed_depths = [5, 7, 9, 11, 13]
    stat_types = ['pd', 'pf', 'acc', 'prec', 'f', 'g']
    # NEED: (table to look up high/low)
    for x in knns:
        for g in datasets:
            #path = fileroot + "k-" + str(x) + "\\" + g + "\\"
            path = fileroot + "\\" + g + "\\"
            files.extend([path+n for n in os.listdir(path) if 'stats' in n ])
    pd = -0.0
    pf = -0.0
    acc = -0.0
    prec = -0.0
    f = -0.0
    g = -0.0

    bigsummary = dict()
    knn = dict()
    spill = dict()
    depth = dict()
    method = dict()

    #lowhigh = {'acc': 'high', 'f': 'high', 'g':'high', 'pd': 'high', 'pf': 'low', 'prec': 'high'}

    for fi in files:
        print(fi)
        right_num = 0
        if 'pf_stats' not in fi:
            with open(fi, 'r') as f:
                f.readline()
                f.readline()
                for cin in f:
                    cur_num = int(cin.strip().split(",")[0])
                    if cur_num > right_num:
                        right_num = cur_num

        # open file
        f = open(fi, 'r')
        #print(fi)
        stattype = fi.split('\\')[-1].split('_')[0]

        # first two lines are header

        if 'pf_stats' in fi: # kludge for now
            high = False
        else:
            high = True

        # read in something
        with open(fi, 'r') as f:
            f.readline()
            f.readline()
            for cin in f:
                line = cin.strip().split(',')
                check = int(line[0].strip())
                if not high and check == 1:
                #if not high and (check == 1 or check==2):
                    #print("1")
                    update(line)
                elif high and check == right_num:
                #elif high and (check == right_num or check == right_num-1):
                    #print("2")
                    update(line)
                elif check != 1 and not high:
                    # elif check != 1 and check != 2 and not high:
                    #print("3")
                    continue
                else:
                    pass

    # make folder if not already exists
    if not os.path.exists(fileroot + "Summaries"):
        os.makedirs(fileroot+"Summaries")

    with open(fileroot + "Summaries\\spill.txt", 'w') as outfile:
        pprint.pprint(spill, outfile)
    with open(fileroot + "Summaries\\method.txt", 'w') as outfile:
        pprint.pprint(method, outfile)
    with open(fileroot + "Summaries\\depth.txt", 'w') as outfile:
        pprint.pprint(depth, outfile)
    with open(fileroot + "Summaries\\bigSummary.txt", 'w') as outfile:
        pprint.pprint(bigsummary, outfile)


    order = ['pd', 'prec', 'acc', 'pf', 'f', 'g']


    compilation = dict()
    compilation2 = dict()
    compilationtmp = dict()
    for ord in order:
        with open(fileroot+'Summaries\\' + ord + '.txt', 'w') as cout:
            #cout.write(ord +',')
            compilationtmp.clear()

            # method[stattype][dataset][method]
            for second in method[ord]:
                for third in method[ord][second]:
                    for repetitions in range(int(method[ord][second][third])):
                        if third in compilation:
                            compilation[third] += 1
                        else:
                            compilation[third] = 1
                        if third in compilationtmp:
                            compilationtmp[third] += 1
                        else:
                            compilationtmp[third] = 1
                        #cout.write(str(third).strip() + ',')
                    #cout.write("\n")

            # spill[stattype][dataset][spill]
            for second in spill[ord]:
                for third in spill[ord][second]:
                    for repetitions in range(int(spill[ord][second][third])):
                        if third in compilation:
                            compilation[third] += 1
                        else:
                            compilation[third] = 1
                        if third in compilationtmp:
                            compilationtmp[third] += 1
                        else:
                            compilationtmp[third] = 1
                        #cout.write(str(third).strip() + ',')
                    #cout.write("\n")

            # depth[stattype][dataset][depth]
            for second in depth[ord]:
                for third in depth[ord][second]:
                    for repetitions in range(int(depth[ord][second][third])):
                        if third in compilation:
                            compilation[third] += 1
                        else:
                            compilation[third] = 1
                        if third in compilationtmp:
                            compilationtmp[third] += 1
                        else:
                            compilationtmp[third] = 1
                        #cout.write(str(third).strip() + ',')
                    #cout.write("\n")

            # method[stattype][dataset][method]
            for third in bigsummary[ord]:
                for repetitions in range(int(bigsummary[ord][third])):
                    if third in compilation:
                        compilation[third] += 1
                    else:
                        compilation[third] = 1
                    if third in compilationtmp:
                        compilationtmp[third] += 1
                    else:
                        compilationtmp[third] = 1
                    #cout.write(str(third).strip() + ',')
                #cout.write("\n")
            pprint.pprint(compilationtmp, cout)

    with open(fileroot + "Summaries\\summary.txt", 'w') as outfile:
        pprint.pprint(compilation, outfile)
    with open(fileroot + "Summaries\\otherSummary.txt", 'w') as outfile:
        pprint.pprint(compilation2, outfile)

def stats_on_stats2(fileroot, datasets, knns, spills, depths, methods):
    # read in stats summaries one-at-a-time, keep a tally of how often each shows up

    # helper methods
    # adds values to the dictionaries that are counters
    def update(input):
        parsin = input[1]
        parsebit = parsin.split('_')
        dataset = parsebit[4]
        dep = allowed_depths[int(parsebit[3])] if int(parsebit[3]) < 5 else int(parsebit[3])
        index = "_".join(parsebit[0:4])+"_"+parsebit[5]
        med = eval(input[2].strip())/100
        iqr = eval(input[3].split('(')[0].strip())/100

        try:
            bigsummary[stattype][index] += 1
        except KeyError:
            nested_set(bigsummary, [stattype, index], 1)
        try:
            whichdatasets[stattype][dataset][0] += 1
            whichdatasets[stattype][dataset][1] += med
            whichdatasets[stattype][dataset][2] += iqr
        except KeyError:
            nested_set(whichdatasets, [stattype, dataset, 0], 1)
            nested_set(whichdatasets, [stattype, dataset, 1], med)
            nested_set(whichdatasets, [stattype, dataset, 2], iqr)

        try:
            knn[stattype][dataset][int(parsebit[1])] += 1
        except KeyError:
            nested_set(knn, [stattype, dataset, int(parsebit[1])], 1)
        try:
            spill[stattype][dataset][eval(parsebit[2])]['count'] += 1
            spill[stattype][dataset][eval(parsebit[2])]['medians'] += med
            spill[stattype][dataset][eval(parsebit[2])]['iqr'] += iqr
        except KeyError:
            nested_set(spill, [stattype, dataset, eval(parsebit[2]), 'count'], 1)
            nested_set(spill, [stattype, dataset, eval(parsebit[2]), 'medians'], med)
            nested_set(spill, [stattype, dataset, eval(parsebit[2]), 'iqr'], iqr)
        try:
            depth[stattype][dataset][dep]['count'] += 1
            depth[stattype][dataset][dep]['medians'] += med
            depth[stattype][dataset][dep]['iqr'] += iqr
        except KeyError:
            nested_set(depth, [stattype, dataset, dep, 'count'], 1)
            nested_set(depth, [stattype, dataset, dep, 'medians'], med)
            nested_set(depth, [stattype, dataset, dep, 'iqr'], iqr)
        try:
            method[stattype][dataset][parsebit[5]]['count'] += 1
            method[stattype][dataset][parsebit[5]]['medians'] += med
            method[stattype][dataset][parsebit[5]]['iqr'] += iqr
        except KeyError:
            nested_set(method, [stattype, dataset, parsebit[5], 'count'], 1)
            nested_set(method, [stattype, dataset, parsebit[5], 'medians'], med)
            nested_set(method, [stattype, dataset, parsebit[5], 'iqr'], iqr)

    # allows inserted to nested dictionary
    def nested_set(dic, keys, value):
        for key in keys[:-1]:
            dic = dic.setdefault(key, {})
        dic[keys[-1]] = value

    files = []
    allowed_depths = [5, 7, 9, 11, 13]
    stat_types = ['pd', 'pf', 'acc', 'prec', 'f', 'g']
    # NEED: (table to look up high/low)
    for x in knns:
        for g in datasets:
            #path = fileroot + "k-" + str(x) + "\\" + g + "\\"
            path = fileroot + "\\" + g + "\\"
            files.extend([path+n for n in os.listdir(path) if 'stats' in n ])

    bigsummary = dict()
    knn = dict()
    spill = dict()
    depth = dict()
    method = dict()
    whichdatasets = dict()
    summary = dict()
    #lowhigh = {'acc': 'high', 'f': 'high', 'g':'high', 'pd': 'high', 'pf': 'low', 'prec': 'high'}

    for fi in files:
        print(fi)
        right_num = 0
        if 'pf_stats' not in fi:
            with open(fi, 'r') as f:
                f.readline()
                f.readline()
                for cin in f:
                    cur_num = int(cin.strip().split(",")[0])
                    if cur_num > right_num:
                        right_num = cur_num

        # open file
        f = open(fi, 'r')
        #print(fi)
        stattype = fi.split('\\')[-1].split('_')[0]

        # first two lines are header

        if 'pf_stats' in fi: # kludge for now
            high = False
        else:
            high = True

        # read in something
        with open(fi, 'r') as f:
            f.readline()
            f.readline()
            for cin in f:
                line = cin.strip().split(',')
                check = int(line[0].strip())
                if not high and check == 1:
                #if not high and (check == 1 or check==2):
                    #print("1")
                    update(line)
                elif high and check == right_num:
                #elif high and (check == right_num or check == right_num-1):
                    #print("2")
                    update(line)
                elif check != 1 and not high:
                    # elif check != 1 and check != 2 and not high:
                    #print("3")
                    continue
                else:
                    pass

    # make folder if not already exists
    if not os.path.exists(fileroot + "Summaries"):
        os.makedirs(fileroot+"Summaries")

    for one in whichdatasets:
        for two in whichdatasets[one]:
            if one in summary:
                summary[one][0] += whichdatasets[one][two][0]
                summary[one][1] += whichdatasets[one][two][1]
                summary[one][2] += whichdatasets[one][two][2]
            else:
                nested_set(summary, [one, 0], whichdatasets[one][two][0])
                nested_set(summary, [one, 1], whichdatasets[one][two][1])
                nested_set(summary, [one, 2], whichdatasets[one][two][2])
            whichdatasets[one][two][1] /= (whichdatasets[one][two][0]+0.0)
            whichdatasets[one][two][2] /= (whichdatasets[one][two][0]+0.0)

    for one in summary:
        summary[one][1] /= summary[one][0]
        summary[one][2] /= summary[one][0]

    with open(fileroot + "Summaries\\realSummary.txt", 'w') as outfile:
        pprint.pprint(summary, outfile)
    with open(fileroot + "Summaries\\datasets.txt", 'w') as outfile:
        pprint.pprint(whichdatasets, outfile)
    sys.exit(0)


    order = ['pd', 'prec', 'acc', 'pf', 'f', 'g']


    compilation = dict()
    compilation2 = dict()
    compilationtmp = dict()
    for ord in order:
        with open(fileroot+'Summaries\\' + ord + '2.txt', 'w') as cout:
            #cout.write(ord +',')
            compilationtmp.clear()

            # method[stattype][dataset][method]
            for second in method[ord]:
                for third in method[ord][second]:
                    try:
                        count = method[ord][second][third]['count']
                        compilation[third]['count'] += count
                        compilation[third]['meds'] += method[ord][second][third]['medians']/(count+0.0)
                        compilation[third]['iqr'] += method[ord][second][third]['iqrs']/(count+0.0)
                    except KeyError:
                        count = method[ord][second][third]['count']
                        nested_set(compilation, [third, 'count'],count)
                        nested_set(compilation, [third, 'meds'], method[ord][second][third]['medians']/(count+0.0))
                        nested_set(compilation, [third, 'iqr'], method[ord][second][third]['iqr']/(count+0.0))
                    try:
                        count = method[ord][second][third]['count']
                        compilationtmp[third]['count'] += count
                        compilationtmp[third]['meds'] += method[ord][second][third]['medians'] / (count + 0.0)
                        compilationtmp[third]['iqr'] += method[ord][second][third]['iqrs'] / (count + 0.0)
                    except KeyError:
                        count = method[ord][second][third]['count']
                        nested_set(compilationtmp, [third, 'count'], count)
                        nested_set(compilationtmp, [third, 'meds'],
                                   method[ord][second][third]['medians'] / (count + 0.0))
                        nested_set(compilationtmp, [third, 'iqr'], method[ord][second][third]['iqr'] / (count + 0.0))
                        #cout.write(str(third).strip() + ',')
                    #cout.write("\n")

            # spill[stattype][dataset][spill]
            for second in spill[ord]:
                for third in spill[ord][second]:
                    try:
                        count = spill[ord][second][third]['count']
                        compilation[third]['count'] += count
                        compilation[third]['meds'] += spill[ord][second][third]['medians'] / (count + 0.0)
                        compilation[third]['iqr'] += spill[ord][second][third]['iqrs'] / (count + 0.0)
                    except KeyError:
                        count = spill[ord][second][third]['count']
                        nested_set(compilation, [third, 'count'], count)
                        nested_set(compilation, [third, 'meds'],
                                   spill[ord][second][third]['medians'] / (count + 0.0))
                        nested_set(compilation, [third, 'iqr'], spill[ord][second][third]['iqr'] / (count + 0.0))
                    try:
                        count = spill[ord][second][third]['count']
                        compilationtmp[third]['count'] += count
                        compilationtmp[third]['meds'] += spill[ord][second][third]['medians'] / (count + 0.0)
                        compilationtmp[third]['iqr'] += spill[ord][second][third]['iqrs'] / (count + 0.0)
                    except KeyError:
                        count = spill[ord][second][third]['count']
                        nested_set(compilationtmp, [third, 'count'], count)
                        nested_set(compilationtmp, [third, 'meds'],
                                   spill[ord][second][third]['medians'] / (count + 0.0))
                        nested_set(compilationtmp, [third, 'iqr'], spill[ord][second][third]['iqr'] / (count + 0.0))
                        #cout.write(str(third).strip() + ',')
                    #cout.write("\n")

            # depth[stattype][dataset][depth]
            for second in depth[ord]:
                for third in depth[ord][second]:
                    try:
                        count = depth[ord][second][third]['count']
                        compilation[third]['count'] += count
                        compilation[third]['meds'] += depth[ord][second][third]['medians'] / (count + 0.0)
                        compilation[third]['iqr'] += depth[ord][second][third]['iqrs'] / (count + 0.0)
                    except KeyError:
                        count = depth[ord][second][third]['count']
                        nested_set(compilation, [third, 'count'], count)
                        nested_set(compilation, [third, 'meds'],
                                   depth[ord][second][third]['medians'] / (count + 0.0))
                        nested_set(compilation, [third, 'iqr'], depth[ord][second][third]['iqr'] / (count + 0.0))
                    try:
                        count = depth[ord][second][third]['count']
                        compilationtmp[third]['count'] += count
                        compilationtmp[third]['meds'] += depth[ord][second][third]['medians'] / (count + 0.0)
                        compilationtmp[third]['iqr'] += depth[ord][second][third]['iqrs'] / (count + 0.0)
                    except KeyError:
                        count = depth[ord][second][third]['count']
                        nested_set(compilationtmp, [third, 'count'], count)
                        nested_set(compilationtmp, [third, 'meds'],
                                   depth[ord][second][third]['medians'] / (count + 0.0))
                        nested_set(compilationtmp, [third, 'iqr'], depth[ord][second][third]['iqr'] / (count + 0.0))
                        #cout.write(str(third).strip() + ',')
                    #cout.write("\n")


                    #cout.write(str(third).strip() + ',')
                #cout.write("\n")
            pprint.pprint(compilationtmp, cout)

    with open(fileroot + "Summaries\\summary2.txt", 'w') as outfile:
        pprint.pprint(compilation, outfile)
    with open(fileroot + "Summaries\\otherSummary2.txt", 'w') as outfile:
        pprint.pprint(compilation2, outfile)

def times_stats(fileroot, datafiles):
    print("here")
    infiles = []
    for dirName, subdirList, fileList in os.walk(fileroot):
        for k in datafiles:
            infiles.append([x for x in fileList if '.txt' in x if 'idea' not in x if 'error' not in x if "times" in x if k in x])
        return infiles
    # run times files through stats (make sure right files)

    # probably that's it

"""
print("x")
#files = ['accumulo', 'bookkeeper', 'camel', 'cassandra', 'cxf', 'derby', 'felix', 'hive', 'openjpa', 'pig', 'wicket']
#files = ['jm1', 'kc2']
files = ['ant2', 'arc2', 'berek2', 'camel2', 'elearning2', 'ivy2','jedit2', 'log4j2', 'lucene2', 'poi2', 'synapse2', 'xerces2']
#firstpart = 'C:\\Users\\Andrew\\Documents\\Schools\\Grad School\\NCSU - Comp Sci\\Research\\Overlaping Trees\\Data\\10 Datasets k3 Weighted\\'
firstpart = 'C:\\Users\\Andrew\\Documents\\Schools\\Grad School\\NCSU - Comp Sci\\Research\\Overlaping Trees\\Data\\Second Dataset\\Make Times\\'
dats = times_stats(firstpart, files)
menzies_time_stats(firstpart, dats)
"""
"""
firstpart = 'C:\\Users\\Andrew\\Documents\\Schools\\Grad School\\NCSU - Comp Sci\\Research\\Overlaping Trees\\Data\\Third Dataset - Mini\\'
abcd_master(firstpart, 'cart')
menzies_stats(firstpart+"cart\\")
"""

#files = ['accumulo', 'bookkeeper', 'camel', 'cassandra', 'cxf', 'derby', 'felix', 'hive', 'openjpa', 'pig', 'wicket']
#files = ['jm1', 'kc2']
files = ['ant2', 'arc2', 'berek2', 'camel2', 'elearning2', 'ivy2','jedit2', 'log4j2', 'lucene2', 'poi2', 'synapse2', 'xerces2']
firstpart = 'C:\\Users\\Andrew\\Documents\\Schools\\Grad School\\NCSU - Comp Sci\\Research\\Overlaping Trees\\Data\\1st Run\\'
# only accumulo and cxf for duplicates removed
for m in files:
    abcd_master(firstpart, m)

for f in files:
    path = firstpart + f + "\\"
    menzies_stats(path)

#files = ['accumulo', 'bookkeeper', 'camel', 'cassandra', 'cxf', 'derby', 'felix', 'hive', 'openjpa', 'pig', 'wicket']

"""
tree_depths = [5]
trees = ['kd']
spill_rates = [0.25]
nearneighbors = [3]
stats_on_stats(firstpart, files, nearneighbors, spill_rates, tree_depths, trees)
"""
"""
#path = 'C:\\Users\\Andrew\\Documents\\Schools\\Grad School\\NCSU - Comp Sci\\Research\\Overlaping Trees\Data\\10 Datasets Classifier Run 1\\Summaries\\'
#menzies_stats_mod(path)
#summary_master(write_all)
#statstree_master(menzies_statsmod)
#statstree_master(prep_stats)
#menzies_stats("C:\\Users\\Andrew\\PycharmProjects\\spatialtree\\1000samples\\unconstrained\\")
#sort_files()
# del_txt()
# graph_data()
# need to cut the first element off of mccabes_mc12 before using
"""
"""
outpath = "C:\\Users\\Andrew\\PycharmProjects\\spatialtree\\Mining Datasets\\Bellweather\\Promise Datasets\\CK2\\"
inpath = "C:\\Users\\Andrew\\PycharmProjects\\spatialtree\\Mining Datasets\\Bellweather\\Promise Datasets\\CK\\"
files = ['ant.csv', 'arc.csv', 'berek.csv', 'camel.csv', 'elearning.csv', 'ivy.csv', 'jedit.csv', 'log4j.csv', 'lucene.csv', 'poi.csv', 'prop6.csv', 'synapse.csv', 'tomcat.csv', 'xalan.csv', 'xerces.csv']
for f in files:
    for i in range(3):
        strip_csv(f, 0, inpath, outpath, 3)
"""
