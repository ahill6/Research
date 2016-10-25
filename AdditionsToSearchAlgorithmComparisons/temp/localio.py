from math import floor
from collections import defaultdict
from random import sample, random
from stats import rdivDemo
from contextlib import contextmanager
from abcd import Abcd
from copy import deepcopy
import sys, os, numpy, csv, difflib, re
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


def csv_reader(filename):
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
    with open(filename) as temp_file:
        data = [[float(r) for r in line.rstrip('\n').split(',')] for line in temp_file]

    rand_smpl = [data[i] for i in sorted(sample(range(len(data)), min(len(data),1000)))]

    # record information about this particular dataset
    f = open(filename.split(".")[0] + "_stats.txt", 'w')
    f.write("lines\n" + str(len(data)) + "\n")
    f.write("decisions\n" + str(len(data[0])) + "\n")
    f.write("medians\n" + str(numpy.median(data, axis=0)) + "\n")
    f.write("variable minima\n" + str(numpy.amin(data, axis=0)) + "\n")
    f.write("variable maxima\n" + str(numpy.amax(data, axis=0)) + "\n")
    f.write("standard deviations\n" + str(numpy.std(data, axis=0)) + "\n")
    # numpy.corrcoef(a, rowvar=0) # save this for if needed

    #return data
    return rand_smpl


def csv_reader2(filename, mini=False):
    first = True
    with open(filename) as temp_file:
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


def strip_csv(file, col):
    cout = open('tmp.csv', 'w')
    with open(file, 'r') as f:
        for line in f:
            results = line.strip("\n").split(',')
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
    for method in ['recall', 'prec', 'select', 'acc']:
        fname = path + method + ".txt"
        data = []
        print(fname)
        print(os.path.isfile(fname))
        #input current file data
        if not os.path.isfile(fname):
            return
        with open(fname, 'r') as file:
            for cin in file:
                #cin = f.readline()
                cin = cin[:-2] # I don't think this is needed here (?)
                results = cin.strip("\n").split(',')
                label = results[0]
                results = results[1:]
                input = []
                input.append(label)
                input.extend([eval(x) for x in results])
                data.append(input)
        #make stats
        with open(path + method +"stats.txt", 'w') as out:
            rdivDemo(data, out)
        print("done")

def menzies_statsmod(path):
    files = [f for f in os.listdir(path) if '.txt' in f]
    input = []
    d = dict()
    i = 0
    collabel = "-1"
    first = False
    if len(files) <= 0:
        return
    for f in files:
        i += 1
        with open(path+f, 'r') as file:
            for cin in file:
                results = cin.strip("\n").split(',')
                label = results[0]
                results = results[1:]
                if i == 1:
                    d[label] = [eval(x) for x in results]
                else:
                    d[label].extend([eval(x) for x in results])
                #input.extend([eval(x) for x in results])
    data = []
    for k in d.keys():
        inp = [str(k)]
        inp.extend(list(d[k]))
        data.append(inp)
        #make stats
    with open(path+"\\summaries\\" + title, 'w') as out:
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
        infiles = [x for x in fileList if '.txt' in x if 'idea' not in x if 'error' not in x if (want == None or want in x) if dontwant not in x]
        print(dirName)
        func(path, infiles, **kwargs)

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

    directory_master(dir, abcd_caller, want = wanted, dontwant="times", acc=acc, pd=pd, pf=pf, prec=prec, f=f, g=g)

    #with filemanager(dir, "times"):
        #directory_master(dir, summarizeData, want="times")

    #come back at some point and make these a context manager
    acc.close()
    pd.close()
    pf.close()
    prec.close()
    f.close()
    g.close()


def abcd_caller(dir, files, **kwargs):
    log = None
    oldlabel = None
    files.sort(key=lambda x: x.replace(".txt", "").strip().split('-')[1])

    for f in files:
        #read in file
        label = f.replace(".txt", "").split('-')[1]
        log = None
        if os.stat(dir + f).st_size != 0:
            with open(dir + f, 'r') as cin:
                print(f)
                for line in cin:
                    words = re.sub(r"[\n\r]", "", line).split(",")
                    one, two = words[0], words[1]
                    log = Abcd(one, two)

                    # put the pieces in each file
                if label != oldlabel:
                    print("1")
                    if oldlabel == None:
                        print("2")
                        for x, s in sorted(log.scores().items()):
                            print(numpy.dot(numpy.array([s.acc for x, s in sorted(log.scores().items())]),
                                            [log.yes, log.no]) / (log.yes + log.no))
                            kwargs['acc'].write(label + "," + str(numpy.dot(numpy.array([s.acc for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                            kwargs['pd'].write(label + "," + str(numpy.dot(numpy.array([s.pd for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                            kwargs['pf'].write(label + "," + str(numpy.dot(numpy.array([s.pf for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                            kwargs['prec'].write(label + "," + str(numpy.dot(numpy.array([s.prec for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                            kwargs['f'].write(label + "," + str(numpy.dot(numpy.array([s.f for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                            kwargs['g'].write(label + "," + str(numpy.dot(numpy.array([s.g for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                    else:
                        print("3")
                        for x, s in sorted(log.scores().items()):
                            print(numpy.dot(numpy.array([s.acc for x, s in sorted(log.scores().items())]),
                                            [log.yes, log.no]) / (log.yes + log.no))
                            kwargs['acc'].write("\n" + label + "," + str(numpy.dot(numpy.array([s.acc for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                            kwargs['pd'].write("\n" + label + "," +  str(numpy.dot(numpy.array([s.pd for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                            kwargs['pf'].write("\n" + label + "," +  str(numpy.dot(numpy.array([s.pf for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                            kwargs['prec'].write("\n" + label + "," +  str(numpy.dot(numpy.array([s.prec for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                            kwargs['f'].write("\n" + label + "," +  str(numpy.dot(numpy.array([s.f for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                            kwargs['g'].write("\n" + label + "," +  str(numpy.dot(numpy.array([s.g for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                    oldlabel = deepcopy(label)
                else:
                    print("4")
                    for x, s in sorted(log.scores().items()):
                        print(numpy.dot(numpy.array([s.acc for x, s in sorted(log.scores().items())]),
                                        [log.yes, log.no]) / (log.yes + log.no))
                        kwargs['acc'].write("," + str(numpy.dot(numpy.array([s.acc for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['pd'].write("," +  str(numpy.dot(numpy.array([s.pd for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['pf'].write("," +  str(numpy.dot(numpy.array([s.pf for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['prec'].write("," +  str(numpy.dot(numpy.array([s.prec for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['f'].write("," +  str(numpy.dot(numpy.array([s.f for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))
                        kwargs['g'].write("," +  str(numpy.dot(numpy.array([s.g for x,s in sorted(log.scores().items())]),[log.yes, log.no])/(log.yes + log.no)))

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

string = 'C:\\Users\\Andrew\\Documents\\Schools\\Grad School\\NCSU - Comp Sci\\Research\\Overlaping Trees\\Data\\10 Datasets Classifier Run 1'
files = ['accumulo']
# accumulo was not in here when the abcd master thing was run...will need to run it for acc
for m in files:
    abcd_master(string, m)

#path = 'C:\\Users\\Andrew\\PycharmProjects\\spatialtree\\1000samples\\'
#menzies_statsmod(path)
#summary_master(write_all)
#statstree_master(menzies_statsmod)
#statstree_master(prep_stats)
#menzies_stats("C:\\Users\\Andrew\\PycharmProjects\\spatialtree\\1000samples\\unconstrained\\")
#sort_files()
# del_txt()
# graph_data()
# need to cut the first element off of mccabes_mc12 before using
