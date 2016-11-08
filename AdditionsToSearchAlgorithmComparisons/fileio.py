from math import floor
from collections import defaultdict
from tools import items
from copy import deepcopy
import sys, os, numpy, csv, classes

# Helper methods
def find_median(lst):
    if len(lst)%2 == 1:
        mid = int(floor(len(lst)/2))
        return lst[mid], lst[:mid], lst[(mid+1):]
    else:
        mid = int(floor(len(lst)/2))
        avg = (float(lst[mid-1]) + float(lst[mid]))/2.0
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
        
    # record information about this particular dataset    
    f = open(filename.split(".")[0]+"_stats.txt", 'w')
    f.write("lines\n"+str(len(data))+"\n")
    f.write("decisions\n"+str(len(data[0]))+"\n")
    meds = numpy.median(data, axis=0)
    mins = numpy.amin(data, axis=0)
    maxs = numpy.amax(data, axis=0)
    stds = numpy.std(data, axis=0)
    cov = numpy.corrcoef(data, rowvar=0)
    useless1 = []
    #useless2 = [ind for ind, mn, md, mx, sd in enumerate(zip(mins,meds, maxs, stds)) if (md-md < sd) if (mx-md < sd)]
    for i in xrange(len(meds)-1):
        if ((maxs[i] - mins[i]) < 1.96*stds[i]):
            useless1.append(i)
    #print(cov)
    #print(useless2)
    
    data_normed = data / maxs

    f.write("medians\n"+str(numpy.median(data, axis=0))+"\n")
    f.write("variable minima\n"+str(numpy.amin(data, axis=0))+"\n")
    f.write("variable maxima\n"+str(numpy.amax(data, axis=0))+"\n")
    f.write("standard deviations\n"+str(numpy.std(data, axis=0))+"\n")
    #numpy.corrcoef(a, rowvar=0) # save this for if needed
    """
    row_sums = data.sum(axis=1)
    new_matrix = data / row_sums[:, numpy.newaxis]
    row_sums[:, numpy.newaxis]
    """
    #data_normed = data / data.max(axis=0)
    return data_normed
    
def csv_reader2(filename):
    with open(filename) as temp_file:
        data = [[float(r) for r in line.rstrip('\n').split(',')] for line in temp_file]
    
    meds = numpy.median(data, axis=0)
    mins = numpy.amin(data, axis=0)
    maxs = numpy.amax(data, axis=0)
    stds = numpy.std(data, axis=0)
    cov = numpy.corrcoef(data, rowvar=0)
    useless = set()
    #useless2 = [ind for ind, mn, md, mx, sd in enumerate(zip(mins,meds, maxs, stds)) if (md-md < sd) if (mx-md < sd)]
    #print(useless2)
    
    # HERE RUN THROUGH COVARIANCE AND FIGURE OUT WHO NEEDS TO GO
    limit = .95
    for row in cov:
        useless.update([i for i, val in enumerate(row) if (val > limit and val != 1.0)])
    
    #print(useless)
        
    reduced = [[item[i] for i in xrange(len(item)) if i not in useless] for item in data]
    data_normed = data / data.max(axis=0)
    reduced_normed = reduced / reduced.max(axis=0)
    return data_normed, reduced_normed
    #return data, reduced
    
def pretty_print(f, d):
    for k in sorted(d.keys()):
        f.write(k,"\t\t : ", d[k])
        
def summarize(txt):
    path = '.'        
    #files = [f for f in os.listdir(path) if f.endswith('.txt') & (txt in f) & ('summary' not in f)]
    files = [f for f in os.listdir(path) if f.endswith('.txt') & ("_" in f) & ('summary' not in f)]
    
    """for f in files:
        print f
        print txt in f"""
        
    if len(files) > 0:
        summary = open('summary.txt', 'w')
        print "Summarizing..."
        
        for f in files:
            # open file
            f = open(f, 'r')
            print f.name
            
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
            
            summary.write(f.name+"\n"+output +"\n")
            
            # close data file and repeat
            f.close()
        pretty_print(summary, txt)    
        summary.close()
        print "Done"
        
def make_stats(num_entries=None, num_decisions=None):
    path = '.'        
    #files = [f for f in os.listdir(path) if f.endswith('.txt') & (txt in f) & ('summary' not in f)]
    files = [f for f in os.listdir(path) if f.endswith('.txt') if ("_" in f) if ('summary' not in f) if  ("stat" not in f)]
    timers = [f for f in os.listdir(path) if "." not in f]
    
    if len(files) > 0:
        stats = open('stats.txt', 'w')
        medians = open("medians.txt",'w')
        times = open('times.txt','w')
        print "Making Stats..."
        
        for g in timers:
            h = open(g, 'r')
            times.write(g + "\n" + h.readline()+"\n")
            h.close()
        times.close()
        
        for f in sorted(files):
            # open file
            f = open(f, 'r')
            #print f.name
            
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
                tp,fp,tn = item.split('-')
                tp = float(tp)
                fp = float(fp)
                tn = float(tn)
                fn = fp

                true_pos.append(tp)
                false_pos.append(fp)
                false_neg.append(fp)    #this would normally be otherwise
                true_neg.append(tn)
                
                recall.append((tp+0.0)/(fn + tp))
                pf.append((fp+0.0)/(tn+fp))
                prec.append((tp+0.0)/(tp+fp))
                acc.append((tn+tp+0.0)/(tn+fn+fp+tp))
                select.append((fp+tp+0.0)/(tn+fn+fp+tp))
                neg_pos.append((tn+fp+0.0)/(fn+tp))
                
                mean_recall += (tp+0.0)/(fn + tp)
                mean_pf += (fp+0.0)/(tn+fp)
                mean_prec += (tp+0.0)/(tp+fp)
                mean_acc += (tn+tp+0.0)/(tn+fn+fp+tp)
                mean_select += (fp+tp+0.0)/(tn+fn+fp+tp)
                mean_neg_pos += (tn+fp+0.0)/(fn+tp)
            
            # find iqr
            recall = sorted(recall)
            min = recall[0]
            _25 = find_median(find_median(recall)[1])[0]
            median = find_median(recall)[0]
            _75 = find_median(find_median(recall)[2])[0]
            max = recall[-1]
            
            # print that data to a summary file   
            total = len(recall)
            stats.write(f.name+"\n")
            medians.write(f.name+"\n")
            medians.write(str(min)+'\t'+str(_25) + "\t" + str(median) + "\t" + str(_75) + "\t" + str(max) + "\n")
            #summary.write(" ".join(str(_25),str(median),str(_75)) + "\n")
            stats.write("rec " + str(mean_recall/total) + "\n")
            stats.write("pf " + str(mean_pf/total) + "\n")
            stats.write("prec " + str(mean_prec/total) + "\n")
            stats.write("acc " + str(mean_acc/total) + "\n")
            stats.write("select " + str(mean_select/total) + "\n")
            stats.write("neg/pos " + str(mean_neg_pos/total) + "\n")
            
            # close data file and repeat
            f.close()
        #pretty_print(summary, txt)    
        stats.close()
        medians.close()
        print "Done"
        
        table_please(infile = stats.name, num_decisions=num_decisions, num_entries=num_entries) # need some way to pass num_entries and num_decisions to table_please
        table_please(infile = medians.name, num_decisions=num_decisions, num_entries=num_entries) # need some way to pass num_entries and num_decisions to table_please

def strip_csv(file, col):
    cout = open('tmp.csv', 'w')
    with open(file, 'r') as f:
        for line in f:
            results = line.strip("\n").split(',')
            del results[col]
            cout.write(','.join([r for r in results])+"\n")
    cout.close()

def table_please(num_entries=None, num_decisions=None, infile=None):
    results = dict()
    #print(infile)

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
        x,y,z = cin.split()
        e = [y, eval(z) - eval(x)]
        nested_set(results,[s,d,m], e)
    def stats(cin):
        x,y = cin.split()
        nested_set(results,[s,d,x,m], y)
    def graphing(cin):
        x,y,z = cin.split()
        e = [y, eval(z) - eval(x)]
        nested_set(results,[s,d,m], e)
    
    file = infile or 'summary.txt'
    if 'median' in infile:
        process = medians
        out = 'medians.ods'
    elif 'stat' in infile:
        process = stats
        out = 'stats.ods'
    else:
        raise ValueError('Invalid filename.', infile)
        
    f = open(file,'r')
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
        spl = dictionary.keys()[0]
        dpt = dictionary[spl].keys()[0]
        tmp = dictionary[spl][dpt].keys()
        cout.write(','+',,'.join(tmp))
        header = 'median,iqr,'*len(tmp)
        cout.write("\nSpill-depth," + header + "\n")
        for spill in dictionary:
            for depth in dictionary[spill]:
                tmp = [str(x[0]) + ',' + str(x[1]) for x in dictionary[spill][depth].values()]
                cout.write(str(spill) + ' ' + str(depth) + ',' + ','.join(tmp) + "\n")
    def stats():
        for spill in dictionary:
            for depth in dictionary[spill]:
                ind = dictionary[spill][depth].keys()[0]
                tmp = dictionary[spill][depth][ind].keys()
                cout.write(str(spill) + ' ' + str(depth) + ',' + ','.join(tmp) + "\n")
                for statistic in dictionary[spill][depth]:
                    tmp = dictionary[spill][depth][statistic].values()
                    cout.write(statistic + ',' + ','.join(tmp) + "\n")
                newline = ','*(len(dictionary[spill][depth][statistic])+1)+"\n"
                cout.write(newline*2)
        
    cout = open(outfile,'w')
    
    #this is a terrible kludge.  Fix eventually.
    if 'median' in outfile:
        printprocess = medians
    else:
        printprocess = stats
    #cout = open('table.ods','w')
    
    # print number of entries and decisions for reference/interest
    cout.write("" + str(entries) + ' entries , ' + str(decisions) + " decisions\n")
    # print the whole dictionary in a particular order/way (using the keys as headers)
    printprocess()
    #del_txt()

def graph_data():
    path = '.'      
    files = [f for f in os.listdir(path) if f.endswith('.txt') if ("_" in f) if ('summary' not in f) if  ("stat" not in f)]
    #timers = [f for f in os.listdir(path) if "." not in f]

    if len(files) > 0:
        chart = open("chart_data.txt",'w')
        print "Prepping Data for Graphs..."
        
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
            #print f.name
            
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
                tp,fp,tn = item.split('-')
                tp = float(tp)
                fp = float(fp)
                tn = float(tn)
                fn = fp

                true_pos.append(tp)
                false_pos.append(fp)
                false_neg.append(fp)    #this would normally be otherwise
                true_neg.append(tn)
                
                recall.append((tp+0.0)/(fn + tp))
                pf.append((fp+0.0)/(tn+fp))
                prec.append((tp+0.0)/(tp+fp))
                acc.append((tn+tp+0.0)/(tn+fn+fp+tp))
                select.append((fp+tp+0.0)/(tn+fn+fp+tp))
                neg_pos.append((tn+fp+0.0)/(fn+tp))
                
                
                mean_recall += (tp+0.0)/(fn + tp)
                mean_pf += (fp+0.0)/(tn+fp)
                mean_prec += (tp+0.0)/(tp+fp)
                mean_acc += (tn+tp+0.0)/(tn+fn+fp+tp)
                mean_select += (fp+tp+0.0)/(tn+fn+fp+tp)
                mean_neg_pos += (tn+fp+0.0)/(fn+tp)
            
            # find iqr
            recall = sorted(recall)
            min = recall[0]
            _25 = find_median(find_median(recall)[1])[0]
            median = find_median(recall)[0]
            _75 = find_median(find_median(recall)[2])[0]
            max = recall[-1]
            
            # print that data to a summary file   
            total = len(recall)
            chart.write(f.name+"\n")
            chart.write(str(min)+'\t'+str(_25) + "\t" + str(median) + "\t" + str(_75) + "\t" + str(max) + "\n")
            #summary.write(" ".join(str(_25),str(median),str(_75)) + "\n")
            chart.write("rec " + str(mean_recall/total) + "\n")
            chart.write("pf " + str(mean_pf/total) + "\n")
            chart.write("prec " + str(mean_prec/total) + "\n")
            chart.write("acc " + str(mean_acc/total) + "\n")
            chart.write("select " + str(mean_select/total) + "\n")
            chart.write("neg/pos " + str(mean_neg_pos/total) + "\n")
            
            # close data file and repeat
            f.close()
        #pretty_print(summary, txt)    
        chart.close()
        print "Done"    
        
def make_me_a_table(num_entries, num_decisions, infile=None):
    rob = ResultFactory('name')
    
    file = infile or 'summary.txt'
    f = open(file,'r')
    lst = []
    
    # read in the data and put into separate Results objects
    with cin as f:
        try:
            m, s, d = cin.strip('.txt').split('_') # strip removes all instances of everything passed, not exact matches.
            if hasattr(bob, 'method'):
                bob.add_values(lst)
                lst = []
            bob = rob.make_result(method=m, spill=s, depth=d)
        except:
            x,y = cin.split()
            lst.append(dict(name = x, value = y))
    
    # data is in Result objects in ResultFactory.  Send to printer (writer)
    write_table(rob, num_entries, num_decisions)
    
def write_table(data, num_entries, num_decisions):
    #methods = # num of distinct methods?  Do I need to write a table, then go back and write the header?
    # put everything into a dictionary, then pull it out in a comprehensible order
    for x in xrange(2*len(methods)+1):
        out.write(',')
    cout = "\n,data = " + str(num_entries) + ', ' + str(num_decisions) + "decisions"
    for x in xrange(2*len(methods)):
        cout += ','
    out.write(cout+"\n,,")
    for x in methods:
        out.write(x+',,')
    out.write(',,\n,Spill-depth,')
    cout = 'median,iqr,'*len(methods)
    out.write(cout+',\n')
    for a in spill:
        for b in tree_depth:
            out.write(',' + str(b) + ' ' + str(a) + ',')
            for t in trees: # need to guarantee these go in order
                out.write(median[t] + ',' + iqr[t] + ',')
            out.write(',\n')
    cout = [',' for i in 2*len(methods) + 1].tostring
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
            
#strip_csv('arc.csv',-1)
#make_stats()
#del_txt()
#csv_reader("camel967.csv")
#graph_data()
#need to cut the first element off of mccabes_mc12 before using