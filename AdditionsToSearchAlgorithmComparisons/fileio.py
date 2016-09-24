from math import floor
import sys, os, numpy, csv

# Helper methods
def find_median(lst):
    if len(lst)%2 == 1:
        mid = int(floor(len(lst)/2))
        return lst[mid], lst[:mid], lst[(mid+1):]
    else:
        mid = int(floor(len(lst)/2))
        avg = (float(lst[mid-1]) + float(lst[mid]))/2.0
        return avg, lst[:mid], lst[mid:] 

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

    return data
    
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
        
def make_stats():
    path = '.'        
    #files = [f for f in os.listdir(path) if f.endswith('.txt') & (txt in f) & ('summary' not in f)]
    files = [f for f in os.listdir(path) if f.endswith('.txt') & ("_" in f) & ('summary' not in f)]
    
    if len(files) > 0:
        summary = open('summary.txt', 'w')
        print "Making Stats..."
        
        for f in sorted(files):
            # open file
            f = open(f, 'r')
            print f.name
            
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
            _25 = find_median(find_median(recall)[1])[0]
            median = find_median(recall)[0]
            _75 = find_median(find_median(recall)[2])[0]
            
            # print that data to a summary file   
            total = len(recall)
            summary.write(f.name+"\n")
            summary.write(str(_25) + "\t" + str(median) + "\t" + str(_75) + "\n")
            #summary.write("\t".join(str(_25),str(median),str(_75)) + "\n")
            summary.write("rec:\t " + str(mean_recall/total) + "\n")
            summary.write("pf:\t " + str(mean_pf/total) + "\n")
            summary.write("prec:\t " + str(mean_prec/total) + "\n")
            summary.write("acc:\t " + str(mean_acc/total) + "\n")
            summary.write("select:\t " + str(mean_select/total) + "\n")
            summary.write("neg/pos:\t " + str(mean_neg_pos/total) + "\n")
            
            # close data file and repeat
            f.close()
        #pretty_print(summary, txt)    
        summary.close()
        print "Done"   

def strip_csv(file, col):
    cout = open('tmp.csv', 'w')
    with open(file, 'r') as f:
        for line in f:
            results = line.strip("\n").split(',')
            del results[col]
            cout.write(','.join([r for r in results])+"\n")
    cout.close()


def make_table(methods, spills, tree_depths, num_entries, num_decisions, summary data):
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
            
#strip_csv('mccabes_mc12.csv',0)            
make_stats()
#need to cut the first element off of mccabes_mc12 before using