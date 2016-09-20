from math import floor
import sys, os, numpy

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

def summarize(txt):
    path = '.'        
    files = [f for f in os.listdir(path) if f.endswith('.txt') & (txt in f) & ('summary' not in f)]
    
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
            in_tree = cin.split(', ')
            
            cin = f.readline()
            out_of_tree = cin.split(', ')
            
            # order that, find percentiles (min, 25%, median, 75%, max)
            in_tree = sorted(in_tree)
            out_of_tree = sorted(out_of_tree)
            
            in_min = in_tree[0]
            in_25 = find_median(find_median(in_tree)[1])[0]
            in_median = find_median(in_tree)[0]
            in_75 = find_median(find_median(in_tree)[2])[0]
            in_max = in_tree[-1]
            
            out_min = out_of_tree[0]
            out_25 = find_median(find_median(out_of_tree)[1])[0]
            out_median = find_median(out_of_tree)[0]
            out_75 = find_median(find_median(out_of_tree)[2])[0]
            out_max = out_of_tree[-1]
            
            # print that data to a summary file
            in_tree_output = "" + str(in_min) + "," + str(in_25) + "," + str(in_median) + "," + str(in_75) + "," + str(in_max)
            out_of_tree_output = "" + str(out_min) + "," + str(out_25) + "," + str(out_median) + "," + str(out_75) + "," + str(out_max)
            
            summary.write(in_tree_output +"\n")
            summary.write(out_of_tree_output +"\n")
            
            # close data file and repeat
            f.close()
            
        summary.close()
        print "Done"

#summarize('.01_5')
"""
w = open('data2.csv', 'w')
with open('data.csv') as temp_file:
    for line in temp_file:
        if not line[0]=='$':
            line = line[:-4]+"\n"
        else:
            line = line[:-8]+"\n"
        w.write(line)
"""