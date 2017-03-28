from clang.cindex import *
import plyj.parser as plyj
#import plyj.model
from urllib2 import urlopen, HTTPError, URLError
from bs4 import BeautifulSoup
from numpy import median, mean, std
from sys import exit, stdout
from random import sample
from shutil import rmtree

def gitMiner(url, cout, cerr, base='https://github.com'):
    try:
        response = urlopen(url)
    except HTTPError as err:
        if err.code == 404:
            return 0,0
        else:
            cerr.write(url + ","+err.message)
            return 0,0
    except URLError as err:
        cerr.write(url+"--URLError,"+err.message)
        return 0,0
    except Exception as err:
        cerr.write(url+"--Other Exception,"+err.message)
        return 0,0
    html = response.read()
    #soup = BeautifulSoup(html, 'lxml')
    soup = BeautifulSoup(html, "html.parser")
    fileCount = 0
    functionCount = 0
    for possibles in soup.find_all('td'):
        if 'class' in possibles.attrs and possibles['class'][0] == 'content':
            folders = possibles.find_all('a')

            if len(folders) > 0:
                #if possibles.text.strip()[-2:] == '.c':
                # if '.c' in possibles.text.strip():
                if '.java' in possibles.text.strip():
                    print("Go read file " + folders[0].get('href'))
                    try:
                        #functionCount += readCFile(folders[0].get('href'), cout, cerr, base)
                        functionCount += readJavaFile(folders[0].get('href'), cout, cerr, base)
                    except Exception as e:
                        cerr.write(folders[0].get('href')+",")
                        cerr.write(e.message)
                        cerr.write("\n")
                    fileCount += 1
                elif '.' in possibles.text:
                    continue
                else:
                    for link in folders:
                        print("Recurse on "  + base+link.get('href'))
                        f1, f2 = gitMiner(base+link.get('href'), cout, cerr, base)
                        fileCount += f1
                        functionCount += f2
        # open file
    return fileCount, functionCount

def readCFile(url, outfile, err, base='https://github.com'):
    response = urlopen(base+url)
    html = response.read()
    #soup = BeautifulSoup(html, 'lxml')
    soup = BeautifulSoup(html)
    for link in soup.find_all('a'):
        if link.text == 'Raw':
            file = urlopen(base + link.get('href'))
            with open('./Output/tmp.c', 'w') as ctmp:
                ctmp.write(file.read())
            return functionExtractor(outfile, err)


def readJavaFile(url, outfile, err, base='https://github.com'):
    response = urlopen(base+url)
    html = response.read()
    #soup = BeautifulSoup(html, 'lxml')
    soup = BeautifulSoup(html)
    for link in soup.find_all('a'):
        if link.text == 'Raw':
            file = urlopen(base + link.get('href'))
            with open('./Java/tmp.java', 'w') as ctmp:
                ctmp.write(file.read())
            return javaFunctionExtractor(outfile, err)


def inFunctionData(cur):
    ct = 0
    for child_node in cur.get_children():
        #print(child_node.kind, child_node.spelling)
        ct += inFunctionData(child_node)
        if child_node.kind == CursorKind.CALL_EXPR:
            # print(child_node.spelling)
            ct += 1
    return ct

def functionExtractor(outFile, errFile, inFile=None):
    # need to return : 1) number of functions
    # 2) also get types of input and return parameters

    index = Index.create()
    count = 0
    functionCalls = 0
    fctnCheck = []
    paramsCheck = []
    if inFile is None:
        inFile = './Output/tmp.c'
    tu = index.parse(inFile)
    #print('Translation unit:', tu.spelling)
    for c in tu.cursor.get_children():
        if c.kind == CursorKind.FUNCTION_DECL or c.kind == CursorKind.CXX_METHOD:
            vals = [i for i, x in enumerate(fctnCheck) if c.spelling == x]
            if len(vals) < 1 or len([1 for r in paramsCheck if r in vals if paramsCheck[r] == c.type.spelling.strip(')').split('(') ]) < 1:
                outFile.write(c.spelling)
                outFile.write('-')
                fctnCheck.append(c.spelling)
                tmp = c.type.spelling.strip(')').split('(')
                paramsCheck.append(tmp)
                functionCalls = inFunctionData(c)
                if len(tmp[1:]) > 0:
                    invars = tmp[1:][0].split(',')
                    invars = [r.strip() for r in invars]
                    outFile.write(','.join(sorted(invars))) # input variables
                    outFile.write('-')
                    outFile.write(tmp[0].strip()) # output variables
                    outFile.write('-')
                    outFile.write(str(functionCalls))
                    outFile.write(",")
                    outFile.write("\n")
                    count += 1
                else:
                    continue

    return count

def javaFunctionExtractor(outFile, errFile, inFile='./Java/tmp.java'):

    # need to return : 1) number of functions
    # 2) also get types of input and return parameters
    parser = plyj.Parser()
    count = 0
    tree = parser.parse_file(inFile)
    for t in tree.type_declarations[0].body:
        if str(type(t)) == "<class 'plyj.model.MethodDeclaration'>":
            invars = []
            functionCalls = 0
            outFile.write(str(t.name))
            outFile.write('-')
            tmp = str(t.body)
            functionCalls = tmp.count('MethodInvocation')
            for r in t.parameters:
                if isinstance(r.type, str):
                    invars.append(r.type)
                else:
                    invars.append(str(r.type.name.value))
                # Note that if has input type of List<Object>, for all s in r.type.name.type_arguments, s.name.value has Object
            invars = [r.strip() for r in invars]
            outFile.write(','.join(sorted(invars))) # input variables
            outFile.write('-')
            outFile.write(str(t.return_type)) # output variables
            outFile.write('-')
            outFile.write(str(functionCalls))
            outFile.write(",")
            outFile.write("\n")
            count += 1

    return count



def readGitList(fname='./Output/githubList.txt'):
    with open(fname, 'r') as cin:
        lst = []
        for c in cin:
            tp = c.strip("\n").strip(',').split(',')
            lst.extend(tp)
    return lst

def printData(data, boundaryPoints=[5,20,50], outMethod=stdout):
    #b = sorted(boundaryPoints)
    print(data)
    res = {}
    boundaryPoints.sort()
    for b in range(len(boundaryPoints)):
        res[boundaryPoints[b]] = [r for r in data.keys() if data[r] >= boundaryPoints[b] ]
    total = sum([data[r] for r in data.keys()])
    percents = { b: [(data[r]+0.0)/total for r in data.keys() if r in res[b]] for b in res.keys()}
    for b in res.keys():
        outMethod.write(str(total)+"\n")
        outMethod.write(str(b) + ":")
        outMethod.write(str(res[b]))
        outMethod.write("=")
        outMethod.write(str(percents[b])+"\n")



def dataAnalysis(resultsFiles, path='./Output/'):
    names = {}
    inParams = {}
    outParams = {}
    combined = {}
    functionCalls = []
    total = 0

    for r in resultsFiles:
        with open(r, 'r') as cin:
            for line in cin:
                dats = line.strip("\n").split('-')
                #print(dats)
                if len(dats) == 4:
                    namen = dats[0]
                    #eis = dats[1].split(',')
                    eis = dats[1]
                    ex = dats[2]
                    machinae = dats[3].strip(',')
                elif len(dats) == 5:
                    namen = dats[0]
                    #eis = dats[1].split(',')
                    eis = dats[-3]
                    ex = dats[-2]
                    machinae = dats[-1].strip(',')
                else:
                    print("Whoops")
                    continue
                inout = eis + "-" + ex

                if namen in names:
                    names[namen] += 1
                else:
                    names[namen] = 1
                if eis in inParams:
                    inParams[eis] += 1
                else:
                    inParams[eis] = 1
                if ex in outParams:
                    outParams[ex] += 1
                else:
                    outParams[ex] = 1
                if inout in combined:
                    combined[inout] += 1
                else:
                    combined[inout] = 1
                functionCalls.append(int(machinae))
                total += 1

    reportStats = [[(k, names[k]) for k in sorted(names, key=names.get, reverse=True)],
                   [(k, inParams[k]) for k in sorted(inParams, key=inParams.get, reverse=True)],
                   [(k, outParams[k]) for k in sorted(outParams, key=outParams.get, reverse=True)],
                   [(k, combined[k]) for k in sorted(combined, key=combined.get, reverse=True)]]
    output = [open(path+"names.txt", 'w'), open(path+"inParams.txt", 'w'), open(path+"outParams.txt", 'w'), open(path+"combined.txt", 'w')]

    for i in range(len(reportStats)):
        tmp = 0
        j = 0
        while tmp < .8*total:# and reportStats[i][j][-1] > 10:
            tmp += reportStats[i][j][-1]
            output[i].write(str(reportStats[i][j])+"\n")
            j += 1
        output[i].close()

    if not not functionCalls:
        with open(path+'functionCalls.txt', 'w') as fctns:
            fctns.write("Min: " +str(min(functionCalls))+"\n")
            fctns.write("Max: " + str(max(functionCalls)) + "\n")
            fctns.write("Mean: " + str(mean(functionCalls)) + "\n")
            fctns.write("Std: " + str(std(functionCalls)) + "\n")
            fctns.write("Med: " + str(median(functionCalls)) + "\n")


if __name__ != '__main__':
    stuffs = []
    #with open('./Output/githubMaster.txt','r') as input:
    with open('./Output/githubList2.txt', 'r') as input:
        for line in input:
            stuffs.extend(line.strip().split(','))
    gits = sample(stuffs, len(stuffs)//2)
    #gits = sample(stuffs, len(stuffs)//10000)
    Config.set_library_path('C:/Program Files (x86)/LLVM/bin')
    files = 0
    functions = 0
    realProjects = 0
    #gits = readGitList()
    resultsFile = open('./Output/results.txt', 'w')
    errorFile = open('./Output/errors.txt', 'w')
    fake = open('./fake.txt', 'w')

    for u in gits:
        print(u)
        ft1, ft2 = gitMiner(u, resultsFile, errorFile)
        files += ft1
        functions += ft2
        if files == 0 and functions == 0:
            fake.write(u)
    resultsFile.close()
    errorFile.close()
    fake.close()
    with open('./Output/stats.txt','w') as stats:
        stats.write("Total files: " + str(files)+"\n")
        stats.write("Total functions: " + str(functions))

    print("Total files: " + str(files))
    print("Total functions: " + str(functions))
    #show_ast(tu.cursor, no_system_includes)
    dataAnalysis(['./Output/results.txt'])

# take input url
def tarClangParse(url):
    import tarfile, os
    def c_files(members):
        for tarinfo in members:
            if os.path.splitext(tarinfo.name)[1] == ".c":
                yield tarinfo

    resultsFile = open('./Output/Tars/LimitedToo/results.txt', 'w')
    errorFile = open('./Output/Tars/LimitedToo/errors.txt', 'w')
    fileCount = 0
    functionCount = 0
    inFiles = ['gzip-bug-2010-02-19-3eb6091d69-884ef6d16c.tar.gz', 'libtiff-bug-2010-12-13-96a5fb4-bdba15c.tar.gz', 'python-bug-70120-70124.tar.gz']
    base = 'C:/Users/Andrew/Downloads/Ugh/'


    for inf in inFiles:
        tar = tarfile.open(base + inf)
        tar.extractall(path='./Tmp/Data/', members=c_files(tar))
        tar.close()
        print("Proceeding to analysis for "+inf)
        for root, dirs, files in os.walk("./Tmp/Data/", topdown=False):
            for name in files:
                fileCount += 1
                functionCount += functionExtractor(resultsFile, errorFile, inFile=os.path.join(root, name))
        print("Analysis for " + inf + " complete")
        print("Deleting tmp files")
        rmtree('./Tmp/Data/', ignore_errors=False, onerror=None)
        print("Done \n")

    resultsFile.close()
    errorFile.close()
    #os.remove('./Tmp/tmp.tar.gz')
    with open('./Output/Tars/LimitedToo/output.txt', 'w') as cout:
        cout.write("Total files: " + str(fileCount) + "\n")
        cout.write("Total functions: " + str(functionCount))
    dataAnalysis(['./Output/Tars/LimitedToo/results.txt'], './Output/Tars/LimitedToo/')
    exit()


    # open that url
    try:
        response = urlopen(url)
    except HTTPError as err:
        if err.code == 404:
            print(err.message)
    html = response.read()
    soup = BeautifulSoup(html, "html.parser")
    for possibles in soup.find_all('td'):
        tmp = possibles.find_all('a')
        if len(tmp) > 0 and '.tar.gz' in tmp[0].string and ('gzip' in tmp[0].string or 'libtiff' in tmp[0].string or 'python' in tmp[0].string):
            line = tmp[0].get('href')
            f = urlopen(url+line)
            with open("./Tmp/tmp.tar.gz", "wb") as code:
                code.write(f.read())
            print(line + " tmp tar written")
            tar = tarfile.open('./Tmp/tmp.tar.gz')
            tar = tarfile.open('./Tmp/tmp.tar.gz')
            tar.extractall(path='./Tmp/Data/', members=c_files(tar))
            tar.close()
            print("Proceeding to analysis for "+line)
            for root, dirs, files in os.walk("./Tmp/Data/", topdown=False):
                for name in files:
                    fileCount += 1
                    functionCount += functionExtractor(resultsFile, errorFile, inFile=os.path.join(root, name))
            print("Analysis for " + line + " complete")
            print("Deleting tmp files")
            rmtree('./Tmp/Data/', ignore_errors=False, onerror=None)
            print("Done \n")

    resultsFile.close()
    errorFile.close()
    os.remove('./Tmp/tmp.tar.gz')
    with open('./Output/Tars/Limited/output.txt', 'w') as cout:
        cout.write("Total files: " + str(fileCount) + "\n")
        cout.write("Total functions: " + str(functionCount))
    dataAnalysis(['./Output/Tars/Limited/results.txt'], './Output/Tars/Limited/')

"""
def recurseCheck(cursor):
    ct = 0
    for c in tu.cursor.get_children():
        #print(c.kind, c.spelling)
        d = c.get_children()
        if not not d:
            ct += recurseCheck(d)

        if c.kind == CursorKind.FUNCTION_DECL and '__' not in c.spelling:
            print(c.spelling, c.type.spelling)
            tmp = c.type.spelling.strip(')').split('(')
            print(tmp[1:][0])
            print(tmp[0])


    print(ct)
    return ct


def traverse(cursor, level):
    ct = 0
    for child_node in cursor.get_children():
        print(child_node.kind, child_node.spelling)
        ct += traverse(child_node, level+1)
        if child_node.kind == CursorKind.CALL_EXPR:
            #print(child_node.spelling)
            ct += 1
    return ct


# read median.c
Config.set_library_path('C:/Program Files (x86)/LLVM/bin')
index = Index.create()
tu = index.parse('./Output/tmp.c')
# print('Translation unit:', tu.spelling)
#recurseCheck(tu.cursor.get_children())
print(traverse(tu.cursor, 0))
#print(inFunctionData(tu.cursor))

p = './Output/Tars/LimitedToo/'
dataAnalysis([p+'results.txt'], p)
exit()
tarClangParse('http://dijkstra.cs.virginia.edu/genprog/resources/autorepairbenchmarks/ManyBugs/scenarios/')
"""
p = './Java/'
dataAnalysis([p+'results.txt'], p)
exit()

stuffs = []
with open('./Java/githubJavaList.txt', 'r') as input:
    for line in input:
        stuffs.extend(line.strip().split(','))

gits = sample(stuffs, len(stuffs)//10000)
files = 0
functions = 0
realProjects = 0
resultsFile = open('./Java/results.txt', 'w')
errorFile = open('./Java/errors.txt', 'w')
fake = open('./fake.txt', 'w')

for u in gits:
    print(u)
    ft1, ft2 = gitMiner(u, resultsFile, errorFile)
    files += ft1
    functions += ft2
    if files == 0 and functions == 0:
        fake.write(u)
resultsFile.close()
errorFile.close()
fake.close()
