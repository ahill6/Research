def pretty_print(d):
    for k in d:
        print k,"\t\t : ", d[k]

def tester(demo_type, samples, trials):
    x = 1
    #data = matrixTestMaster(0, 3)
    
    #pretty_print(data)


d = {"a": 5, "b": 2}
e = {"a": 6, "e": 1}
for i in e:
    if i in d:
        d[i] += e[i]
    else:
        d[i] = e[i]
print(d)