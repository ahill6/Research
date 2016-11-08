d = 
counter = dict()
best3 = []
best2 = []
best = []

for k in d:
    for j in d[k]:
        if d[k][j] >= 5:
            best3.append(j)
        if d[k][j] >= 6:
            best2.append(j)
        if d[k][j] == 7:
            best.append(j)
            

for k in best:
    print(k)
#print(best2)
#print(best)