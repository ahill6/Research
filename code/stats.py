"""

## Hyptotheis Testing Stuff


### Standard Stuff

#### Standard Headers

"""
import sys,random,math
sys.dont_write_bytecode = True
"""

#### Standard Utils

"""
class o():
  "Anonymous container"
  def __init__(i,**fields) :
    i.override(fields)
  def override(i,d): i.__dict__.update(d); return i
  def __repr__(i):
    d = i.__dict__
    name = i.__class__.__name__
    return name+'{'+' '.join([':%s %s' % (k,pretty(d[k]))
                     for k in i.show()])+ '}'
  def show(i):
    return [k for k in sorted(i.__dict__.keys())
            if not "_" in k]
"""

Misc functions:

"""
rand = random.random
any  = random.choice
seed = random.seed
exp  = lambda n: math.e**n
ln   = lambda n: math.log(n,math.e)
g    = lambda n: round(n,2)

def median(lst,ordered=False):
  if not ordered: lst= sorted(lst)
  n = len(lst)
  p = n//2
  if n % 2: return lst[p]
  q = p - 1
  q = max(0,min(q,n))
  return (lst[p] + lst[q])/2

def msecs(f):
  import time
  t1 = time.time()
  f()
  return (time.time() - t1) * 1000

def pairs(lst):
  "Return all pairs of items i,i+1 from a list."
  last=lst[0]
  for i in lst[1:]:
    yield last,i
    last = i

def xtile(lst,lo=0,hi=100,width=50,
             chops=[0.1 ,0.3,0.5,0.7,0.9],
             marks=["-" ," "," ","-"," "],
             bar="|",star="*",show=" %3.0f"):
  """The function _xtile_ takes a list of (possibly)
  unsorted numbers and presents them as a horizontal
  xtile chart (in ascii format). The default is a
  contracted _quintile_ that shows the
  10,30,50,70,90 breaks in the data (but this can be
  changed- see the optional flags of the function).
  """
  def pos(p)   : return ordered[int(len(lst)*p)]
  def place(x) :
    return int(width*float((x - lo))/(hi - lo+0.00001))
  def pretty(lst) :
    return ', '.join([show % x for x in lst])
  ordered = sorted(lst)
  lo      = min(lo,ordered[0])
  hi      = max(hi,ordered[-1])
  what    = [pos(p)   for p in chops]
  where   = [place(n) for n in  what]
  out     = [" "] * width
  for one,two in pairs(where):
    for i in range(one,two):
      out[i] = marks[0]
    marks = marks[1:]
  out[int(width/2)]    = bar
  out[place(pos(0.5))] = star
  return '('+''.join(out) +  ")," +  pretty(what)

def _tileX() :
  import random
  random.seed(1)
  nums = [random.random()**2 for _ in range(100)]
  print(xtile(nums,lo=0,hi=1.0,width=25,show=" %5.2f"))
"""

### Standard Accumulator for Numbers

Note the _lt_ method: this accumulator can be sorted by median values.

Warning: this accumulator keeps _all_ numbers. Might be better to use
a bounded cache.

"""
class Num:
  "An Accumulator for numbers"
  def __init__(i,name,inits=[]):
    i.n = i.m2 = i.mu = 0.0
    i.all=[]
    i._median=None
    i.name = name
    i.rank = 0
    for x in inits: i.add(x)
  def s(i)       : return (i.m2/(i.n - 1))**0.5
  def add(i,x):
    i._median=None
    i.n   += 1
    i.all += [x]
    delta  = x - i.mu
    i.mu  += delta*1.0/i.n
    i.m2  += delta*(x - i.mu)
  def __add__(i,j):
    return Num(i.name + j.name,i.all + j.all)
  def quartiles(i):
    def p(x) : return int(100*g(xs[x]))
    i.median()
    xs = i.all
    n  = int(len(xs)*0.25)
    return p(n) , p(2*n) , p(3*n)
  def median(i):
    if not i._median:
      i.all = sorted(i.all)
      i._median=median(i.all)
    return i._median
  def __lt__(i,j):
    return i.median() < j.median()
  def spread(i):
    i.all=sorted(i.all)
    n1=i.n*0.25
    n2=i.n*0.75
    if len(i.all) <= 1:
      return 0
    if len(i.all) == 2:
      return i.all[1] - i.all[0]
    else:
      return i.all[int(n2)] - i.all[int(n1)]


"""

### The A12 Effect Size Test

"""
def a12slow(lst1,lst2):
  "how often is x in lst1 more than y in lst2?"
  more = same = 0.0
  for x in lst1:
    for y in lst2:
      if    x == y : same += 1
      elif  x >  y : more += 1
  x= (more + 0.5*same) / (len(lst1)*len(lst2))
  return x

def a12(lst1,lst2):
  "how often is x in lst1 more than y in lst2?"
  def loop(t,t1,t2):
    while t1.j < t1.n and t2.j < t2.n:
      h1 = t1.l[t1.j]
      h2 = t2.l[t2.j]
      h3 = t2.l[t2.j+1] if t2.j+1 < t2.n else None
      if h1>  h2:
        t1.j  += 1; t1.gt += t2.n - t2.j
      elif h1 == h2:
        if h3 and h1 > h3 :
            t1.gt += t2.n - t2.j  - 1
        t1.j  += 1; t1.eq += 1; t2.eq += 1
      else:
        t2,t1  = t1,t2
    return t.gt*1.0, t.eq*1.0
  #--------------------------
  lst1 = sorted(lst1, reverse=True)
  lst2 = sorted(lst2, reverse=True)
  n1   = len(lst1)
  n2   = len(lst2)
  t1   = o(l=lst1,j=0,eq=0,gt=0,n=n1)
  t2   = o(l=lst2,j=0,eq=0,gt=0,n=n2)
  gt,eq= loop(t1, t1, t2)
  return gt/(n1*n2) + eq/2/(n1*n2)

def _a12():
  def f1(): return a12slow(l1,l2)
  def f2(): return a12(l1,l2)
  for n in [100,200,400,800,1600,3200,6400]:
    l1 = [rand() for _ in range(n)]
    l2 = [rand() for _ in range(n)]
    t1 = msecs(f1)
    t2 = msecs(f2)
    print(n, g(f1()),g(f2()),int((t1/t2)))


"""Output:

````
n   a12(fast)       a12(slow)       tfast / tslow
--- --------------- -------------- --------------
100  0.53           0.53               4
200  0.48           0.48               6
400  0.49           0.49              28
800  0.5            0.5               26
1600 0.51           0.51              72
3200 0.49           0.49             109
6400 0.5            0.5              244
````


## Non-Parametric Hypothesis Testing

The following _bootstrap_ method was introduced in
1979 by Bradley Efron at Stanford University. It
was inspired by earlier work on the
jackknife.
Improved estimates of the variance were [developed later][efron01].

[efron01]: http://goo.gl/14n8Wf "Bradley Efron and R.J. Tibshirani. An Introduction to the Bootstrap (Chapman & Hall/CRC Monographs on Statistics & Applied Probability), 1993"


To check if two populations _(y0,z0)_
are different, many times sample with replacement
from both to generate _(y1,z1), (y2,z2), (y3,z3)_.. etc.

"""
def sampleWithReplacement(lst):
  "returns a list same size as list"
  def any(n)  : return random.uniform(0,n)
  def one(lst): return lst[ int(any(len(lst))) ]
  return [one(lst) for _ in lst]
"""


Then, for all those samples,
 check if some *testStatistic* in the original pair
hold for all the other pairs. If it does more than (say) 99%
of the time, then we are 99% confident in that the
populations are the same.

In such a _bootstrap_ hypothesis test, the *some property*
is the difference between the two populations, muted by the
joint standard deviation of the populations.

"""
def testStatistic(y,z):
    """Checks if two means are different, tempered
     by the sample size of 'y' and 'z'"""
    tmp1 = tmp2 = 0
    for y1 in y.all: tmp1 += (y1 - y.mu)**2
    for z1 in z.all: tmp2 += (z1 - z.mu)**2
    s1    = (float(tmp1)/(y.n - 1))**0.5
    s2    = (float(tmp2)/(z.n - 1))**0.5
    delta = z.mu - y.mu
    if s1+s2:
      delta =  delta/((s1/y.n + s2/z.n)**0.5)
    return delta
"""

The rest is just details:

+ Efron advises
  to make the mean of the populations the same (see
  the _yhat,zhat_ stuff shown below).
+ The class _total_ is a just a quick and dirty accumulation class.
+ For more details see [the Efron text][efron01].

"""
def bootstrap(y0,z0,conf=0.01,b=1000):
  """The bootstrap hypothesis test from
     p220 to 223 of Efron's book 'An
    introduction to the boostrap."""
  class total():
    "quick and dirty data collector"
    def __init__(i,some=[]):
      i.sum = i.n = i.mu = 0 ; i.all=[]
      for one in some: i.put(one)
    def put(i,x):
      i.all.append(x);
      i.sum +=x; i.n += 1; i.mu = float(i.sum)/i.n
    def __add__(i1,i2): return total(i1.all + i2.all)
  y, z   = total(y0), total(z0)
  x      = y + z
  tobs   = testStatistic(y,z)
  yhat   = [y1 - y.mu + x.mu for y1 in y.all]
  zhat   = [z1 - z.mu + x.mu for z1 in z.all]
  bigger = 0.0
  for i in range(b):
    if testStatistic(total(sampleWithReplacement(yhat)),
                     total(sampleWithReplacement(zhat))) > tobs:
      bigger += 1
  return bigger / b < conf
"""

#### Examples

"""
def _bootstraped():
  def worker(n=1000,
             mu1=10,  sigma1=1,
             mu2=10.2, sigma2=1):
    def g(mu,sigma) : return random.gauss(mu,sigma)
    x = [g(mu1,sigma1) for i in range(n)]
    y = [g(mu2,sigma2) for i in range(n)]
    return n,mu1,sigma1,mu2,sigma2,\
        'different' if bootstrap(x,y) else 'same'
  # very different means, same std
  print(worker(mu1=10, sigma1=10,
               mu2=100, sigma2=10))
  # similar means and std
  print(worker(mu1= 10.1, sigma1=1,
               mu2= 10.2, sigma2=1))
  # slightly different means, same std
  print(worker(mu1= 10.1, sigma1= 1,
               mu2= 10.8, sigma2= 1))
  # different in mu eater by large std
  print(worker(mu1= 10.1, sigma1= 10,
               mu2= 10.8, sigma2= 1))
"""

Output:

````
_bootstraped()

(1000, 10, 10, 100, 10, 'different')
(1000, 10.1, 1, 10.2, 1, 'same')
(1000, 10.1, 1, 10.8, 1, 'different')
(1000, 10.1, 10, 10.8, 1, 'same')
````

Warning- the above took 8 seconds to generate since we used 1000 bootstraps.
As to how many bootstraps are enough, that depends on the data. There are
results saying 200 to 400 are enough but, since I am  suspicious man, I run it for 1000.

Which means the runtimes associated with bootstrapping is a significant issue.
To reduce that runtime, I avoid things like an all-pairs comparison of all treatments
(see below: Scott-knott).  Also, BEFORE I do the boostrap, I first run
the effect size test (and only go to bootstrapping in effect size passes:

"""
def different(l1,l2):
  #return bootstrap(l1,l2) and a12(l2,l1)
  return a12(l2,l1) and bootstrap(l1,l2)

"""

## Saner Hypothesis Testing

The following code, which you should use verbatim does the following:


+ All treatments are clustered into _ranks_. In practice, dozens
  of treatments end up generating just a handful of ranks.
+ The numbers of calls to the hypothesis tests are minimized:
    + Treatments are sorted by their median value.
    + Treatments are divided into two groups such that the
      expected value of the mean values _after_ the split is minimized;
    + Hypothesis tests are called to test if the two groups are truly difference.
          + All hypothesis tests are non-parametric and include (1) effect size tests
            and (2) tests for statistically significant numbers;
          + Slow bootstraps are executed  if the faster _A12_ tests are passed;

In practice, this means that the hypothesis tests (with confidence of say, 95%)
are called on only a logarithmic number of times. So...

+ With this method, 16 treatments can be studied using less than _&sum;<sub>1,2,4,8,16</sub>log<sub>2</sub>i =15_ hypothesis tests  and confidence _0.99<sup>15</sup>=0.86_.
+ But if did this with the 120 all-pairs comparisons of the 16 treatments, we would have total confidence _0.99<sup>120</sup>=0.30.

For examples on using this code, see _rdivDemo_ (below).

"""
def scottknott(data,cohen=0.3,small=3, useA12=False,epsilon=0.01):
  """Recursively split data, maximizing delta of
  the expected value of the mean before and
  after the splits.
  Reject splits with under 3 items"""
  all  = reduce(lambda x,y:x+y,data)
  same = lambda l,r: abs(l.median() - r.median()) <= all.s()*cohen
  if useA12:
    same = lambda l, r:   not different(l.all,r.all)
  big  = lambda    n: n > small
  return rdiv(data,all,minMu,big,same,epsilon)

def rdiv(data,  # a list of class Nums
         all,   # all the data combined into one num
         div,   # function: find the best split
         big,   # function: rejects small splits
         same, # function: rejects similar splits
         epsilon): # small enough to split two parts
  """Looks for ways to split sorted data,
  Recurses into each split. Assigns a 'rank' number
  to all the leaf splits found in this way.
  """
  def recurse(parts,all,rank=0):
    "Split, then recurse on each part."
    inputcut, inputleft, inputright = div(parts, all, big, epsilon) # python 3 no longer supports tuple unpacking, repairing maybeIgnore
    cut,left,right = maybeIgnore(inputcut, inputleft, inputright,
                                 same,parts)
    if cut:
      # if cut, rank "right" higher than "left"
      rank = recurse(parts[:cut],left,rank) + 1
      rank = recurse(parts[cut:],right,rank)
    else:
      # if no cut, then all get same rank
      for part in parts:
        part.rank = rank
    return rank
  recurse(sorted(data),all)
  return data

def maybeIgnore(cut,left,right, same,parts):
  if cut:
    if same(sum(parts[:cut],Num('upto')),
            sum(parts[cut:],Num('above'))):
      cut = left = right = None
  return cut,left,right

def minMu(parts,all,big,epsilon):
  """Find a cut in the parts that maximizes
  the expected value of the difference in
  the mean before and after the cut.
  Reject splits that are insignificantly
  different or that generate very small subsets.
  """
  cut,left,right = None,None,None
  before, mu     =  0, all.mu
  for i,l,r in leftRight(parts,epsilon):
    if big(l.n) and big(r.n):
      n   = all.n * 1.0
      now = l.n/n*(mu- l.mu)**2 + r.n/n*(mu- r.mu)**2
      if now > before:
        before,cut,left,right = now,i,l,r
  return cut,left,right

def leftRight(parts,epsilon=0.01):
  """Iterator. For all items in 'parts',
  return everything to the left and everything
  from here to the end. For reasons of
  efficiency, take a first pass over the data
  to pre-compute and cache right-hand-sides
  """
  rights = {}
  n = j = len(parts) - 1
  while j > 0:
    rights[j] = parts[j]
    if j < n: rights[j] += rights[j+1]
    j -=1
  left = parts[0]
  for i,one in enumerate(parts):
    if i> 0:
      if parts[i]._median - parts[i-1]._median > epsilon:
        yield i,left,rights[i]
      left += one
"""

## Putting it All Together

Driver for the demos:

"""

count = 1

def rdivDemo(data, output):
  print("Starting")
  def z(x):
    return int(100 * (x - lo) / (hi - lo + 0.00001))
  data = map(lambda lst:Num(lst[0],lst[1:]),
             data)
  output.write("")
  ranks=[]
  for x in scottknott(data,useA12=True):
    ranks += [(x.rank,x.median(),x)]
  all=[]
  for _,__,x in sorted(ranks): all += x.all
  all = sorted(all)
  lo, hi = all[0], all[-1]
  line = "----------------------------------------------------"
  last = None
  output.write(('%4s , %12s ,    %s   , %4s ' % \
               ('rank', 'name', 'med', 'iqr'))+ "\n"+ line + "\n")
  for _,__,x in sorted(ranks):
    q1,q2,q3 = x.quartiles()
    output.write(('%4s , %12s ,    %4s  ,  %4s ' % \
                 (x.rank+1, x.name, q2, q3 - q1))  + \
              xtile(x.all,lo=lo,hi=hi,width=30,show="%5.2f") + "\n")
    last = x.rank
"""

The demos:

"""
def rdiv0():
  rdivDemo([
      ['2-means',0.36,0.5,0.49,0.4,0.41,0.36,0.46,0.2,0.46,0.34,0.5,0.49,0.38,0.48,0.46,0.33,0.47,0.35,0.49,0.48,0.36,0.48,0.5,0.42,0.42,0.45,0.4,0.46,0.5,0.46,0.46,0.48,0.47,0.48,0.41,0.47,0.4,0.46,0.46,0.46,0.44,0.39,0.26,0.34,0.41,0.41,0.41,0.1,0.39,0.44,0.48,0.44,0.49,0.48,0.48,0.5,0.39,0.44,0.46,0.34,0.4,0.5,0.34,0.49,0.49,0.39,0.5,0.15,0.5,0.37,0.48,0.45,0.17,0.24,0.49,0.5,0.38,0.41,0.39,0.41,0.59,0.45,0.24,0.5,0.26,0.69,0.48,0.22,0.46,0.03,0.49,0.48,0.48,0.46,0.5,0.35,0.46,0.49,0.46,0.44,0.29,0.45,0.5,0.62,0.4,0.32,0.5,0.33,0.45,0.33,0.35,0.31,0.28,0.48,0.36,0.46,0.39,0.35,0.5,0.5,0.42,0.47,0.45,0.25,0.46,0.22,0.28,0.29,0.42,0.48,0.5,0.47,0.46,0.46,0.13,0.43,0.47,0.41,0.31,0.49,0.37,0.39,0.18,0.26,0.41,0.24,0.41,0.5,0.45,0.59,0.36,0.41,0.37,0.72,0.43,0.46,0.37,0.44,0.5,0.49,0.48,0.15,0.49,0.32,0.18,0.5,0.83,0.25,0.28,0.46,0.35,0.48,0.34,0.48,0.2,0.1,0.32,0.43,0.46,0.77,0.49,0.34,0.49,0.43,0.47,0.46,0.35,0.41,0.47,0.19,0.44,0.46,0.32],
['kd',0.33,0.36,0.41,0.51,0.34,0.45,0.39,0.39,0.37,0.26,0.34,0.5,0.05,0.35,0.35,0.36,0.22,0.5,0.26,0.38,0.46,0.52,0.32,0.29,0.47,0.33,0.35,0.25,0.14,0.32,0.28,0.1,0.52,0.49,0.28,0.44,0.48,0.23,0.48,0.31,0.38,0.49,0.39,0.7,0.21,0.43,0.42,0.33,0.48,0.29,0.42,0.39,0.33,0.36,0.24,0.49,0.29,0.28,0.39,0.47,0.48,0.58,0.18,0.32,0.34,0.32,0.34,0.45,0.49,0.5,0.16,0.15,0.37,0.39,0.74,0.34,0.69,0.28,0.34,0.38,0.46,0.46,0.34,0.24,0.25,0.36,0.04,0.24,0.42,0.46,0.46,0.27,0.29,0.45,0.19,0.27,0.33,0.36,0.38,0.27,0.39,0.29,0.48,0.43,0.41,0.36,0.36,0.34,0.43,0.23,0.36,0.33,0.35,0.18,0.3,0.41,0.24,0.48,0.25,0.35,0.3,0.32,0.18,0.44,0.2,0.4,0.36,0.4,0.22,0.44,0.81,0.31,0.43,0.51,0.59,0.42,0.21,0.25,0.2,0.23,0.48,0.23,0.5,0.35,0.44,0.36,0.05,0.51,0.05,0.43,0.4,0.32,0.47,0.36,0.45,0.36,0.35,0.36,0.54,0.45,0.27,0.46,0.33,0.48,0.34,0.32,0.23,0.38,0.34,0.48,0.47,0.5,0.3,0.24,0.4,0.45,0.37,0.35,0.42,0.15,0.47,0.48,0.35,0.71,0.37,0.35,0.33,0.35,0.39,0.39,0.3,0.36,0.36],
['pca',0.5,0.13,0.26,0.5,0.43,0.22,0.43,0.49,0.3,0.15,0.49,0.44,0.04,0.19,0.5,0.41,0.15,0.48,0.41,0.35,0.5,0.37,0.49,0.45,0.5,0.45,0.43,0.44,0.82,0.45,0.37,0.04,0.44,0.41,0.27,0.38,0.5,0.36,0.46,0.11,0.13,0.49,0.29,0.5,0.5,0.48,0.47,0.32,0.45,0.42,0.4,0.39,0.24,0.39,0.33,0.48,0.45,0.27,0.48,0.38,0.41,0.46,0.67,0.74,0.47,0.45,0.28,0.34,0.26,0.48,0.48,0.43,0.35,0.04,0.95,0.45,0.31,0.37,0.19,0.33,0.49,0.34,0.47,0.5,0.5,0.17,0.5,0.45,0.5,0.29,0.45,0.49,0.47,0.47,0.35,0.49,0.39,0.62,0.27,0.5,0.48,0.29,0.5,0.45,0.5,0.62,0.5,0.26,0.45,0.05,0.33,0.14,0.31,0.45,0.28,0.31,0.43,0.49,0.25,0.24,0.34,0.5,0.46,0.5,0.45,0.4,0.51,0.43,0.45,0.32,0.31,0.48,0.32,0.39,0.5,0.5,0.16,0.5,0.34,0.31,0.43,0.36,0.49,0.24,0.33,0.47,0.46,0.29,0.46,0.34,0.5,0.45,0.39,0.5,0.41,0.31,0.49,0.13,0.33,0.45,0.24,0.35,0.4,0.49,0.17,0.5,0.46,0.49,0.23,0.21,0.44,0.47,0.15,0.49,0.45,0.36,0.5,0.25,0.3,0.44,0.43,0.48,0.48,0.24,0.48,0.5,0.24,0.21,0.27,0.31,0.49,0.36,0.35],
['random',0.22,0.46,0.37,0.63,0.33,0.48,0.34,0.38,0.41,0.41,0.41,0.6,0.43,0.37,0.48,0.45,0.37,0.44,0.48,0.38,0.3,0.83,0.37,0.44,0.29,0.44,0.35,0.42,0.37,0.32,0.46,0.35,0.44,0.38,0.36,0.79,0.28,0.48,0.37,0.4,0.46,0.39,0.44,0.14,0.35,0.38,0.31,0.18,0.42,0.32,0.37,0.39,0.34,0.48,0.39,0.45,0.45,0.44,0.46,0.42,0.41,0.43,0.21,0.41,0.38,0.41,0.67,0.44,0.3,0.35,0.49,0.31,0.47,0.43,0.49,0.42,0.44,0.44,0.48,0.43,0.39,0.49,0.49,0.42,0.36,0.38,0.23,0.48,0.28,0.39,0.45,0.35,0.34,0.48,0.45,0.41,0.45,0.39,0.11,0.37,0.38,0.33,0.38,0.44,0.48,0.29,0.34,0.4,0.34,0.18,0.37,0.38,0.37,0.44,0.18,0.84,0.48,0.44,0.46,0.48,0.47,0.34,0.46,0.39,0.48,0.32,0.49,0.64,0.28,0.47,0.33,0.45,0.46,0.48,0.34,0.36,0.47,0.48,0.37,0.34,0.29,0.15,0.44,0.43,0.48,0.15,0.32,0.48,0.45,0.47,0.18,0.44,0.47,0.4,0.41,0.34,0.31,0.41,0.47,0.42,0.32,0.48,0.43,0.43,0.4,0.26,0.44,0.48,0.42,0.44,0.4,0.24,0.47,0.49,0.47,0.3,0.37,0.43,0.45,0.23,0.36,0.36,0.36,0.48,0.4,0.39,0.48,0.46,0.38,0.32,0.29,0.89,0.44],
['rp',0.12,0.36,0.42,0.43,0.46,0.29,0.43,0.32,0.36,0.29,0.46,0.22,0.4,0.49,0.45,0.45,0.19,0.32,0.27,0.49,0.39,0.26,0.24,0.29,0.45,0.28,0.42,0.29,0.4,0.46,0.72,0.25,0.21,0.49,0.46,0.43,0.3,0.37,0.35,0.04,0.36,0.43,0.07,0.12,0.19,0.28,0.44,0.39,0.24,0.09,0.21,0.18,0.33,0.38,0.64,0.43,0.32,0.46,0.29,0.47,0.13,0.14,0.29,0.45,0.29,0.18,0.42,0.45,0.09,0.29,0.35,0.21,0.43,0.18,0.16,0.21,0.41,0.43,0.5,0.29,0.11,0.35,0.48,0.18,0.29,0.31,0.47,0.19,0.43,0.46,0.44,0.5,0.36,0.24,0.46,0.45,0.49,0.43,0.35,0.34,0.19,0.36,0.33,0.29,0.08,0.38,0.25,0.48,0.5,0.28,0.26,0.16,0.46,0.48,0.44,0.29,0.21,0.32,0.31,0.47,0.61,0.09,0.01,0.38,0.35,0.23,0.12,0.45,0.41,0.43,0.31,0.48,0.25,0.46,0.48,0.47,0.31,0.37,0.3,0.44,0.45,0.22,0.25,0.48,0.47,0.29,0.43,0.18,0.2,0.32,0.29,0.3,0.29,0.29,0.19,0.45,0.3,0.35,0.28,0.46,0.23,0.3,0.41,0.47,0.3,0.35,0.4,0.23,0.46,0.41,0.4,0.16,0.29,0.16,0.31,0.07,0.41,0.45,0.45,0.28,0.5,0.37,0.43,0.31,0.34,0.41,0.4,0.29,0.27,0.34,0.36,0.47,0.16],
['spectral',0.45,0.3,0.36,0.35,0.29,0.39,0.43,0.43,0.36,0.6,0.4,0.39,0.39,0.39,0.29,0.45,0.39,0.5,0.45,0.4,0.33,0.39,0.27,0.66,0.43,0.35,0.35,0.42,0.35,0.24,0.35,0.17,0.5,0.43,0.48,0.34,0.4,0.12,0.64,0.39,0.24,0.25,0.4,0.39,0.39,0.35,0.45,0.45,0.45,0.33,0.34,0.28,0.3,0.31,0.23,0.34,0.68,0.49,0.39,0.36,0.39,0.36,0.35,0.19,0.29,0.48,0.33,0.39,0.16,0.3,0.25,0.3,0.22,0.37,0.3,0.35,0.33,0.4,0.5,0.37,0.26,0.5,0.35,0.3,0.18,0.45,0.59,0.47,0.32,0.27,0.33,0.19,0.33,0.36,0.27,0.36,0.39,0.34,0.33,0.4,0.39,0.4,0.37,0.42,0.36,0.49,0.26,0.42,0.43,0.4,0.45,0.5,0.21,0.41,0.34,0.33,0.4,0.33,0.42,0.41,0.35,0.39,0.38,0.24,0.34,0.38,0.26,0.32,0.38,0.03,0.33,0.28,0.36,0.63,0.36,0.4,0.34,0.18,0.4,0.37,0.31,0.27,0.43,0.41,0.36,0.27,0.36,0.33,0.37,0.29,0.33,0.31,0.26,0.35,0.37,0.34,0.41,0.21,0.36,0.35,0.4,0.41,0.5,0.33,0.36,0.28,0.43,0.31,0.33,0.44,0.29,0.5,0.41,0.48,0.41,0.24,0.59,0.27,0.32,0.3,0.65,0.38,0.35,0.39,0.35,0.23,0.36,0.25,0.38,0.36,0.39,0.37,0.19],
['where',0.24,0.33,0.37,0.19,0.34,0.32,0.16,0.27,0.46,0.51,0.25,0.21,0.27,0.21,0.31,0.23,0.22,0.23,0.22,0.23,0.31,0.09,0.46,0.19,0.22,0.51,0.51,0.23,0.33,0.2,0.18,0.26,0.19,0.51,0.36,0.15,0.51,0.29,0.25,0.17,0.22,0.33,0.18,0.3,0.28,0.37,0.14,0.43,0.29,0.23,0.26,0.21,0.28,0.28,0.22,0.25,0.2,0.28,0.26,0.21,0.32,0.26,0.21,0.21,0.31,0.19,0.36,0.32,0.21,0.15,0.3,0.23,0.23,0.15,0.22,0.25,0.51,0.27,0.24,0.24,0.29,0.22,0.37,0.21,0.25,0.24,0.24,0.14,0.22,0.21,0.29,0.3,0.15,0.31,0.3,0.29,0.29,0.29,0.35,0.22,0.42,0.27,0.26,0.3,0.36,0.21,0.06,0.25,0.22,0.47,0.27,0.29,0.37,0.27,0.18,0.29,0.24,0.07,0.26,0.28,0.29,0.28,0.22,0.28,0.26,0.23,0.13,0.36,0.25,0.49,0.27,0.2,0.24,0.51,0.3,0.28,0.16,0.26,0.27,0.39,0.27,0.29,0.17,0.21,0.07,0.2,0.26,0.18,0.34,0.24,0.2,0.25,0.34,0.22,0.16,0.31,0.13,0.2,0.47,0.51,0.42,0.23,0.2,0.29,0.3,0.38,0.18,0.28,0.27,0.28,0.36,0.24,0.27,0.34,0.15,0.24,0.48,0.14,0.11,0.86,0.16,0.26,0.49,0.39,0.4,0.16,0.15,0.14,0.37,0.51,0.24,0.52,0.14],
])
"""

````
rank ,         name ,    med   ,  iqr
----------------------------------------------------
   1 ,           x1 ,      51  ,    11 (*              |              ), 0.34,  0.49,  0.51,  0.51,  0.60
   2 ,           x2 ,     800  ,   200 (               |   ----   *-- ), 6.00,  7.00,  8.00,  8.00,  9.00
````

"""
def rdiv1():
  rdivDemo([
        ["x1",0.1,  0.2,  0.3,  0.4],
        ["x2",0.1,  0.2,  0.3,  0.4],
        ["x3",6,  7,  8,  9] ])
"""

````
rank ,         name ,    med   ,  iqr
----------------------------------------------------
   1 ,           x1 ,      30  ,    20 (*              |              ), 0.10,  0.20,  0.30,  0.30,  0.40
   1 ,           x2 ,      30  ,    20 (*              |              ), 0.10,  0.20,  0.30,  0.30,  0.40
   2 ,           x3 ,     800  ,   200 (               |   ----   *-- ), 6.00,  7.00,  8.00,  8.00,  9.00
````

"""
def rdiv2():
  rdivDemo([
        ["x1",0.34, 0.49, 0.51, 0.6],
        ["x2",0.6,  0.7,  0.8,  0.9],
        ["x3",0.15, 0.25, 0.4,  0.35],
        ["x4",0.6,  0.7,  0.8,  0.9],
        ["x5",0.1,  0.2,  0.3,  0.4] ])
"""

````
rank ,         name ,    med   ,  iqr
----------------------------------------------------
   1 ,           x5 ,      30  ,    20 (---    *---    |              ), 0.10,  0.20,  0.30,  0.30,  0.40
   1 ,           x3 ,      35  ,    15 ( ----    *-    |              ), 0.15,  0.25,  0.35,  0.35,  0.40
   2 ,           x1 ,      51  ,    11 (        ------ *--            ), 0.34,  0.49,  0.51,  0.51,  0.60
   3 ,           x2 ,      80  ,    20 (               |  ----    *-- ), 0.60,  0.70,  0.80,  0.80,  0.90
   3 ,           x4 ,      80  ,    20 (               |  ----    *-- ), 0.60,  0.70,  0.80,  0.80,  0.90
````

"""
def rdiv3():
  rdivDemo([
      ["x1",101, 100, 99,   101,  99.5],
      ["x2",101, 100, 99,   101, 100],
      ["x3",101, 100, 99.5, 101,  99],
      ["x4",101, 100, 99,   101, 100] ])
"""

````
rank ,         name ,    med   ,  iqr
----------------------------------------------------
   1 ,           x1 ,    10000  ,   150 (-------       *|              ),99.00, 99.50, 100.00, 101.00, 101.00
   1 ,           x2 ,    10000  ,   100 (--------------*|              ),99.00, 100.00, 100.00, 101.00, 101.00
   1 ,           x3 ,    10000  ,   150 (-------       *|              ),99.00, 99.50, 100.00, 101.00, 101.00
   1 ,           x4 ,    10000  ,   100 (--------------*|              ),99.00, 100.00, 100.00, 101.00, 101.00
````

"""
def rdiv4():
  rdivDemo([
      ["x1",11,12,13],
      ["x2",14,31,22],
      ["x3",23,24,31],
      ["x5",32,33,34]])
"""

````
rank ,         name ,    med   ,  iqr
----------------------------------------------------
   1 ,           x1 ,    1100  ,     0 ( *             |              ),11.00, 11.00, 12.00, 13.00, 13.00
   1 ,           x2 ,    1400  ,     0 (              *|              ),14.00, 14.00, 22.00, 31.00, 31.00
   2 ,           x3 ,    2300  ,     0 (               |*             ),23.00, 23.00, 24.00, 31.00, 31.00
   2 ,           x5 ,    3200  ,     0 (               |            * ),32.00, 32.00, 33.00, 34.00, 34.00
````

"""
def rdiv5():
  rdivDemo([
      ["x1",11,11,11],
      ["x2",11,11,11],
      ["x3",11,11,11]])
"""

````
rank ,         name ,    med   ,  iqr
----------------------------------------------------
   1 ,           x1 ,    1100  ,     0 (*              |              ),11.00, 11.00, 11.00, 11.00, 11.00
   1 ,           x2 ,    1100  ,     0 (*              |              ),11.00, 11.00, 11.00, 11.00, 11.00
   1 ,           x3 ,    1100  ,     0 (*              |              ),11.00, 11.00, 11.00, 11.00, 11.00
````

"""
def rdiv6():
  rdivDemo([
      ["x1",11,11,11],
      ["x2",11,11,11],
      ["x4",32,33,34,35]])
"""

````
rank ,         name ,    med   ,  iqr
----------------------------------------------------
   1 ,           x1 ,    1100  ,     0 (*              |              ),11.00, 11.00, 11.00, 11.00, 11.00
   1 ,           x2 ,    1100  ,     0 (*              |              ),11.00, 11.00, 11.00, 11.00, 11.00
   2 ,           x4 ,    3400  ,   200 (               |          - * ),32.00, 33.00, 34.00, 34.00, 35.00
````

"""
def rdiv7():
  rdivDemo([
    ["x1"] +  [rand()**0.5 for _ in range(256)],
    ["x2"] +  [rand()**2   for _ in range(256)],
    ["x3"] +  [rand()      for _ in range(256)]
  ])
"""

````
rank ,         name ,    med   ,  iqr
----------------------------------------------------
   1 ,           x2 ,      25  ,    50 (--     *      -|---------     ), 0.01,  0.09,  0.25,  0.47,  0.86
   2 ,           x3 ,      49  ,    47 (  ------      *|   -------    ), 0.08,  0.29,  0.49,  0.66,  0.89
   3 ,           x1 ,      73  ,    37 (         ------|-    *   ---  ), 0.32,  0.57,  0.73,  0.86,  0.95
````

"""

def _rdivs():
  rdiv0()
"""
  seed(1)
  rdiv0();  rdiv1(); rdiv2(); rdiv3();
  rdiv4(); rdiv5(); rdiv6(); rdiv7()
"""
    #read data in
    # generate graph


#_rdivs()