import sys, fileio, random

"""
def less(x,y): return x < y
def more(x,y): return x > y

def above(x,y,tbl,better=more):
  "single objective"
  klass=tbl.klass[0].pos
  return better(x[klass], y[klass])

def below(x,y,tbl):
  "single objective"
  return above(x,y,tbl, better=less)

def bdom(x, y, abouts):
  "multi objective"
  x=abouts.objs(x)
  y=abouts.objs(y)
  betters = 0
  for obj in abouts._objs:
    x1,y1 = x[obj.pos], y[obj.pos]
    if obj.better(x1,y1) : betters += 1
    elif x1 != y1: return False # must be worse, go quit
  return betters > 0
"""

def sway(population, tbl, better) :
  swayCull = 0.5
  swayStop = 2
  swayBigger = 0.2
  
  def cluster(items, out):
    if len(items) < max(len(population)**swayCull, swayStop):
      out.append(tbl.clone(items))
    else:
      west, east, left, right = split(items, int(len(items)/2)) 
      if not better(east,west,tbl): cluster( left, out )  #need to implement better
      if not better(west,east,tbl): cluster( right, out )  
    return out
    
  def output(x):
    sys.stdout.write(str(x)); sys.stdout.flush()
    
  def split(items, mid,west=None, east=None,redo=20):
    assert redo>0
    cosine = lambda a,b,c: ( a*a + c*c - b*b )/( 2*c+ 0.0001 )
    #if west is None: west = any(items) 
    #if east is None: east = any(items)
    if west is None: west = tbl.furthest(any(items))
    #if east is None: east = tbl.furthest(west)
    if east is None: east = any(items)
    while east.rid == west.rid:
      east = any(items)
    c      = tbl.distance(west, east)
    xs     = {}
    for n,item in enumerate(items):
       a = tbl.distance(item, west)
       b = tbl.distance(item, east)
       x = xs[ item.rid ] = cosine(a,b,c) # cosine rule
       if a > c and abs(a-c)  > swayBigger:
         output(">%s " % n)
         return split(items, mid, west=west, east=item, redo=redo-1)
       if b > c and abs(b-c) > swayBigger:
         output("<%s " % n)
         return split(items, mid, west=item, east=east, redo=redo-1)   
    items = sorted(items, key= lambda item: xs[ item.rid ]) # sorted by 'x'
    return west, east, items[:mid], items[mid:] 
  # --------
  return cluster(population, [])
  
def _sway(file="data.csv"):
  random.seed(101)
  data = fileio.csv_reader(file)
  leafs = sway(tbl0._rows,tbl0,below)
  n=0
  for c,tbl in enumerate(leafs):
    n += len(tbl._rows)
  print(n)
  
_sway()