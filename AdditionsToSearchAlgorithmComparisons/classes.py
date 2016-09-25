class O(object):
    """
    Basic Class which every other class inherits
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __repr__(self):
        return self.__class__.__name__ + kv(self.__dict__)
        
class Result(O):
    """
    A data structure to store the results of an experimental run
    """
    def __init__(self, method, spill=None, depth=None):
        O.__init__(self, method, spill, depth, values = {})
        
    def add_values(self, **data):
        for datum in data:
            print(datum)
            print(datum.name, datum.value)
            sys.exit(0)
            values[datum.name] = datum.value
        
class ResultFactory(O):
    def __init__(self, name):
        """
        Initialize the factory.
        """
        O.__init__(self, name, results = [])
        
    def __lt__(self, other):
        t1 = self.method, self.spill, self.depth
        t2 = other.method, other.spill, other.depth
        return t1 < t2
    
    def make_result(self, method, spill=None, depth=None):
        # TODO 5: Create a new machine and add it to
        # the list "machines" and return the machine
        a = Result(method = method, spill=spill, depth=depth)
        self.results.append(a)
        return True