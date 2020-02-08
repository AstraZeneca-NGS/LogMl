
class CounterDim:
    """
    Counts in 'dims' dimentions up to 'max_count' in each dimension.
    """
    def __init__(self, max_count, dims):
        self.max_count = max_count
        self.dims = dims
        self.counters = np.zeros(self.dims, dtype=int)

    def __iter__(self):
        " Initilize iteration "
        self.counters = np.zeros(self.dims, dtype=int)
        self.counters[0] = -1  # Next number is [0, ... , 0]
        return self

    def __next__(self):
        """ Next array of numbers """
        return self._next_counter()

    def _next_counter(self):
        """ Get next valid counter """
        for i in range(0, self.dims):
            self.counters[i] += 1
            if self.counters[i] < self.max_count:
                return self.counters
            self.counters[i] = 0
        raise StopIteration

    def __repr__(self):
        return str(self.counters)


class CounterDimIncreasing(CounterDim):
    """
    Counts in 'dims' dimentions up to 'max_count' in each dimension, such that
    numbers in the array are in increasing order (i.e. number combinations are
    not repeated)
    """
    def __init__(self, max_count, dims):
        super().__init__(self, max_count, dims)

    def is_increasing(self):
        " Are numbers in the array in increasing order? "
        for i in range(1, self.dims):
            if self.counters[i - 1] <= self.counters[i]:
                return False
        return True

    def __next__(self):
        """ Next (valid) counter """
        while True:
            ret = self._next_counter()
            if self.is_increasing():
                return ret
