
import pickle

from pathlib import Path

from .config import CONFIG_DATASET
from ..util.etc import is_int


scatter_gather = None


def init_scatter_gather(split_num=0, total=-1, pre=False, gather=False, data_path='.'):
    global scatter_gather
    scatter_gather = ScatterGather(split_num=0, total=-1, pre=False, gather=False, data_path='.')


class ScatterGather:
    """
    Scatter & Gather

    Processing is scattered into 'split_total' number of processes. This class helps to
    provides tracking of the scatter & gather processing, as well as caching of results

    The workflow for scatter & gather is:
        - One 'pre' process is executed, which takes care of initialization steps.

        - A number of 'split_total' processes, are exectured, each one initialized with a
          different 'split_num'. Each of these processes executes one part (1 / split_total) of
          the processing. The results are saved to (cache) pickle files.
          Usually several scatter processes are executed in parallel, e.g. in a large server or a cluster.

        - One 'gather' process is executed. This process loads the results from each 'scatter'
          (previous step) and performs some processing with all the data.

    Note: The implicit assumption is that overall the execution times are similar. If one
    scatter process takes much longer than others, spreading processing this way will not be efficient.

    Note: When the scatter & gather processing is disabled, all parts are always executed. No caching,
    loading or saving is performed (i.e. is equivalent to not having any scatter & gather at all)

    Parameters:
        split_num: Split number. During a scatter, only one out of split_total function calls will be invoked, see 'should_run()'
        split_total: Number of processes. -1 means that scatter & gather is disabled
        pre: Set 'pre' mode
        gather: Set 'gather' mode
        data_path: Path for saving cached results
    """
    def __init__(self, split_num=0, split_total=-1, pre=False, gather=False, data_path='.'):
        self.pre = pre
        self.gather = gather
        self.count = -1
        self.split_num = split_num
        self.total = split_total
        self.data_path = Path(data_path)
        if not self.is_disabled() and self.split_num >= self.total:
            raise ValueError(
                f"ScatterGather: 'split_num' ({self.split_num}) cannot be higher than 'total' ({self.total})")
        if pre and gather:
            raise ValueError(f"ScatterGather: 'pre' and 'gather' cannot be True at the same time")

    def file(self, state):
        """ File for saving (caching) results """
        return self.data_path / f"{state}_{self.count}_{self.total}.pkl"

    def is_disabled(self):
        """ Is scatter & gather disabled? """
        return self.total <= 1

    def inc(self):
        """ Increment counter """
        self.count += 1

    def load(self, state='scatter'):
        """
        Load data from cache file
        """
        if self.is_disabled():
            raise ValueError("Attempt to load in disabled 'ScatterGather'")
        fn = self.file(state)
        print(f"Loading data from '{fn}'")
        with open(fn, 'rb') as input:
            return pickle.load(input)

    def save(self, data, state='scatter'):
        """
        Save data to cache file
        """
        if self.is_disabled():
            raise ValueError("Attempt to load in disabled 'ScatterGather'")
        fn = self.file(state)
        print(f"Saving data to '{fn}'")
        with open(fn, 'wb') as output:
            pickle.dump(data, output)

    def should_run(self):
        """ Should this scatter method be executed? """
        return self.count % self.total == self.split_num

    def __repr__(self):
        if self.pre:
            return "ScatterGather(pre=True)"
        if self.gather:
            return "ScatterGather(gather=True)"
        return f"ScatterGather(count={self.count}, split_num={self.split_num}, total={self.total})"


def gather(g):
    """
    Methods annotated as '@gather' are only executed in the 'gather' stage of a scatter/gather
    """
    def scatter_gather_wrapper(self, *args, **kwargs):
        ret = None
        scatter_gather.inc()
        if scatter_gather.is_disabled():
            ret = g(self, *args, **kwargs)
        elif scatter_gather.gather:
            print(f"ScatteGather: {scatter_gather}, executing '{g.__name__}'")
            ret = g(self, *args, **kwargs)
        else:
            print(f"ScatteGather: {scatter_gather}, skipping '{g.__name__}'")
        return ret

    return scatter_gather_wrapper


def pre(g):
    """
    Methods annotated as '@pre' are executed in the 'pre' stage of a scatter/gather
    Otherwise the data is loaded from a (cached) result
    """
    def scatter_gather_wrapper(self, *args, **kwargs):
        ret = None
        scatter_gather.inc()
        if scatter_gather.is_disabled():
            ret = g(self, *args, **kwargs)
        elif scatter_gather.pre:
            print(f"ScatteGather: {scatter_gather}, executing '{g.__name__}'")
            ret = g(self, *args, **kwargs)
            scatter_gather.save(ret, state='pre')
        else:
            print(f"ScatteGather: {scatter_gather}, loading '{g.__name__}'")
            ret = scatter_gather.load(state='pre')
        return ret

    return scatter_gather_wrapper


def scatter(g):
    """
    Methods annotated with '@scatter' are executed in the 'scatter' stage of a scatter/gather
    Otherwise the data is loaded from a (cached) result
    """
    def scatter_gather_wrapper(self, *args, **kwargs):
        ret = None
        scatter_gather.inc()
        if scatter_gather.is_disabled():
            ret = g(self, *args, **kwargs)
        elif scatter_gather.pre:
            print(f"ScatteGather: {scatter_gather}, skipping '{g.__name__}'")
        elif scatter_gather.gather:
            print(f"ScatteGather: {scatter_gather}, loading '{g.__name__}'")
            ret = scatter_gather.load()
        elif scatter_gather.should_run():
            print(f"ScatteGather: {scatter_gather}, executing '{g.__name__}'")
            ret = g(self, *args, **kwargs)
            scatter_gather.save(ret)
        else:
            print(f"ScatteGather: {scatter_gather}, skipping '{g.__name__}'")
        return ret

    return scatter_gather_wrapper
