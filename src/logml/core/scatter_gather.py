
import pickle
import shutil

from pathlib import Path

from ..util.etc import is_int
from .log import MlLogMessages


scatter_gather = None


def init_scatter_gather(scatter_num, scatter_total, data_path='.', force=True):
    """
    Initialize ScatterGather object
    Parameters:
        force: Force initialization even if a ScatterGather object has already been initialized
        other parameters: Passed to ScatterGather constructor
    """
    global scatter_gather
    pre, gather = False, False
    if not is_int(scatter_num):
        pre = (scatter_num == 'pre')
        gather = (scatter_num == 'gather')
        scatter_num = 0
    if scatter_gather is None or scatter_gather.scatter_num != scatter_num or scatter_gather.pre != pre or force:
        scatter_gather = ScatterGather(scatter_num=scatter_num, scatter_total=scatter_total, pre=pre, gather=gather, data_path=data_path)


class ScatterGather(MlLogMessages):
    """
    Scatter & Gather

    Processing is scattered into 'scatter_total' number of processes. This class helps to
    provides tracking of the scatter & gather processing, as well as caching of results

    The workflow for scatter & gather is:
        - One 'pre' process is executed, which takes care of initialization steps.

        - A number of 'scatter_total' processes, are exectured, each one initialized with a
          different 'scatter_num'. Each of these processes executes one part (1 / scatter_total) of
          the processing. The results are saved to (cache) pickle files.
          Usually several scatter processes are executed in parallel, e.g. in a large server or a cluster.

        - One 'gather' process is executed. This process loads the results from each 'scatter'
          (previous step) and performs some processing with all the data.

    Note: The implicit assumption is that overall the execution times are similar. If one
    scatter process takes much longer than others, spreading processing this way will not be efficient.

    Note: When the scatter & gather processing is disabled, all parts are always executed. No caching,
    loading or saving is performed (i.e. is equivalent to not having any scatter & gather at all)

    Parameters:
        scatter_num: Split number. During a scatter, only one out of scatter_total function calls will be invoked, see 'should_run()'
        scatter_total: Number of processes. -1 means that scatter & gather is disabled
        pre: Set 'pre' mode
        gather: Set 'gather' mode
        data_path: Path for saving cached results
    """
    def __init__(self, scatter_num=0, scatter_total=-1, pre=False, gather=False, data_path='.'):
        self.pre = pre
        self.gather = gather
        self.count = -1
        self.scatter_num = scatter_num
        self.total = scatter_total
        self.data_path = Path(data_path)
        if not self.is_disabled() and self.scatter_num >= self.total:
            raise ValueError(f"ScatterGather: 'scatter_num' ({self.scatter_num}) cannot be higher than 'total' ({self.total})")
        if pre and gather:
            raise ValueError(f"ScatterGather: 'pre' and 'gather' cannot be True at the same time")
        self._debug(f"Scatter gather: scatter_num={self.scatter_num}, total={self.total},  pre={pre}, gather={gather}, data_path: '{self.data_path}'")

    def file(self, state):
        """ File for saving (caching) results """
        self.data_path.mkdir(parents=True, exist_ok=True)
        return self.data_path / f"{state}_{self.total}_{self.count}.pkl"

    @staticmethod
    def remove_scatter_folder(folder_path):
        shutil.rmtree(folder_path)

    def is_disabled(self):
        """ Is scatter & gather disabled? """
        return self.total <= 1

    def inc(self):
        """ Increment counter """
        self.count += 1

    def load(self, object, method, state='scatter'):
        """
        Load data from cache file
        """
        if self.is_disabled():
            raise ValueError("Attempt to load in disabled 'ScatterGather'")
        fn = self.file(state)
        method_name = f"{type(object).__name__}.{method.__name__}"
        self._debug(f"Loading {method_name} results from '{fn}'")
        try:
            with open(fn, 'rb') as input:
                data_method = pickle.load(input)
        except FileNotFoundError:
            raise ValueError(f"Scatter & Gather error: Cache file '{fn}' is not exist")
        (data, method_name_ori) = data_method
        # Check that the data is loaded from the same class.method that was originally saved from
        if method_name != method_name_ori:
            raise ValueError(f"Scatter & Gather error: The data in file '{fn}' was saved from method '{method_name_ori}', but it's being retrieved from method '{method_name}'")
        return data

    def save(self, data, object, method, state='scatter'):
        """
        Save data to cache file
        """
        if self.is_disabled():
            raise ValueError("Attempt to load in disabled 'ScatterGather'")
        fn = self.file(state)
        method_name = f"{type(object).__name__}.{method.__name__}"
        data_method = (data, method_name)
        self._debug(f"Saving {method_name} results to '{fn}'")
        with open(fn, 'wb') as output:
            pickle.dump(data_method, output)

    def should_run(self):
        """ Should this scatter method be executed? """
        ok = self.count % self.total == self.scatter_num
        if ok:
            fn = self.file('scatter')
            return not fn.exists()
        return False

    def __repr__(self):
        if self.pre:
            return f"ScatterGather(pre=True, count={self.count}, scatter_num={self.scatter_num}, total={self.total})"
        if self.gather:
            return f"ScatterGather(gather=True, count={self.count}, scatter_num={self.scatter_num}, total={self.total})"
        return f"ScatterGather(count={self.count}, scatter_num={self.scatter_num}, total={self.total})"


def gather(g):
    """
    Methods annotated as '@gather' are only executed in the 'gather' stage of a scatter/gather
    """
    def gather_wrapper(self, *args, **kwargs):
        ret = None
        if scatter_gather is not None:
            scatter_gather.inc()
        if scatter_gather is None or scatter_gather.is_disabled():
            ret = g(self, *args, **kwargs)
        elif scatter_gather.gather:
            scatter_gather._debug(f"ScatteGather, gather: {scatter_gather}, executing '{g.__name__}'")
            ret = g(self, *args, **kwargs)
        else:
            scatter_gather._debug(f"ScatteGather, gather: {scatter_gather}, skipping '{g.__name__}'")
        return ret

    return gather_wrapper


def pre(g):
    """
    Methods annotated as '@pre' are executed in the 'pre' stage of a scatter/gather
    Otherwise the data is loaded from a (cached) result
    """
    def pre_wrapper(self, *args, **kwargs):
        if scatter_gather is not None:
            scatter_gather.inc()
        if scatter_gather is None or scatter_gather.is_disabled():
            ret = g(self, *args, **kwargs)
        elif scatter_gather.pre:
            scatter_gather._debug(f"ScatteGather, pre: {scatter_gather}, executing '{g.__name__}'")
            ret = g(self, *args, **kwargs)
            scatter_gather.save(ret, self, g, state='pre')
        else:
            scatter_gather._debug(f"ScatteGather, pre: {scatter_gather}, loading '{g.__name__}'")
            ret = scatter_gather.load(self, g, state='pre')
        return ret

    return pre_wrapper


def scatter(g):
    """
    Methods annotated with '@scatter' are executed in the 'scatter' stage of a scatter/gather
    Otherwise the data is loaded from a (cached) result
    """
    def scatter_wrapper(self, *args, **kwargs):
        ret = None
        if scatter_gather is not None:
            scatter_gather.inc()
        if scatter_gather is None or scatter_gather.is_disabled():
            ret = g(self, *args, **kwargs)
        elif scatter_gather.pre:
            scatter_gather._debug(f"ScatteGather, scatter: {scatter_gather}, skipping '{g.__name__}'")
        elif scatter_gather.gather:
            scatter_gather._debug(f"ScatteGather, scatter: {scatter_gather}, loading '{g.__name__}'")
            ret = scatter_gather.load(self, g)
        elif scatter_gather.should_run():
            scatter_gather._debug(f"ScatteGather, scatter: {scatter_gather}, executing '{g.__name__}'")
            ret = g(self, *args, **kwargs)
            scatter_gather.save(ret, self, g)
        else:
            scatter_gather._debug(f"ScatteGather, scatter: {scatter_gather}, skipping '{g.__name__}'")
        return ret

    return scatter_wrapper


def scatter_all(g):
    """
    Methods annotated with '@scatter_all' are executed in all 'scatter' steps.
    """
    def scatter_all_wrapper(self, *args, **kwargs):
        ret = None
        if scatter_gather is not None:
            scatter_gather.inc()
        if scatter_gather is None or scatter_gather.is_disabled():
            ret = g(self, *args, **kwargs)
        elif scatter_gather.pre:
            scatter_gather._debug(f"ScatteGather, scatter_all: {scatter_gather}, skipping '{g.__name__}'")
        elif scatter_gather.gather:
            scatter_gather._debug(f"ScatteGather, scatter_all: {scatter_gather}, loading '{g.__name__}'")
            ret = scatter_gather.load(self, g)
        else:
            scatter_gather._debug(f"ScatteGather, scatter_all: {scatter_gather}, executing '{g.__name__}'")
            ret = g(self, *args, **kwargs)
            scatter_gather.save(ret, self, g)
        return ret

    return scatter_all_wrapper
