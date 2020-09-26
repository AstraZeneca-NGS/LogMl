
import pickle

from pathlib import Path

from .config import CONFIG_DATASET
from ..util.etc import is_int


class ScatterCounter:
    """
    This class is only used to "decide" if we are on scatter number "n" and should execute that part of the processing.
    """
    def __init__(self, total, num):
        self.total, self.num = total, num
        self.n = -1
        if not(self.num in ['pre', 'gather'] or is_int(self.num)):
            raise ValueError(f"Split number should be either 'pre', 'gather' or an int number")

    def __bool__(self):
        """
        Should we execute this split number?
        Execute all if splitting disabled, execute split 'num' if enabled
        Return 'is_scatter_n' if enabled, always true if disabled
        """
        return self.is_scatter_num() if self.is_enabled() else True

    def inc(self):
        """ Increment split counter, modulo 'split_num' """
        self.n = (self.n + 1) % self.total

    def is_enabled(self):
        """ Is scatter enabled? I.e. are scatter parameters set? """
        return self.total is not None and self.num is not None

    def is_gather(self):
        return self.num == 'gather'

    def is_pre(self):
        return self.num == 'pre'

    def is_scatter(self):
        return self.is_enabled() and self.num != 'pre' and self.num != 'gather'

    def is_scatter_num(self):
        return self.n == self.num


class Scatter:
    """
    Utility class to scatter processing

    It also helps to keep track of partial results using file backed "data shards" which
    can be later retrieved and combined during a "gather" step.

    During "gather" steps, the data will be gathered in the appropriate "data shard".

    This object is a context manager that returns a data shard or None, depending on
    whether se section should be executed or not

    if scatter(section, subsection):
        with scatter as shard:
            shard[name] = results
    """

    def __init__(self, config, section='logml', subsection=None):
        self.config = config
        self.config_hash = config.config_hash
        self.section, self.subsection = section, subsection
        self.scatter_counter = ScatterCounter(config.scatter_total, config.scatter_num)
        self.pre_sections = set()
        self.gather_sections = set()
        self.shards = dict()

    def add_pre(self, section):
        """ Add section to 'pre' """
        self.pre_sections.add(section)

    def add_gather(self, section):
        """ Add section to 'gather' """
        self.gather_sections.add(section)

    def __call__(self, section=None, subsection=None):
        """
        Check if a scatter should run
        For the scatter to run, it should:
            - if scatter is disabled, always run
            - Be the appropriate scatter number (i.e. self.scatter_counter evaluates to True)
            - And the corresponding shard file does not exist
        """
        self.set(section, subsection)
        self.scatter_counter.inc()
        if self.scatter_counter:
            sfile = self.shard_file()
            self.config._debug(f"Scatter: section '{self.section}', subsection '{self.subsection}', number {self.scatter_counter.n}, file '{sfile}', file exists {sfile.exists()}")
            return not sfile.exists()
        return False

    def __enter__(self):
        """ Enter context manager, create a new shard """
        sfname = self.shard_file()
        self.shard = DataShard(sfname)
        self.config._debug(f"Scatter: Enter context manager. Shard file '{self.shard.file_name}'")
        return self.shard

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit context manager, save current shard
        Return True, so any exceptions will by thrown
        """
        self.config._debug(f"Scatter: Exit context manager. Shard file '{self.shard.file_name}'")
        self.shard.save()
        return True

    def shard_file(self):
        """ Create a file name for a data shard """
        path = self.config.get_parameters_section(CONFIG_DATASET, 'dataset_path', '.')
        path = Path(path) / f"scatter.{self.scatter_counter.num}.{self.config_hash}"
        return self.config.get_file_path(path, self.section, file_type=self.subsection, _id=self.scatter_counter.n)

    def set(self, section, subsection=None):
        if section:
            self.section = section
        if subsection:
            self.subsection = subsection


class DataShard:
    """
    A shard of data that can be saved to a file.

    Many shards can be retrieved from files and combined into a larger shard. This is usually
    performed during the "gather" step of a "scatter / gather" processing

    A shard is implemented using a simple dictionary.
    Combining shards is just updating the dictionary while checking that key/value pairs are not overwritten.
    """
    def __init__(self, file_name):
        self.data = dict()
        self.file_name = file_name

    def __getitem__(self, key):
        print(f"GET ITEM: {key}")
        return self.data[key]

    def __contains__(self, item):
        return self.data.__contains__(item)

    def __delattr__(self, item):
        del self.data[item]

    def __iter__(self, item):
        return iter(self.data)

    def load(self):
        """
        Load data from file, update all dictionary entries
        Raise ValueError if a key already exists in the current shard (i.e. do not overwrite values)
        """
        with open(self.file_name, 'rb') as input:
            data = pickle.load(input)
            for k, v in data.items():
                if k in self.data:
                    raise ValueError(f"Key '{k}' already exists in shard, file: '{self.file_name}'")
                self.data[k] = data[k]

    def __len__(self):
        return len(self.data)

    def save(self):
        with open(self.file_name, 'wb') as output:
            pickle.dump(self.data, output)

    def __setitem__(self, key, value):
        print(f"SET ITEM: {key} = {value}")
        self.data[key] = value


