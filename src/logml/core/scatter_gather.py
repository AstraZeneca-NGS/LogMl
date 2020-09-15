
from pathlib import Path

from .config import CONFIG_DATASET
from ..util.etc import is_int


class Scatter:
    """
    Utility class to scatter processing

    This class is only used to "decide" if we are on scatter number "n" and should execute
    that part of the processing. Also decide on "pre" and "gather" steps.

    It also helps to keep track of partial results using file backed "data shards" which
    can be later retrieved and combined during a "gather" step.

    During "gather" steps, the data will be gathered in the appropriate "data shard".
    """

    def __init__(self, config, section='logml', subsection=None):
        self.config = config
        self.split, self.split_num = config.split, config.split_num
        if self.config is not None:
            self.config_hash = config.config_hash
            self.section, self.subsection = section, subsection
        self.n = -1
        self.pre_sections = set()
        self.gather_sections = set()
        self.shards = dict()
        if not(self.split_num in ['pre', 'gather'] or is_int(self.split_num)):
            raise ValueError(f"Split number should be either 'pre', 'gather' or an int number")

    def add_pre(self, section):
        """ Add section to 'pre' """
        self.pre_sections.add(section)

    def add_gather(self, section):
        """ Add section to 'gather' """
        self.gather_sections.add(section)

    def inc(self):
        """ Increment split counter, modulo 'split_num' """
        self.n = (self.n + 1) % self.split

    def is_enabled(self):
        """ Is scatter enabled? I.e. are scatter parameters set? """
        return self.split is not None and self.split_num is not None

    def _is_gather(self):
        return self.split_num == 'gather'

    def _is_pre(self):
        return self.split_num == 'pre'

    def _is_scatter(self):
        return self.is_enabled() and self.split_num != 'pre' and self.split_num != 'gather'

    def is_scatter_n(self):
        return self.n == self.split_num

    def _file_name(self, section, subsection, _id):
        """ Create a file name for saving scatter number 'n' """
        path = self.config.get_parameters_section(CONFIG_DATASET, 'dataset_path', '.')
        path = Path(path) / f"scatter.{self.split}.{self.config_hash}"
        return self.config._get_file_name(path, section, file_type=subsection, _id=_id)

    def file_name_n(self):
        """ Create a file name for saving scatter number 'n' """
        return self._file_name(self.section, self.subsection, self.n)

    def file_name_section(self):
        """ Create a file name for saving scatter section 'section' """
        return self._file_name(self.section, None, None)

    def set_section(self, section):
        """ Set section name and reset scatter split counter """
        self.section = section
        self.n = 0

    def set_subsection(self, subsection):
        self.subsection = subsection

    def should_run(self, section=None, subsection=None):
        """ This should run only if this is the appropriate scatter number and the results file does not exists """
        if not self.is_enabled():
            return True  # All steps run in a single process when scatter is not enabled, so 'should_run() always returns 'True'
        if section is not None:
            self.set_section(section)
        if subsection is not None:
            self.set_subsection(subsection)
        self.inc()
        if not self.is_scatter_n():
            return False
        fname = self.file_name_n()
        self.config._debug(f"Scatter file: section '{self.section}', subsection '{self.subsection}', number {self.n}, file '{fname}', file exists {fname.exists()}")
        return not fname.exists()


class DataShard:
    """
    A shard of data that can be saved to a file.

    Many shards can be retrieved from files and combined into a larger shard. This is usually
    performed during the "gather" step of a "scatter / gather" processing

    A shard is implemented using a simple dictionary.
    Combining shards is just updating the dictionary while checking that key/value pairs are not overwritten.
    """
    def __init__(self):
        self.data = dict()

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
        pass

    def __len__(self):
        return len(self.data)

    def save(self):
        pass

    def __setitem__(self, key, value):
        print(f"SET ITEM: {key} = {value}")
        self.data[key] = value


