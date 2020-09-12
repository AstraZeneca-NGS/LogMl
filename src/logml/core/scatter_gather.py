
from pathlib import Path

from .config import CONFIG_DATASET


class Scatter:
    """
    Utility class to scatter processing
    Note: This class is only used to "decide" if we are on scatter number "n" and should execute that part of the processing
    """

    def __init__(self, config, section='logml', subsection=None):
        self.config = config
        self.split, self.split_num = config.split, config.split_num
        if self.config is not None:
            self.config_hash = config.config_hash
            self.section, self.subsection = section, subsection
        self.n = -1

    def inc(self):
        """ Increment split counter, modulo 'split_num' """
        self.n = (self.n + 1) % self.split

    def is_enabled(self):
        """ Is scatter enabled? I.e. are scatter parameters set? """
        return self.split is not None and self.split_num is not None

    def is_scatter_n(self):
        return self.n == self.split_num

    def file_name(self):
        """ Create a file name for saving scatter number 'n' """
        path = self.config.get_parameters_section(CONFIG_DATASET, 'dataset_path', '.')
        path = Path(path) / f"scatter.{self.split}.{self.config_hash}"
        return self.config._get_file_name(path, self.section, file_type=self.subsection, _id=self.n)

    def set_section(self, section, subsection=None):
        self.section, self.subsection = section, subsection

    def set_subsection(self, subsection):
        self.subsection = subsection

    def should_run(self):
        """ This should run only if this is the appropriate scatter number and the results file does not exists """
        if not self.is_enabled():
            return True  # All steps run in a single process when scatter is not enabled, so 'should_run() always returns 'True'
        self.inc()
        if not self.is_scatter_n():
            return False
        fname = self.file_name()
        self.config._debug(f"Scatter section: '{self.section}', subsection: '{self.subsection}', number {self.n}, file '{fname}', file exists {fname.exists()}")
        return not fname.exists()
