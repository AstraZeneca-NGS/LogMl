
from .gene_set_enrichment import GeneSetEnrichmnet
from ..core.config import CONFIG_ANALYSIS
from ..core.files import MlFiles
from ..core.scatter_gather import scatter


class AnalysisDf(MlFiles):
    """ Perform several data analyses """

    def __init__(self, config, datasets, set_config=True):
        super().__init__(config, CONFIG_ANALYSIS)
        self.datasets = datasets
        self.gene_set_enrichment = dict()
        if set_config:
            self._set_from_config()
        self.config_analysis = self.config[CONFIG_ANALYSIS]

    def __call__(self):
        """ Analyses """
        if not self.enable:
            self._debug(f"Analysis disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_ANALYSIS}', enable='{self.enable}'")
            return True
        self._info(f"Analysis: Start")
        self.gse()
        self._info(f"Analysis: End")
        return True

    @scatter
    def gse(self):
        gse = GeneSetEnrichmnet(self.config, self.datasets)
        gse()
