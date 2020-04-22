
from ..core.config import CONFIG_ANALYSIS
from ..core.files import MlFiles

CONFIG_ANALYSIS_GENE_SET_ENRICHMENT = 'gene_set_enrichment'


class GeneSetEnrichmnet(MlFiles):
    """
    Gene set enrichment analysis (GSEA) using GSEAPY package
    https://gseapy.readthedocs.io/en/master/index.html
    """
    def __init__(self, config, datasets, set_config=True):
        super().__init__(config, None)
        self.datasets = datasets
        config_analysis = self.config[CONFIG_ANALYSIS]
        if set_config and CONFIG_ANALYSIS_GENE_SET_ENRICHMENT in config_analysis:
            self._set_from_dict(config_analysis[CONFIG_ANALYSIS_GENE_SET_ENRICHMENT])

    def __call__(self):
        """ Perform the analysis """
        if not self.enable:
            self._debug(f"Analysis '{type(self).__name__}' disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_ANALYSIS}', sub-section '{CONFIG_ANALYSIS_GENE_SET_ENRICHMENT}', enable='{self.enable}'")
            return True
        self._info(f"Analysis: Start")
        self._info(f"Analysis: End")
        return True
