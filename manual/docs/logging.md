
# Logging

LogMl automatically logs a lot of information so you can retrieve, debug, and reproduce your experiments.

**Datasets**: Datasets are saved to (pickle) files after pre-processing and augmentation. The next time you run LogMl on the same dataset, LogMl checks if there is a *Processed_dataset_file* and loads it, this can save significant processing, particularly if you re-run LogMl many times (e.g. when fine-tuning a model). This can be customized by a *user_defined_function*.

**Parameters**: Parameters from *config_YAML* are stored for future references. Even if you change the original *config_YAML*, the original parameters are saved to a YAML log file.Note that  *user_defined_functions* can have parameters defined in a YAML file, this makes it convenient for repeatability, since YAML file is logged for each model.

**Models**: All models are saved to a (pickle) file. This can be customized by a *user_defined_function*.

**Model training output**: All output (STDOUT and STDERR) from model training is redirected to log files. This can be very useful to debug models.

**Plots and charts**: Plots are by default saved to the `logml_plots` directory as PNG images

**Summary data**: Summary tables and outputs are stored as CSV files for further analysis and references (e.g. feature importance tables, model training tables, correlation rankings, feature importance weights, dot graphs, etc.)
