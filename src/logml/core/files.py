
import inspect
import logging
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import sys
import yaml

from IPython.core.display import display
from yamlinclude import YamlIncludeConstructor

from .log import MlLog
from ..util.sanitize import sanitize_name

# Default plot preferences
PLOTS_ADJUSTMENT_FACTOR_COUNT_VARS = 50
PLOTS_DISABLE = False
PLOTS_PATH = 'logml_plots'
PLOTS_FIGSIZE_X = 16    # Figure size (inches), x axis
PLOTS_FIGSIZE_Y = 10    # Figure size (inches), y axis
PLOTS_SAVE = True
PLOTS_SHOW = True


def set_plots(disable=None, show=None, save=None, path=None):
    global PLOTS_DISABLE, PLOTS_SHOW, PLOTS_SAVE, PLOTS_PATH
    if disable is not None:
        PLOTS_DISABLE = disable
    if show is not None:
        PLOTS_SHOW = show
    if save is not None:
        PLOTS_SAVE = save
    if path is not None:
        PLOTS_PATH = path


class MlFiles(MlLog):
    '''
    ML Files: Basic file load / save capabilities
    '''
    def __init__(self, config=None, config_section=None):
        super().__init__(config, config_section)

    def _display(self, obj):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            display(obj)

    def _get_file_name(self, path, name, file_type=None, ext='pkl', _id=None):
        ''' Create a file name
        Make sure all intermediate directories exists, so
        that the file can be created
        '''
        self._debug(f"path={path}, name={name}, file_type={file_type}, ext='{ext}'")
        if not path or not name:
            return None
        os.makedirs(path, exist_ok=True)
        fname = os.path.join(path, name)
        if file_type:
            fname = f"{fname}.{file_type}"
        if _id:
            fname = f"{fname}.{_id}"
        if ext:
            fname = f"{fname}.{ext}"
        return fname

    def _load_pickle(self, file_pickle, tag):
        ''' Load a pickle file, return data (on success) or None (on failure) '''
        if not file_pickle:
            self._debug(f"{tag}: Empty file name, skipping")
            return None
        if not os.path.isfile(file_pickle):
            self._debug(f"{tag}: File '{file_pickle}' not found, skipping")
            return None
        self._info(f"{tag}: Loading pickle file '{file_pickle}'")
        with open(file_pickle, 'rb') as input:
            dataset = pickle.load(input)
            return dataset

    def _load_yaml(self, file_yaml):
        ''' Load a yaml file '''
        yaml_dir = os.path.dirname(file_yaml)
        self._debug(f"Loading YAML file '{file_yaml}', base dir '{yaml_dir}'")
        YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=yaml_dir)
        with open(file_yaml) as yaml_file:
            return yaml.load(yaml_file, Loader=yaml.FullLoader)

    def _plot_show(self, title, section, figure=None, count_vars_x=None, count_vars_y=None):
        '''
        Show a plot in a way that we can continue processing
            count_vars_x: Number of variables plotted along the x-axis, the size of the figure is adjusted (increased) by `max(1.0, count_vars_x / PLOTS_ADJUSTMENT_FACTOR_COUNT_VARS)`
            count_vars_y: Number of variables plotted along the y-axis
        '''
        # Adjust figure size
        fig = plt.gcf()
        figsize_x = PLOTS_FIGSIZE_X if count_vars_x is None else PLOTS_FIGSIZE_X * max(1.0, count_vars_x / PLOTS_ADJUSTMENT_FACTOR_COUNT_VARS)
        figsize_y = PLOTS_FIGSIZE_Y if count_vars_y is None else PLOTS_FIGSIZE_Y * max(1.0, count_vars_y / PLOTS_ADJUSTMENT_FACTOR_COUNT_VARS)
        fig.set_size_inches(figsize_x, figsize_y)
        figure = figure if figure else 'all'
        if PLOTS_DISABLE:
            plt.close(figure)
            return
        plt.title(title)
        if PLOTS_SAVE:
            file = self._get_file_name(PLOTS_PATH, name=section, file_type=sanitize_name(title), ext='png', _id=None)
            self._debug(f"Saving plot '{title}' to '{file}'")
            plt.savefig(file)
        if PLOTS_SHOW:
            plt.draw()  # Show plot in a non-blocking maner
            plt.pause(0.1)  # Pause, so that GUI can update the images
        else:
            plt.close(figure)

    def _save_csv(self, file_csv, tag, df, save_index=False):
        ''' Save a dataFrame to a CSV file, return True (on success) or False (on failure) '''
        if not file_csv:
            self._debug(f"{tag}: Empty file name, skipping")
            return False
        if df is None:
            self._debug(f"{tag}: DataFrame is None, skipping")
            return False
        self._debug(f"{tag}: Saving CSV file '{file_csv}'")
        df.to_csv(file_csv, index=save_index)

    def _save_pickle(self, file_pickle, tag, data):
        ''' Save a pickle file, return True (on success) or False (on failure) '''
        if not file_pickle:
            self._debug(f"{tag}: Empty file name, skipping")
            return False
        self._debug(f"{tag}: Saving pickle file '{file_pickle}'")
        with open(file_pickle, 'wb') as output:
            dataset = pickle.dump(data, output)
            return True

    def _save_yaml(self, file_name, data):
        ''' Save data to YAML file '''
        with open(file_name, 'w') as out:
            yaml.dump(data, out, default_flow_style=False)
