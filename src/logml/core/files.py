
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
DISABLE_PLOTS = False
SHOW_PLOTS = True
SAVE_PLOTS = True
PLOTS_PATH = 'logml_plots'


def set_plots(disable=None, show=None, save=None, path=None):
    global DISABLE_PLOTS, SHOW_PLOTS, SAVE_PLOTS, PLOTS_PATH
    if disable is not None:
        DISABLE_PLOTS = disable
    if show is not None:
        SHOW_PLOTS = show
    if save is not None:
        SAVE_PLOTS = save
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

    def _plot_show(self, title, section, figure=None):
        ''' Show a plot in a way that we can continue processing '''
        figure = figure if figure else 'all'
        if DISABLE_PLOTS:
            plt.close(figure)
            return
        plt.title(title)
        if SAVE_PLOTS:
            file = self._get_file_name(PLOTS_PATH, name=section, file_type=sanitize_name(title), ext='png', _id=None)
            self._debug(f"Saving plot '{title}' to '{file}'")
            plt.savefig(file)
        if SHOW_PLOTS:
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
