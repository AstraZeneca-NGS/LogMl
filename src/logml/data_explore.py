#!/usr/bin/env python

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import warnings

from IPython.display import display
from scipy.cluster import hierarchy as hc

from .config import CONFIG_DATASET_EXPLORE
from .files import MlFiles


# Remove some scikit warnings
warnings.filterwarnings("ignore", category=UserWarning)


class DataExplore(MlFiles):
    '''
    Perform data exploratory analysis.
    There are two types of analysis being performed here:
        1) Basic data exploration
        2) ML-based exploration
    '''

    def __init__(self, datasets_df, config, set_config=True):
        super().__init__(config, CONFIG_DATASET_EXPLORE)
        self.corr_thresdold = 0.7
        self.datasets_df = datasets_df
        self.df = self.datasets_df.dataset
        self.df_ori = self.datasets_df.dataset_ori
        self.is_use_ori = False
        self.is_summary = True
        self.is_nas = True
        self.is_plot_pairs = True
        self.is_correlation_analysis = True
        self.is_dendogram = True
        self.is_describe_all = True
        self.figsize = (20, 20)
        self.shapiro_wilks_threshold = 0.1
        if set_config:
            self._set_from_config()

    def __call__(self):
        ''' Explore dataset '''
        if not self.enable:
            self._info(f"Dataset exploration disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_EXPLORE}', enable='{self.enable}'")
            return True
        self._info("Explore data: Start")
        if self.is_use_ori:
            self.explore(self.df_ori, "Original dataset")
        self.explore(self.df, "Transformed dataset")

    def explore(self, df, name):
        # Analysis: Single variable analysis
        self._info("Explore data '{name}': End")
        print(f"Summary: {name}")
        self.summary(df)
        print(f"Missing data: {name}")
        self.nas(df)
        print(f"Describe fields: {name}")
        self.describe_all(df)
        # Analysis: Pairs of variables
        print(f"Show pair-plot: {name}")
        self.plots_pairs(df)
        self.correlation_analysis(df)
        # Analysis: Multiple variables analysis
        print(f"Dendogram: {name}")
        self.dendogram(df, name)
        # TODO: Dimmensionality reduction {PCA, LDA, tSNE, KL}
        # TODO: Remove outliers
        # TODO: Multimodal analysys
        self._info("Explore data '{name}': End")
        return True

    def correlation_analysis(self, df):
        " Correlation between all variables "
        if not self.is_correlation_analysis:
            return
        self._debug("Correlation analysis")
        corr = self.rank_correlation(df)
        # Sort and get index in correlation matrix
        ind = np.unravel_index(np.argsort(corr, axis=None), corr.shape)
        cols = self.df.columns
        # Create a dataframe of high correlated / anit-correlated variables
        df_corr = pd.DataFrame()
        add_idx = 0
        for idx in range(len(ind[0])):
            i, j = ind[0][-idx], ind[1][-idx]
            if i < j and abs(corr[i, j]) > self.corr_thresdold:
                row = pd.DataFrame({'col_i': cols[i], 'col_j': cols[j], 'i': i, 'j': j, 'corr': corr[i, j]}, index=[add_idx])
                add_idx += 1
                df_corr = pd.concat([df_corr, row], ignore_index=True)
        self.print_all("Correlations", df_corr)
        # Plot in a heatmap
        plt.figure(figsize=self.figsize)
        df_corr = pd.DataFrame(corr, columns=df.columns, index=df.columns)
        sns.heatmap(df_corr, square=True)
        plt.title('Correlation (numeric features)')
        plt.show()

    def dendogram(self, df, name):
        """
        Plot a dendogram.
        Remove columns having stdDev lower than 'std_threshold', this
        is used to avoid having 'nan' in Spearsman's correlation
        """
        if not self.is_dendogram:
            return
        corr = self.rank_correlation(df)
        corr = np.round(corr, 4)
        # Convert to distance
        dist = 1 - corr
        corr_condensed = hc.distance.squareform(dist)
        z = hc.linkage(corr_condensed, method='average')
        plt.figure(figsize=self.figsize)
        den = hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=16)
        plt.title(f"Dendogram rank correlation: {name}")
        plt.show()
        # Another method for the same
        msno.dendrogram(df)
        plt.title(f"Dendogram: {name}")
        plt.show()

    def describe_all(self, df, max_bins=100):
        " Show basic stats and histograms for every column "
        if not self.is_describe_all:
            return
        dfs = self.keep_uniq(df)
        print(f"Plotting histograms for columns: {list(dfs.columns)}")
        for c in sorted(dfs.columns):
            xi = dfs[c]
            self.describe(xi, c)
            bins = min(len(xi.unique()), max_bins)
            sns.distplot(xi, bins=bins)
            plt.title(c)
            plt.show()

    def describe(self, x, field_name):
        " Describe a single field (i.e. a single column from a dataframe) "
        df_desc = pd.DataFrame(x.describe())
        # Skewness
        df_skew = pd.DataFrame({field_name: scipy.stats.skew(x)}, index=['skewness'])
        # Kurtosis
        df_kurt = pd.DataFrame({field_name: scipy.stats.skew(x)}, index=['kurtosis'])
        # Distribution fit
        df_fit = self.distribution_fit(x, field_name)
        # Show all information
        df_desc = pd.concat([df_desc, df_skew, df_kurt, df_fit])
        self.print_all(f"Summary {field_name}", df_desc)

    def distribution_fit(self, x, field_name):
        " Check if a sample matches a distribution "
        df_norm = self.distribution_fit_normal(x, field_name)
        df_log_norm = self.distribution_fit_log_normal(x, field_name)
        return pd.concat([df_norm, df_log_norm])

    def distribution_fit_log_normal(self, x, field_name):
        """
        Check if a sample matches a Log-Normal distribution
        """
        is_log_normal, p = False, 0.0
        if np.all(x > 0):
            lx = np.log(x)
            (w, p) = scipy.stats.shapiro(lx)
            is_log_normal = (p >= self.shapiro_wilks_threshold)
        return pd.DataFrame({field_name: [is_log_normal, p]}, index=['Log_Normality', 'Log_Normality_test_pvalue'])

    def distribution_fit_normal(self, x, field_name):
        """
        Check if a sample matches a Normal distribution
        Use a Shapiro-Wilks normality test
        """
        (w, p) = scipy.stats.shapiro(x)
        is_normal = (p >= self.shapiro_wilks_threshold)
        return pd.DataFrame({field_name: [is_normal, p]}, index=['Normality', 'Normality_test_pvalue'])

    def is_numeric(self, x):
        return pd.api.types.is_numeric_dtype(x)

    def keep_uniq(self, df, min_count=10):
        " Create a new dataFrame, keep only columns having more than 'min_count' unique values "
        df_new = pd.DataFrame()
        for c in df.columns:
            xi = df[c]
            if not self.is_numeric(xi):
                continue
            if len(xi.unique()) > min_count:
                df_new[c] = xi
        return df_new

    def nas(self, df):
        " Analysis of missing values "
        if not self.is_nas:
            return
        # Number of missing values, sorted
        nas_count = df.isna().sum().sort_values(ascending=False)
        nas_perc = nas_count / len(df)
        dfnas = pd.DataFrame({'count': nas_count, 'percent': nas_perc})
        self.print_all("Missing by column", dfnas)
        # Show plot of percent of missing values
        plt.plot(nas_perc)
        plt.title("Percent of missing values")
        plt.show()
        # Missing values plots
        self.na_plots(df, "all")
        # Create a plot of missing values: Only numeric types
        if len(df.select_dtypes(include=[np.number]).columns) != len(df.columns):
            self.na_plots(df.select_dtypes(include=[np.number]), "numeric")

    def na_plots(self, df, name):
        " Plot missing values "
        # Show missing values in data frame
        msno.matrix(df)
        plt.title(f"Missing value dataFrame plot ({name})")
        plt.show()
        # Barplot of number of misisng values
        msno.bar(df)
        plt.title(f"Missing value by column ({name})")
        plt.show()
        # Heatmap: Correlation of missing values
        msno.heatmap(df)
        plt.title(f"Nullity correlation ({name})")
        plt.show()

    def numeric_non_zero_std(self, df, std_threshold=0.0):
        " Return a new dataFrame with only numeric columns having stdev > std_threshold"
        to_drop = list()
        for c in df.columns:
            if not self.is_numeric(df[c]):
                to_drop.append(c)
                continue
            stdev = df[c].std()
            if stdev <= std_threshold:
                self._debug(f"Dropping column '{c}': stdev {stdev}")
                to_drop.append(c)
        df_copy = df.drop(to_drop, axis=1) if to_drop else df
        return df_copy

    def plots_pairs(self, df):
        if not self.is_plot_pairs:
            return
        dfs = self.keep_uniq(df)
        print(f"Plotting pairs for columns: {dfs.columns}")
        sns.set_style('darkgrid')
        sns.set()
        sns.pairplot(dfs, kind='scatter', diag_kind='kde')
        plt.show()

    def print_all(self, msg, df):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(msg)
            display(df)

    def rank_correlation(self, df):
        " Rank correlation (Spearman's R)"
        # Drop columns having zero variance
        df_copy = self.numeric_non_zero_std(df)
        # Remove column names ending with '_na'
        cols_na = [c for c in df.columns if c.endswith('_na') or c.endswith('_nan')]
        df_copy.drop(cols_na, inplace=True, axis=1)
        # Calculate spearsman's correlation
        sp_r = scipy.stats.spearmanr(df_copy, nan_policy='omit')
        return sp_r.correlation

    def summary(self, df):
        " Look into basic column statistics"
        if not self.is_summary:
            return
        # Look at the data
        self.print_all("Head", df.head())
        self.print_all("Tail", df.tail())
        self.print_all("Summary statistics", df.describe())
        # Numeric / non-numeric columns
        nums = df.select_dtypes(include=[np.number]).columns
        non_nums = df.select_dtypes(include=[np.object]).columns
        print(f"Numeric columns: {sorted(list(nums))}")
        print(f"Non-numerical (e.g. categorical) columns: {sorted(list(non_nums))}")
