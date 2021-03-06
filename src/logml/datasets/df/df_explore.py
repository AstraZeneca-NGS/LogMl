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

from ...core.config import CONFIG_DATASET_EXPLORE
from ...core.files import MlFiles
from ...core.scatter_gather import init_scatter_gather, pre, scatter, gather
from ...util.results_df import ResultsDf


# Remove some scikit warnings
warnings.filterwarnings("ignore", category=UserWarning)


class DfExplore(MlFiles):
    """
    Perform data exploratory analysis.
    There are two types of analysis being performed here:
    """

    def __init__(self, df, name, config, files_base, set_config=True):
        super().__init__(config, CONFIG_DATASET_EXPLORE)
        self.corr_thresdld = 0.7
        self.correlation_analysis_max = 100
        self.dendogram_max = 100
        self.describe_kde_min_uniq_values = 100
        self.df = df
        self.files_base = files_base
        self.is_dendogram = True
        self.is_describe_all = True
        self.is_correlation_analysis = True
        self.is_nas = True
        self.is_plot_pairs = True
        self.is_summary = True
        self.is_use_ori = True
        self.name = name
        self.plot_pairs_max = 20
        self.rank_correlation_matrix = None
        self.rank_correlation_colums = None
        self.shapiro_wilks_threshold = 0.05
        if set_config:
            self._set_from_config()

    def __call__(self):
        """ Explore dataset """
        if not self.enable:
            self._debug(f"Dataset explore ({self.name}) disabled, skipping. Config file '{self.config.config_file}', section '{CONFIG_DATASET_EXPLORE}', enable='{self.enable}'")
            return True
        self.explore()
        return True

    def correlation_analysis(self):
        """ Correlation between all variables """
        if not self.is_correlation_analysis:
            return
        self._debug(f"Correlation analysis: {self.name}")
        if len(self.df.columns) > self.correlation_analysis_max:
            self._debug(f"Correlation analysis {self.name}: Too many columns to compare, {len(self.df.columns)} > correlation_analysis_max ({self.correlation_analysis_max}), skipping")
            return
        corr, cols = self.rank_correlation()
        if corr is None:
            self._error(f"Correlation analysis {self.name}: Could not calculate correlation")
            return
        # Sort and get index in correlation matrix
        ind = np.unravel_index(np.argsort(corr, axis=None), corr.shape)
        # Create a dataframe of high correlated / annti-correlated variables
        self.top_correlations = ResultsDf()
        add_idx = 0
        for idx in range(len(ind[0])):
            i, j = ind[0][-idx], ind[1][-idx]
            if i < j and abs(corr[i, j]) > self.corr_thresdld:
                row = pd.DataFrame({'col_i': cols[i], 'col_j': cols[j], 'i': i, 'j': j, 'corr': corr[i, j]}, index=[add_idx])
                add_idx += 1
                self.top_correlations.add_row_df(row)
        tcdf = self.top_correlations.df
        if tcdf is not None:
            self.top_correlations.print(f"Top correlations {self.name}: {tcdf.shape}  {tcdf.shape[0]}")
            self._save_csv(f"{self.files_base}.top_correlations.csv", "Correlation analysis (top correlations)", tcdf, save_index=True)
        else:
            print(f"Top correlations {self.name}: There are no variables correlated over corr_thresdld={self.corr_thresdld}")
        # Plot in a heatmap
        plt.figure()
        self.correlation_df = pd.DataFrame(corr, columns=cols, index=cols)
        self._save_csv(f"{self.files_base}.correlations_matrix.csv", "Correlation matrix", self.correlation_df, save_index=True)
        sns.heatmap(self.correlation_df, square=True)
        num_vars = len(self.correlation_df.columns)
        self._plot_show(f'Correlation (numeric features)', f'dataset_explore.{self.name}', count_vars_x=num_vars, count_vars_y=num_vars)

    def dendogram(self):
        """
        Plot a dendogram.
        Remove columns having stdDev lower than 'std_threshold', this
        is used to avoid having 'nan' in Spearsman's correlation
        """
        if not self.is_dendogram:
            return
        if len(self.df.columns) > self.dendogram_max:
            self._debug(f"Dendogram {self.name}: Too many columns to compare ({len(self.df.columns)}), skipping")
            return
        corr, cols = self.rank_correlation()
        if corr is None:
            self._error(f"Dendogram {self.name}: Could not calculate correlation")
            return
        corr = np.round(corr, 4)
        # Convert to distance
        dist = 1 - corr
        corr_condensed = hc.distance.squareform(dist)
        z = hc.linkage(corr_condensed, method='average')
        plt.figure()
        den = hc.dendrogram(z, labels=cols, orientation='left', leaf_font_size=16)
        num_vars = len(cols)
        self._plot_show(f"Dendogram rank correlation", f'dataset_explore.{self.name}', count_vars_x=num_vars, count_vars_y=num_vars)

    def dendogram_na(self):
        """ Dendogram of missing values """
        count_na = self.df.isna().sum().sum()
        if count_na <= 0:
            self._debug(f"Dendogram of missing values {self.name}: No missing values, skipping")
            return
        msno.dendrogram(self.df)
        num_vars = len(self.df.columns)
        self._plot_show(f"Dendogram missing values", f'dataset_explore.{self.name}', count_vars_x=num_vars)

    def describe_all(self, max_bins=100):
        """ Show basic stats and histograms for every column """
        if not self.is_describe_all:
            return
        dfs = self.remove_non_numeric_cols(self.df)
        print(f"Plotting histograms for columns {self.name}: {list(dfs.columns)}")
        descr = ResultsDf()
        for c in sorted(dfs.columns):
            xi = dfs[c]
            xi_no_na = xi[~np.isnan(xi)]  # Remove 'nan'
            df_desc = self.describe(xi_no_na, c)
            descr.add_df(df_desc)
            count_uniq = len(xi_no_na.unique())
            bins = min(count_uniq, max_bins)
            fig = plt.figure()
            # Create histogram. Only show 'kernel density estimate' if there are more than 'self.describe_min_kde_count' unique values
            show_kde = (count_uniq >= self.describe_kde_min_uniq_values)
            if bins > 0:
                try:
                    sns.distplot(xi_no_na, kde=show_kde, bins=bins)
                    self._plot_show(f"Distribution {c}", f'dataset_explore.{self.name}', fig)
                except Exception as e:
                    self._error(f"Exception '{e}', when plotting variable '{c}': {xi_no_na}")
        descr_csv = f"{self.files_base}.describe_features.csv"
        descr.print(f"Describe variables {self.name}, saving to file '{descr_csv}'")
        self._save_csv(descr_csv, "Describe Features", descr.df, save_index=True)

    def describe(self, x, field_name):
        " Describe a single field (i.e. a single column from a dataframe) "
        df_desc = pd.DataFrame(x.describe())
        # Number of unique values
        df_uniq = pd.DataFrame({field_name: len(x.unique())}, index=['unique'])
        # Skewness
        df_skew = pd.DataFrame({field_name: scipy.stats.skew(x)}, index=['skewness'])
        # Kurtosis
        df_kurt = pd.DataFrame({field_name: scipy.stats.kurtosis(x)}, index=['kurtosis'])
        # Distribution fit
        df_fit = self.distribution_fit(x, field_name)
        # Show all information (filter out None values first)
        dfs = [df_desc, df_uniq, df_skew, df_kurt, df_fit]
        dfs = [df for df in dfs if df is not None]
        df_desc = pd.concat([df_desc, df_uniq, df_skew, df_kurt, df_fit])
        self.print_all(f"Summary {self.name}: {field_name}", df_desc)
        return df_desc

    def distribution_fit(self, x, field_name):
        " Check if a sample matches a distribution "
        if len(x) < 3:
            return None
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

    @pre
    def explore(self):
        """ Explore dataFrame"""
        if self.df is None:
            self._debug(f"Explore data '{self.name}': DataFrame is None, skipping.")
            return
        self._info(f"Explore data '{self.name}': Start")
        print(f"Summary: {self.name}")
        self.summary()
        print(f"Missing data: {self.name}")
        self.nas()
        print(f"Describe fields: {self.name}")
        self.describe_all()
        # Analysis: Pairs of variables
        print(f"Show pair-plot: {self.name}")
        self.plots_pairs()
        print(f"Correlation analysis: {self.name}")
        self.correlation_analysis()
        print(f"Dendogram: {self.name}")
        self.dendogram()
        print(f"Dendogram of missing values: {self.name}")
        self.dendogram_na()
        # TODO: Dimmensionality reduction {PCA, LDA, tSNE, KL}
        # TODO: Remove outliers
        # TODO: Multimodal analysys
        self.save()
        self._info(f"Explore data '{self.name}': End")
        return True

    def is_numeric(self, x):
        return pd.api.types.is_numeric_dtype(x)

    def keep_uniq(self, min_count=10):
        " Create a new dataFrame, keep only columns having more than 'min_count' unique values "
        df_new = pd.DataFrame()
        for c in self.df.columns:
            xi = self.df[c]
            if not self.is_numeric(xi):
                continue
            if len(xi.unique()) >= min_count:
                df_new[c] = xi
        return df_new

    def nas(self):
        " Analysis of missing values "
        if not self.is_nas:
            self._info(f"Missing values analysis {self.name}: Disabled, skipping")
            return
        # Number of missing values, sorted
        count_nas = self.df.isna().to_numpy().sum()
        if count_nas == 0:
            self._info(f"Missing values analysis {self.name}: There are no missing values, skipping")
            return
        self._info(f"Missing values analysis {self.name}")
        nas_count = self.df.isna().sum().sort_values(ascending=False)
        nas_perc = nas_count / len(self.df)
        keep = nas_count > 0
        self.dfnas = pd.DataFrame({'count': nas_count[keep], 'percent': nas_perc[keep]})
        self.print_all(f"Missing by column {self.name}", self.dfnas)
        self._save_csv(f"{self.files_base}.missing_values.csv", "Missing values", self.dfnas, save_index=True)
        # Show plot of percent of missing values
        plt.plot(nas_perc)
        self._plot_show(f"Percent of missing values", f'dataset_explore.{self.name}')
        # Missing values plots
        self.na_plots(self.df, self.name)
        # TODO: Remove. Overall nullity plot should be enough
        # if len(self.df.select_dtypes(include=[np.number]).columns) != len(self.df.columns):
        # # Create a plot of missing values: Only numeric types
        #     self.na_plots(self.df.select_dtypes(include=[np.number]), f"{self.name}: numeric")

    def na_plots(self, df, name):
        " Plot missing values "
        # Show missing values in data frame
        msno.matrix(df)
        count_vars = len(df.columns)
        self._plot_show(f"Missing value dataFrame plot", f'dataset_explore.{self.name}', count_vars_x=count_vars)
        # Barplot of number of misisng values
        try:
            msno.bar(df)
            self._plot_show(f"Missing value by column", f'dataset_explore.{self.name}', count_vars_x=count_vars)
        except ValueError as ve:
            self._debug(f"Exception when invoking missingno.bar: {ve}")
        # Heatmap: Correlation of missing values
        msno.heatmap(df)
        self._plot_show(f"Nullity correlation", f'dataset_explore.{self.name}', count_vars_x=count_vars, count_vars_y=count_vars)

    def plots_pairs(self):
        if not self.is_plot_pairs:
            return
        df_copy = self.remove_na_cols(self.remove_non_numeric_cols(self.df))
        count_cols = len(df_copy.columns)
        if count_cols == 0:
            self._debug(f"Plot pairs {self.name}: No columns left after removing missing values ({count_cols}), skipping")
            return
        if count_cols > self.plot_pairs_max:
            self._debug(f"Plot pairs {self.name}: Too many columns to compare ({count_cols}), skipping")
            return
        print(f"Plotting pairs for columns {self.name}: {df_copy.columns}")
        sns.set_style('darkgrid')
        sns.set()
        sns.pairplot(df_copy, kind='scatter', diag_kind='kde')
        self._plot_show("Pairs", f'dataset_explore.{self.name}', count_vars_x=count_cols, count_vars_y=count_cols)

    def print_all(self, msg, df):
        print(msg)
        self._display(df)

    def rank_correlation(self):
        " Rank correlation (Spearman's R)"
        if self.rank_correlation_matrix is None:
            # Drop columns having zero variance and non-numeric
            df_copy = self.remove_zero_std_cols(self.remove_non_numeric_cols(self.df))
            # Calculate spearsman's correlation
            self._debug(f"Calculating rank correlation matrix for '{self.name}': shape={df_copy.shape}")
            self.rank_correlation_matrix = self._spearmanr(df_copy)
            self.rank_correlation_colums = df_copy.columns
        if self.rank_correlation_matrix is None:
            return None, None
        return self.rank_correlation_matrix.correlation, self.rank_correlation_colums

    def remove_na_cols(self, df):
        """ Remove 'na' columns """
        # Remove column names ending with '_na' or '_nan'
        df_copy = df.copy()
        cols_na = [c for c in df.columns if df[c].isna().sum() > 0 or c.endswith('_na') or c.endswith('_nan')]
        if cols_na:
            df_copy.drop(cols_na, inplace=True, axis=1)
        return df_copy

    def remove_non_numeric_cols(self, df):
        """ Return a new dataFrame with only numeric columns """
        to_drop = list()
        for c in df.columns:
            if not self.is_numeric(df[c]):
                to_drop.append(c)
        df_copy = df.drop(to_drop, axis=1) if to_drop else df.copy()
        return df_copy

    def remove_zero_std_cols(self, df, std_threshold=0.0):
        """ Return a new dataFrame with numeric columns having stdev < std_threshold removed """
        to_drop = list()
        for c in df.columns:
            if not self.is_numeric(df[c]):
                continue
            stdev = df[c].std()
            if stdev <= std_threshold:
                self._debug(f"Remove low std columns {self.name}: Dropping column '{c}': stdev {stdev}")
                to_drop.append(c)
        df_copy = df.drop(to_drop, axis=1) if to_drop else df.copy()
        return df_copy

    def save(self):
        """ Save as pickle file and save CSV tables """
        self._save_pickle(f"{self.files_base}.data_explore.pkl", "Data explore", self)

    def _spearmanr(self, x):
        try:
            return scipy.stats.spearmanr(x, nan_policy='omit')
        except ValueError as e:
            self._error(f"Error calculating Spearman's R, with nan_policy='omit': {e}")
        try:
            return scipy.stats.spearmanr(x, nan_policy='propagate')
        except IndexError as e:
            self._error(f"Error calculating Spearman's R, with nan_policy='propagate': {e}")
            return None

    def summary(self):
        " Look into basic column statistics"
        if not self.is_summary:
            return
        # Look at the data
        self.print_all(f"Head {self.name}", self.df.head())
        self.print_all(f"Tail {self.name}", self.df.tail())
        self.print_all(f"Summary statistics {self.name}", self.df.describe())
        # Numeric / non-numeric columns
        nums = self.df.select_dtypes(include=[np.number]).columns
        non_nums = self.df.select_dtypes(include=[np.object]).columns
        print(f"Numeric columns {self.name}: {sorted(list(nums))}")
        print(f"Non-numerical (e.g. categorical) columns {self.name}: {sorted(list(non_nums))}")
