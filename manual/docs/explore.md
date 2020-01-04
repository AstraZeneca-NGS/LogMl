
# Explore (df)

These steps implement data exploration when dataset is a DataFrame.

The exploration step is run twice:

- Original (raw) dataset
- Pre-processed dataset

With both analysis you can check whether the pre-processing steps had the desired results, such as properly imputing missing values, or normalizing the data

All missing values are imputed on the pre-processed dataset, so any analysis related to missing values is only performed on the original (raw) dataset.


### Summary statistics

- Show DataFrame head and tail (firs and last lines)
- Summary statistics: count, mean, std, min, 25%, 50%, 75%, max

![Missing analysis](img/summary_stats.png)

### Missing data analysis

- Number of missing values
- Plot: Percent of missing values
- Chart: DataFrame missingness
- Plot: Missing by column
- HeatMap: Missingness correlation

![Missing analysis](img/missing_analysis.png)

### Field description

- Field statistics
- Skewness, Kurtosis
- Normality test and p-value
- Log-Normality test and p-value
- Histogram and kernel density estimate

![Normality analysis](img/normality_test.png)
![Distribution x1](intro/logml_plots/dataset_explore.transformed.Distribution_x1.png)

### Pair plots: scatter plot of variables pairs (up to `plot_pairs_max`, default 20)

![Pairs](intro/logml_plots/dataset_explore.transformed.Pairs.png)

### Correlation analysis

- Rank correlation (Spearsman)
- Top correlations, show correlation over `corr_thresdld` (default 0.7)
- Save top correlations as CSV file
- HeatMap
- Dendogram
- Dendogram of missing values

![Correlation analysis](img/corr_analysis.png)
