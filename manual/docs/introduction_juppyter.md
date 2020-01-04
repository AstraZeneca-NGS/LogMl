
# Running from Jupyter Notebooks

In this introduction, we show an example on how to run `LogMl` from a Jupyter Notebook

The source notebook can be found in `notebooks/intro_bare.ipynb` (source code from GitHub)

### Results

[Click here](intro_bare.html) to see LogMl's output. The results include:

- Dataset exploration
    - Show dataset (head & tail)
    - Variables statistics (including normally analysis)
    - Variables distributions
    - Pairs plots
    - Correlation analysis
    - Correlation dendogram
- Feature importance
    - Model based:  RandomForest, ExtraTrees, GradientBoosting. Using drop column and permutation analysis
    - Regression models
    - Tree decition graph
    - Weighted rank sum of all methods
- Model search with hyper-parameter tunning
    - Summary of all models, ranked by validation performance

# Running the example

In order to run this example, you'll need to

1. Set up your Jupyter Notebook's `LogMl` environment (if you never did it before)
1. Copy the dataset to your `LogMl` install directory
1. Copy the configuration file to your `LogMl` install directory
1. Run LogMl from a new Jupyter Notebook

### Set up environment variables

In the rest of the examples, we assume that the following variables are set to the corresponding directories:

```
# I checked out LogMl GitHub repository to $HOME/workspace/LogMl
# You need to change this variable accordingly
LOGML_SRC="$HOME/workspace/LogMl"

# I installed LogMl in the default directory: $HOME/logml
# You need to change this variable accordingly
LOGML_INSTALL="$HOME/logml"
```

### Set up your Jupyter Notebooks environment

The first step is to make sure the `LogMl` virtual environment is available when you run Jupyter Notebooks
You can do it by running these commands:

```
# Activate virtual environment
cd $LOGML_INSTALL
. ./bin/activate

# Make sure you have `ipykernel` installed, otherwise you can run the following line
pip install ipykernel

# Add kernel to Jupyter Notebooks
python -m ipykernel install --name=logml
```

### Copy dataset file

The dataset in this example consists of three normally distributed variables (`x1`, `x2`, `x3`) and some random noise (`n`), the output variable (`y`) is calculated as:
```
y = 2 * x1 - 1 * x2 + 0.5 * x3 + 0.1 * n
```
The file `data/intro/intro.csv` (from GitHub repository) is a CSV file with `1,000` samples from the above equation.

Copy the file to your `logml` directory
```
# Create data directory
cd $LOGML_INSTALL
mkdir -p data/intro

# Copy dataset file from the source directory.
cp $LOGML_SRC/data/intro/intro.csv data/intro/intro.csv
```

### Copy configuration file

The YAML configuraton file for this example, is in GitHub's repository `config/intro.yaml`, you can copy it to the `logml` directory

```
$ cd $LOGML_INSTALL
$ mkdir -p config
$ cp $LOGML_SRC/config/intro.yaml config/
```

### Running `LogMl`

To run LogMl, create a new Jupyter notebook, making sure to select `logml` environment.

Then (from the new NoteBook) you typically run a cell to configure some defaults:
```
# Show plots in the notebook
%matplotlib inline

# Set path to use 'src' subdir
import os, sys
from pathlib import Path

logml_src = str(Path(os.getcwd())/'src')
sys.path.append(logml_src)
```

After that, all you need to do is to create a `LogMl` object and run it
```
from logml import *
ml = LogMl('config/intro.yaml')
ml()
```

[Click here to see LogMl's output](intro_bare.html)

[Or here](intro.html) to see the full notebook that creates the dataset and explains more details
