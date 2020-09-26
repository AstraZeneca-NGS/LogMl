#!/bin/bash -eu
set -o pipefail

# Tets script 'zzz' with scatter / gather
clear

# Clean up LogMl files
rm -rvf data/zzz/*.pkl
rm -rvf data/zzz/zzz.feature_
rm -rvf data/zzz/zzz.preproc_*
rm -rvf data/zzz/zzz.tree_*
rm -rvf data/zzz/model
rm -rvf logml_plots/
rm -rvf logml.bds.* *.chp

# Clean up scatter / gather files
rm -rvf logml.scatter.*
rm -rvf scatter_*_*

## Run LogMl: bds
time ./src/logml.bds -v -config config/zzz.yaml 2>&1 | tee zzz.out

echo "Exit code: $?"
