#!/bin/bash -eu
set -o pipefail

# Tets script 'zzz'
clear
rm -rvf data/zzz/zzz.pkl data/zzz/model logml_plots/ logml.scatter.* logml.bds.*

rm -rvf scatter_*_*

# Run LogMl: bds
#time ./src/logml.bds -d -v -config config/zzz.yaml 2>&1 | tee zzz.out

## Run LogMl: Python
PS1=''
source bin/activate
time ./src/logml.py -d -v -c config/zzz.yaml 2>&1 | tee zzz.out

echo "Exit code: $?"
