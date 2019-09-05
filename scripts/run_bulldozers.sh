#!/bin/bash -eu
set -o pipefail

DIR=$(cd $(dirname "$0")/..; pwd)
clear

PS1=""
source ./bin/activate

rm -rvf data/bulldozers/bulldozers.pkl data/bulldozers/model

# Create "small" dataset
# cd data/bulldozers
# (head -n 1 TrainAndValid.csv ; tail -n 50000 TrainAndValid.csv) > TrainAndValid_small.csv
# ln -s TrainAndValid_small.csv bulldozers.csv
# cd -

# Run LogMl
time ./logml/bulldozers.py -v -c bulldozers.yaml 2>&1 | tee run_bulldozers.out
