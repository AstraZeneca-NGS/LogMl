#!/bin/bash -eu
set -o pipefail

DIR=$(cd $(dirname "$0")/..; pwd)
clear

PS1=""
source ./bin/activate

rm -rvf data/linear3/linear3.pkl data/linear3/model

# Run LogMl
time ./src/linear3.py -d -v -c config/linear3.yaml 2>&1 | tee run_linear3.out
