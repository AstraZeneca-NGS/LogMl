#!/bin/bash -eu
set -o pipefail

DIR=$(cd $(dirname "$0")/..; pwd)
clear

PS1=""
source ./bin/activate

rm -rvf data/linear3c/linear3c.pkl data/linear3c/model

# Run LogMl
time ./src/linear3c.py -d -v -c config/linear3c.yaml 2>&1 | tee run_linear3c.out
