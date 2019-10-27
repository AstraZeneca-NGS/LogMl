#!/bin/bash -eu
set -o pipefail

DIR=$(cd $(dirname "$0")/..; pwd)
clear

PS1=""
source ./bin/activate

rm -rvf data/linear100c/linear100c.pkl data/linear100c/model

# Run LogMl
time ./src/linear100c.py -v -c config/linear100c.yaml 2>&1 | tee run_linear100c.out
