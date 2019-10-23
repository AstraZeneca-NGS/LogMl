#!/bin/bash -eu
set -o pipefail

DIR=$(cd $(dirname "$0")/..; pwd)
clear

PS1=""
source ./bin/activate

rm -rvf data/class3/class3.pkl data/class3/model

# Run LogMl
time ./src/class3.py -d -c config/class3.yaml 2>&1 | tee run_class3.out
