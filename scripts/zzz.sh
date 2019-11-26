#!/bin/bash -eu
set -o pipefail

# Tets script 'zzz'

DIR=$(cd $(dirname "$0")/..; pwd)
clear

PS1=""
source ./bin/activate

rm -rvf data/zzz/zzz.pkl data/zzz/model

# Run LogMl
time ./src/logml.py -v -c config/zzz.yaml 2>&1 | tee zzz.out
