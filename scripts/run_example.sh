#!/bin/bash -eu
set -o pipefail

NAME="example_03"

clear

PS1=""
source ./bin/activate

mkdir -p data/$NAME/train
rm -rvf data/$NAME/train/*
rm -vf data/$NAME/*.pkl

time ./src/$NAME.py -v -c config/$NAME.yaml 2>&1 | tee $NAME.out
