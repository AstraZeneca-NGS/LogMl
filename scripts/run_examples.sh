#!/bin/bash -eu
set -o pipefail

clear

PS1=""
source ./bin/activate

for NAME in example_01 example_02 example_03; do
    echo
    echo
    echo "Example '$NAME'"

    mkdir -p data/$NAME/train
    rm -rvf data/$NAME/train/*
    rm -vf data/$NAME/*.pkl

    time ./src/$NAME.py -v -c config/$NAME.yaml 2>&1 | tee $NAME.out
done
