#!/bin/bash -eu
set -o pipefail

clear

module purge
module load Tensorflow/1.13.1-foss-2017a-Python-3.7.2

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
