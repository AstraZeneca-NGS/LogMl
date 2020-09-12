#!/bin/bash -eu
set -o pipefail

# Tets script 'zzz'
clear
rm -rvf data/zzz/zzz.pkl data/zzz/model

# Run LogMl
time ./src/logml.bds -d -v -config config/zzz.yaml 2>&1 | tee zzz.out

echo "Exit code: $?"
