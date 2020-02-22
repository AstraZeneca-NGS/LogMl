#!/bin/bash -eu

# Cleanup files from local file system befor commit to repo
rm -rvf tests/tmp/

rm -vf `find . -iname "*.pyc" -type f`
rm -vf `find . -iname "*.pkl" -type f`
rm -vf `find . -iname "*.stdout" -type f`
rm -vf `find . -iname "*.stderr" -type f`
rm -vf tests/integration/model/*
rm -vf tests/integration/data/*.dot
rm -vf tests/integration/data/*.png
rm -vf tests/unit/data/*.preproc_augment.csv
rm -vf tests/unit/data/*.feature_importance*.csv
