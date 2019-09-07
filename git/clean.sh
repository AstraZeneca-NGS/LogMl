#!/bin/bash -eu

# Cleanup files from local file system befor commit to repo
rm -rvf tests/tmp/

rm -vf `find . -iname "*.pkl" -type f`
