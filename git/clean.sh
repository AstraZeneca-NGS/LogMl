#!/bin/bash -eu

# Cleanup files from local file system befor commit to repo
rm -rvf tests/tmp/

rm -vf `find . -iname "*.pkl" -type f`
rm -vf `find . -iname "*.stdout" -type f`
rm -vf `find . -iname "*.stderr" -type f`
rm -vf tests/integration/model/*
