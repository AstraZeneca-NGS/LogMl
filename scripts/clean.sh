#!/bin/bash -eu
set -o pipefail

rm -rvf data/linear*/*.pkl data/linear*/model
rm -rvf data/class*/*.pkl data/class*/model
