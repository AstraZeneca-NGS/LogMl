#!/bin/bash -eu
set -o pipefail

# Tets script 'zzz'
clear
# Clean up LogMl files
rm -rvf data/zzz/*.pkl
rm -rvf data/zzz/zzz.feature_
rm -rvf data/zzz/zzz.preproc_*
rm -rvf data/zzz/zzz.tree_*
rm -rvf data/zzz/model
rm -rvf logml_plots/
rm -rvf logml.bds.* *.chp

## Clean up scatter / gather files
#rm -rvf logml.scatter.*
#rm -rvf scatter_*_*

### Run LogMl: Python
PS1=''
source bin/activate
#time ./src/logml.py -d -v -c config/zzz.yaml 2>&1 | tee zzz.out
./src/logml.py --verbose --config 'config/zzz.yaml' --scatter_total 8 --scatter_num 'gather' 2>&1 | tee zzz.out

echo "Exit code: $?"
